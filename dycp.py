import numpy as np
import torch
from embedder import ContrieverEmbedder


def KadaneDial(sim_matrix, tau=0.6, threshold=1.0):
    s = sim_matrix.detach().squeeze().numpy()
    z = (s - s.mean()) / s.std()

    gain = z - tau
    spans = []
    gain = gain.copy()
    while True:
        best_sum = -np.inf
        cur_sum = 0.0
        cur_start = 0
        start = end = 0
        for i in range(len(gain)):
            if cur_sum + gain[i] < gain[i]:
                cur_sum = gain[i]
                cur_start = i
            else:
                cur_sum += gain[i]
            if cur_sum > best_sum:
                best_sum, start, end = cur_sum, cur_start, i
        if best_sum < threshold:
            break
        spans.append((start, end))
        gain[start:end + 1] = -np.inf
    return spans


class DyCPRetriever:
    def __init__(self, model_name="facebook/contriever-msmarco", hf_token="",
                 tau=0.6, threshold=1.0, device=None):
        self.embedder = ContrieverEmbedder(model_name=model_name, hf_token=hf_token, device=device)
        self.tau = tau
        self.threshold = threshold
        self.turns = [] # "q [SEP] a"
        self.turn_embeddings = []

    def add_turn(self, query: str, answer: str):
        text = f"{query} {self.embedder.sep_token} {answer}"
        emb = self.embedder.encode(text)  # (1, dim)
        self.turns.append(text)
        self.turn_embeddings.append(emb)

    def retrieve(self, query: str) -> list[str]:
        if not self.turns:
            return []

        query_emb = self.embedder.encode(query) # (1, dim)
        history_embs = torch.cat(self.turn_embeddings) # (n, dim)
        sim = torch.matmul(history_embs, query_emb.T) # (n, 1)

        spans = KadaneDial(sim, tau=self.tau, threshold=self.threshold)
        spans = sorted(spans, key=lambda x: x[0]) # chronologically ordered

        retrieved = []  # list of (turn_index, turn_text)
        for start, end in spans:
            for idx in range(start, end + 1):
                retrieved.append((idx, self.turns[idx]))
        return retrieved

    def format_context(self, retrieved: list) -> list[str]:
        sep = self.embedder.sep_token
        result = []
        for _, turn in retrieved:
            for t in turn.split(sep):
                result.append(t.strip())
        return result

    def reset(self):
        self.turns = []
        self.turn_embeddings = []