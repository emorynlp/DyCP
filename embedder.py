import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class ContrieverEmbedder:
    def __init__(self, model_name="facebook/contriever-msmarco", hf_token="", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModel.from_pretrained(model_name, token=hf_token, use_safetensors=True).to(self.device)
        self.model.eval()

    @property
    def sep_token(self):
        return self.tokenizer.sep_token or "[SEP]"

    def _mean_pool(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts: list[str] | str, normalize: bool = True) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        embs = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if normalize:
            embs = F.normalize(embs, dim=-1)
        return embs  # (len(texts), dim)