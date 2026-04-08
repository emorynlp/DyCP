"""
DyCP Gradio Web Demo using SCM4LLMs dataset.

Usage:
    python demo_app.py --openai_api_key {openai_api_key} --hf_token {hf_token}
"""

import argparse
import json
import os
import random

import gradio as gr
from openai import OpenAI

from dycp import DyCPRetriever

# ── Data paths ────────────────────────────────────────────────────────────────
QUESTIONS_PATH = "SCM4LLMs/annotation_data/dialogue/dialogue_en_questions.json"
DIALOGUES_PATH = "SCM4LLMs/data/dialogue/en.json"

# ── Load data once at startup ─────────────────────────────────────────────────
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    _all_questions = json.load(f)

with open(DIALOGUES_PATH, "r", encoding="utf-8") as f:
    _all_dialogues = {d["id"]: d for d in json.load(f)}

_questions_by_dialogue: dict = {}
for q in _all_questions:
    _questions_by_dialogue.setdefault(q["id"], []).append(q["question"])

DIALOGUE_IDS = sorted(_all_dialogues.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_dialogue_history(dialogue_id: str) -> str:
    turns = _all_dialogues[dialogue_id]["dialogue"]
    lines = ['**Previous Dialogue History**\n\n']
    for i in range(0, len(turns) - 1, 2):
        user = turns[i].strip()
        asst = turns[i + 1].strip()
        lines.append(f"**[Turn {i//2 + 1}]**\n")
        lines.append(f"🧑 {user}\n")
        lines.append(f"🤖 {asst}\n")
        lines.append("")
    return "\n".join(lines)


def pop_placeholder(remaining: list) -> tuple:
    if not remaining:
        return "Ask a follow-up question...", []
    remaining = remaining.copy()
    idx = random.randrange(len(remaining))
    question = remaining[idx]
    remaining[idx] = remaining[-1]
    remaining.pop()
    return question, remaining


def format_retrieved_turns(retrieved: list, sep: str) -> str:
    if not retrieved:
        return "_No relevant turns retrieved._"
    lines = []
    for turn_idx, turn in retrieved:
        parts = turn.split(sep)
        q = parts[0].strip() if len(parts) > 0 else ""
        a = parts[1].strip() if len(parts) > 1 else ""
        lines.append(f"**[Turn {turn_idx + 1}]** 🧑 {q[:120]}{'...' if len(q) > 120 else ''}\n")
        lines.append(f"🤖 {a[:120]}{'...' if len(a) > 120 else ''}")
        lines.append("")
    return "\n".join(lines)


# ── Load + pre-embed ──────────────────────────────────────────────────────────
def on_dialogue_select(dialogue_id: str, hf_token: str):
    _hf_token = hf_token or os.environ.get("HF_TOKEN", "")
    gr.Info(f"Loading {dialogue_id}")

    retriever = DyCPRetriever(hf_token=_hf_token)
    turns = _all_dialogues[dialogue_id]["dialogue"]
    n_turns = len(turns) // 2
    for i in range(0, len(turns) - 1, 2):
        retriever.add_turn(turns[i], turns[i + 1])

    history_md = format_dialogue_history(dialogue_id)
    status_md = (
        f"**{dialogue_id}** loaded · **{n_turns} turns** pre-embedded "
    )

    qs = _questions_by_dialogue.get(dialogue_id, [])
    remaining = qs.copy()
    random.shuffle(remaining)
    first_placeholder, remaining = pop_placeholder(remaining)

    return (
        history_md,
        status_md,
        first_placeholder,
        retriever,
        remaining,
        [],
        "_Load a dialogue and ask a question._",
    )


# ── LLM ───────────────────────────────────────────────────────────────────────
def build_messages(retrieved: list, sep: str, query: str) -> list:
    messages = []
    for _, turn in retrieved:
        parts = turn.split(sep)
        q = parts[0].strip() if len(parts) > 0 else ""
        a = parts[1].strip() if len(parts) > 1 else ""
        if q:
            messages.append({"role": "user", "content": q})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    return messages


def stream_llm(client: OpenAI, model: str, messages: list):
    response = client.chat.completions.create(
        model=model, messages=messages, stream=True,
    )
    for event in response:
        for chunk in event.choices:
            if chunk.delta.content:
                yield chunk.delta.content


# ── Respond ───────────────────────────────────────────────────────────────────
def respond(
    user_input: str,
    chat_history: list,
    retriever_state,
    remaining_qs: list,
    llm_model: str,
    openai_api_key: str,
    hf_token: str,
):
    if retriever_state is None:
        yield chat_history, "_Please load a dialogue first._", retriever_state, remaining_qs, ""
        return
    if not user_input.strip():
        yield chat_history, "", retriever_state, remaining_qs, ""
        return

    client = OpenAI(api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"))
    retriever = retriever_state
    sep = retriever.embedder.sep_token

    retrieved = retriever.retrieve(user_input)
    retrieved_md = (
        f"**Retrieved {len(retrieved)} / {len(retriever.turns)} turns**\n\n"
        + format_retrieved_turns(retrieved, sep)
    )

    messages = build_messages(retrieved, sep, user_input)

    answer = ""
    chat_history = chat_history + [{"role": "user", "content": user_input},
                                    {"role": "assistant", "content": ""}]
    for token in stream_llm(client, llm_model, messages):
        answer += token
        chat_history[-1] = {"role": "assistant", "content": answer}
        yield chat_history, retrieved_md, retriever_state, remaining_qs, ""

    retriever.add_turn(user_input, answer)
    next_placeholder, remaining_qs = pop_placeholder(remaining_qs)
    yield chat_history, retrieved_md, retriever_state, remaining_qs, next_placeholder


# ── UI ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

h1.demo-title { font-size: 1.7rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }

.label-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #555;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px;
}

/* Settings panel */
.settings-panel {
    background: #ffffff !important;
    border: 1px solid #e2e4ed !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-bottom: 8px !important;
}

.settings-panel > .wrap {
    align-items: flex-end !important;
}

/* History panel */
.history-panel {
    background: #ffffff !important;
    border: 1px solid #e2e4ed !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    max-height: 560px;
    overflow-y: auto;
    overflow-x: auto;
}
 
.history-panel * {
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    white-space: normal !important;
}

/* Chat panel */
.chat-panel {
    background: #ffffff !important;
    border: 1px solid #e2e4ed !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}

.retrieved-box {
    border-left: 3px solid #6366f1 !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    background: #f8f8ff !important;
}

#send-btn {
    background: #6366f1 !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 600;
}
#send-btn:hover { background: #4f46e5 !important; }

/* Input textbox border */
.chat-panel textarea {
    border: 1.5px solid #d0d3e0 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
.chat-panel textarea:focus {
    border-color: #6366f1 !important;
    outline: none !important;
}

#load-btn {
    max-width: 100px !important;
    min-width: 80px !important;
    height: 70px !important;
    margin-top: 30px;
    padding: 0px 12px !important;
    font-size: 0.85rem !important;
}

"""


def build_app(openai_api_key_default: str = "", hf_token_default: str = ""):
    with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="DyCP Demo") as demo:

        retriever_state    = gr.State(None)
        remaining_qs_state = gr.State([])
        openai_key_input = gr.State(openai_api_key_default or os.environ.get("OPENAI_API_KEY", ""))
        hf_token_input   = gr.State(hf_token_default or os.environ.get("HF_TOKEN", ""))

        with gr.Row(elem_classes=["settings-panel"]):
            with gr.Column(scale=2):
                gr.HTML("""
                <div style="padding:4px 0;">
                    <h1 class="demo-title">DyCP</h1>
                    <p style="color:#6b7080;font-size:0.82rem;font-family:'JetBrains Mono',monospace;margin:2px 0 0 0;">
                        using SCM4LLMs dataset: https://arxiv.org/pdf/2304.13343
                    </p>
                </div>
                """)
            with gr.Column(scale=1):
                dialogue_selector = gr.Dropdown(
                    choices=DIALOGUE_IDS, value=DIALOGUE_IDS[4],
                    label="Dialogue ID",
                    info="Pick a long dialogue from SCM4LLMs",
                )
            with gr.Column(scale=1):
                llm_model_input = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o"],
                    value="gpt-4o-mini", label="LLM Model",
                    info="Choose a backbone LLM to use",
                )
            load_btn = gr.Button("Load", variant="primary", elem_id="load-btn")
        status_display = gr.Markdown(value="")

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, elem_classes=["history-panel"]):
                #gr.HTML('<p class="label-text">Previous Dialogue History</p><hr style="border:none;border-top:1px solid #e2e4ed;margin:8px 0 12px 0;">')
                history_display = gr.Markdown(
                    value="_Previous Dialogue History<br>(Select a dialogue and click Load)_",
                )

            with gr.Column(scale=3, elem_classes=["chat-panel"]):
                gr.HTML('<p class="label-text">Chat</p><hr style="border:none;border-top:1px solid #e2e4ed;margin:8px 0 12px 0;">')
                chatbot = gr.Chatbot(value=[], height=300, show_label=False, type="messages")
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Continue the dialogue...",
                        show_label=False, scale=5, container=False,
                    )
                    send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

                retrieved_display = gr.Markdown(
                    value="_Load a dialogue and ask a question._",
                    elem_classes=["retrieved-box"],
                )
                
                gr.HTML('<p class="label-text" style="margin-top:16px;">DyCP Retrieved Context</p><hr style="border:none;border-top:1px solid #e2e4ed;margin:8px 0 12px 0;">')
                
                
        load_outputs = [
            history_display, status_display, user_input,
            retriever_state, remaining_qs_state, chatbot, retrieved_display,
        ]

        load_btn.click(
            fn=on_dialogue_select,
            inputs=[dialogue_selector, hf_token_input],
            outputs=load_outputs,
        )
        dialogue_selector.change(
            fn=on_dialogue_select,
            inputs=[dialogue_selector, hf_token_input],
            outputs=load_outputs,
        )

        send_outputs = [chatbot, retrieved_display, retriever_state, remaining_qs_state, user_input]

        send_btn.click(
            fn=respond,
            inputs=[user_input, chatbot, retriever_state, remaining_qs_state,
                    llm_model_input, openai_key_input, hf_token_input],
            outputs=send_outputs,
        )
        user_input.submit(
            fn=respond,
            inputs=[user_input, chatbot, retriever_state, remaining_qs_state,
                    llm_model_input, openai_key_input, hf_token_input],
            outputs=send_outputs,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyCP Gradio Web Demo")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(
        openai_api_key_default=args.openai_api_key or "",
        hf_token_default=args.hf_token or "",
    )
    app.launch(server_port=args.port, share=args.share)