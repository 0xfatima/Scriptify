from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./model"


@dataclass(frozen=True)
class _ModelBundle:
    tokenizer: any
    model: any
    device: str


_BUNDLE: Optional[_ModelBundle] = None


def get_bundle() -> _ModelBundle:
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    _BUNDLE = _ModelBundle(tokenizer=tokenizer, model=model, device=device)
    return _BUNDLE


def _chat_generate(system: str, user: str, *, max_new_tokens: int) -> str:
    b = get_bundle()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = b.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = b.tokenizer(text, return_tensors="pt").to(b.model.device)
    with torch.no_grad():
        outputs = b.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    input_len = inputs["input_ids"].shape[-1]
    return (
        b.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        .strip()
    )


def generate_writing_assist(*, text: str, mode: str) -> str:
    prompts = {
        "Spell & Grammar": (
            "You are a text corrector. You receive text and return ONLY the corrected version.\n"
            "RULES:\n"
            "- Fix spelling mistakes, grammar errors, and punctuation only.\n"
            "- Do NOT add, remove, or rephrase any sentences.\n"
            "- Do NOT explain, comment, or write anything other than the corrected text.\n"
            "- If already correct, return it exactly as-is.\n"
            "OUTPUT: Only the corrected text, nothing else."
        ),
        "Email": (
            "You are an email formatter. You receive text and rewrite it as a professional email.\n"
            "RULES:\n"
            "- If To/From/Subject are provided, use them. Otherwise infer from context.\n"
            "- Format: **To:** / **From:** / **Subject:** headers, then email body.\n"
            "- Rewrite the user's content into professional tone with greeting and sign-off.\n"
            "- Do NOT add information the user did not provide.\n"
            "- Do NOT explain or add commentary.\n"
            "OUTPUT: Only the formatted email."
        ),
        "Academic": (
            "You are an academic text rewriter. You receive text and rewrite it in scholarly style.\n"
            "RULES:\n"
            "- Fix spelling/grammar and elevate to formal academic English.\n"
            "- Keep the SAME meaning and content. Do NOT add new information.\n"
            "- Use research paper conventions (precise vocabulary, formal tone).\n"
            "- Do NOT add bullet points, notes, explanations, or commentary.\n"
            "OUTPUT: Only the rewritten academic text, nothing else."
        ),
        "LaTeX": (
            "You are a plain-text-to-LaTeX converter. You receive text and convert it to LaTeX.\n"
            "RULES:\n"
            "- Convert the EXACT text given into valid LaTeX markup.\n"
            "- Do NOT add any new content, explanations, stories, or elaboration.\n"
            "- Do NOT include \\documentclass or preamble.\n"
            "- For a title/heading use \\section or \\textbf. For lists use itemize.\n"
            "- If the input is just a phrase, output just that phrase in LaTeX.\n"
            "  Example input: 'the urban legend' → output: \\textbf{The Urban Legend}\n"
            "OUTPUT: Only LaTeX code representing the input text, nothing else."
        ),
    }
    system = prompts.get(mode, prompts["Spell & Grammar"])
    return _chat_generate(system, text, max_new_tokens=350)


def generate_rag_answer(*, query: str, context: str) -> str:
    system = (
        "You are a document QA assistant. You answer questions using ONLY the provided document excerpts.\n\n"
        "FORMAT YOUR ANSWER LIKE THIS:\n"
        "Write a clear, well-formatted answer in your own words (do NOT copy chunks verbatim).\n"
        "After each claim, add an inline citation: (FileName, p. X).\n"
        "At the end, add a References section listing all cited sources.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY facts from the provided excerpts. Never invent information.\n"
        "- Keep answers concise: 3-6 sentences.\n"
        "- Do NOT output raw chunk text. Synthesize and summarize.\n"
        "- Do NOT list chunk numbers or say 'Chunk 1 says...'.\n\n"
        "EXAMPLE OUTPUT:\n"
        "Backdoor attacks inject hidden triggers into training data (paper.pdf, p. 2). "
        "Two main types exist: poison-label and clean-label attacks (paper.pdf, p. 4).\n\n"
        "**References:**\n"
        "- paper.pdf, p. 2\n"
        "- paper.pdf, p. 4"
    )
    user = f"{context}\n\n---\nQuestion: {query}"
    return _chat_generate(system, user, max_new_tokens=400)

