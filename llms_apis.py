#!/usr/bin/env python3
"""
batch_hf_inference_first5.py
----------------------------
Process only the first 5 rows of a JSONL file and
query Gemini-Pro, Qwen3-235B, and Llama-4-Scout-17B
using the OpenAI client with HuggingFace models.
"""
import os, json, sys
from pathlib import Path
from dotenv import load_dotenv
from rich.progress import Progress

import requests
from openai import OpenAI   # pip install openai>=1.14.0

GEMINI_API_KEY="-------------"
HF_TOKEN="-------------"
# ------------------------------------------------------------------
# 1.  Build OpenAI client for HuggingFace models
# ------------------------------------------------------------------
#!/usr/bin/env python3

#!/usr/bin/env python3
"""
llms_apis.py
------------
• Reads the FIRST 5 lines of a JSON-Lines file (each must contain an "input" field)
• Queries three models
    – Google Gemini-Pro                 (Generative Language API, v1)
    – Qwen-1.5-7B-Chat                  (HF Inference, ≤10 GB)
    – Mistral-7B-Instruct-v0.2          (HF Inference, ≤10 GB)
• Adds their answers as  gemini_ans / qwen7b_ans / mistral7b_ans
• Writes <source>.first5.hf.jsonl
----------------------------------------------------------------
!!  Hard-coding secrets makes the file sensitive.
    Keep it OUT of version control and private repos.
----------------------------------------------------------------
pip install openai>=1.14 requests rich
"""


# ──────────────────────────────────────────────────────────────────
import json, sys, requests
from pathlib import Path
from rich.progress import Progress
from openai import OpenAI          # HF Inference is OpenAI-compatible

# 1.  HF Inference client (OpenAI-style)
hf_client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1",
    api_key=HF_TOKEN,
)
# ----------  replace the model list  ---------- #
HF_MODELS = {
    "qwen4b_ans" : "Qwen/Qwen1.5-4B-Chat",                   # 8 GB ✅
    "phi3mini_ans": "microsoft/phi-3-mini-4k-instruct",      # 7 GB ✅
}
# --------

def ask_hf(model_id: str, prompt: str) -> str:
    resp = hf_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

# 2.  Gemini helper
# ------------------------------------------------------------------
GEMINI_FLASH_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.0-flash:generateContent"
)

def ask_gemini(prompt: str) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    r = requests.post(
        GEMINI_FLASH_URL,
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

# 3.  Main enrichment (first 5 rows)
def enrich_first_five(src_path: str) -> None:
    src = Path(src_path)
    dst = src.with_suffix(".10to12_krishnav.hf.jsonl")

    lines = src.read_text().splitlines()[10000:10500]
    if not lines:
        print("⚠️  No records found.")
        return

    with dst.open("w") as fout, Progress() as bar:
        task = bar.add_task("Querying models", total=len(lines))

        for line in lines:
            rec    = json.loads(line)
            prompt = rec["input"]

            # Gemini
            try:
                rec["gemini_ans"] = ask_gemini(prompt)
            except Exception as e:
                rec["gemini_ans"] = f"[ERROR] {e}"

            # Hugging Face models
            for key, model in HF_MODELS.items():
                try:
                    rec[key] = ask_hf(model, prompt)
                except Exception as e:
                    rec[key] = f"[ERROR] {e}"

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            bar.update(task, advance=1)

    print(f"✅  Saved → {dst}")

# 4.  CLI (only need the source path)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage:  python llms_apis.py  <source.jsonl>")
    enrich_first_five(sys.argv[1])
