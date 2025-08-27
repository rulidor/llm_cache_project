# ---- deps (once) ----
# pip install "numpy<2" sentence-transformers requests

import requests
from gptcache.adapter.api import get, put

# ...reuse the SAME GPTCache setup from Option A above (cache = Cache(), etc.)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b-instruct-q4_K_M"

def ollama_generate(prompt: str) -> str:
    # non-streaming call
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=300)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"response": "...", ...}
    return data.get("response", "")

def ask(prompt: str) -> str:
    # 1) try cache
    cached = get(prompt, cache=cache)
    if cached:
        return cached
    # 2) call Ollama
    answer = ollama_generate(prompt)
    # 3) store in cache
    put(prompt, answer, cache=cache)
    return answer

# ---- use it ----
print(ask("Explain BLE advertising briefly."))
print(ask("Explain BLE advertising briefly."))        # should be a cache hit
print(ask("What is BLE advertising?"))                # semantic hit if threshold allows
