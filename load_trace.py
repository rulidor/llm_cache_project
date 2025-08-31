from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import random, json
import numpy as np

USER_PROMPTS = 400
REPEAT_RATE = 0.2

# Load LMSYS Chat-1M user prompts (streaming avoids loading all into RAM)
ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def iter_user_prompts(limit=20000):
    n = 0
    for row in ds:
        if row["language"] != "English":
            continue
        for msg in row.get("conversation", []):
            if msg.get("role") == "user":
                yield msg["content"]
                n += 1
                if n >= limit:
                    return

# 1) Sample a base pool of prompts
base = []
for i, p in enumerate(iter_user_prompts(USER_PROMPTS)):  # adjust limit as needed
    base.append(p)

# 2) Embed
emb = model.encode(base, convert_to_numpy=True, normalize_embeddings=True)

# 3) Define NN lookup (naive cosine similarity)
def nn(idx):
    sims = emb @ emb[idx]
    sims[idx] = -1  # avoid self
    j = int(sims.argmax())
    return base[j]

# 4) Build a stream with repeats
stream = []
reuse_distance = 20

for i, p in enumerate(base):
    # normal prompt → no "synthetically_injected" key
    stream.append({"prompt": p})
    if random.random() < REPEAT_RATE:
        neighbor = nn(i)
        insert_at = min(len(stream) + reuse_distance, len(stream))
        # injected prompt → has "synthetically_injected": true
        stream.insert(insert_at, {"prompt": neighbor, "synthetically_injected": True})

# 5) Add sequential IDs and save to JSON
for idx, entry in enumerate(stream):
    entry["id"] = idx

with open("prompt_stream2.json", "w", encoding="utf-8") as f:
    json.dump(stream, f, ensure_ascii=False, indent=2)

print(f"Saved {len(stream)} prompts to prompt_stream.json")
