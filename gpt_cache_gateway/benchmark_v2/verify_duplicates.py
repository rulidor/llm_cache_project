# verify_duplicates.py
import json, unicodedata, re
from collections import defaultdict

def canon(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = (s.replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("–", "-").replace("—", "-"))
    return s

with open("prompt_stream.json", "r", encoding="utf-8") as f:
    items = json.load(f)

first_idx = {}
dups_total = 0
dups_after_first = 0
for i, it in enumerate(items):
    p = canon(it["prompt"])
    if p in first_idx:
        dups_total += 1
        if first_idx[p] < i:
            dups_after_first += 1
    else:
        first_idx[p] = i

print("duplicates total:", dups_total)
print("duplicates after first occurrence:", dups_after_first)
