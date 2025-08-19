from datasets import load_dataset
import json

# Load conversations from LMSYS-Chat-1M
dataset = load_dataset("lmsys/lmsys-chat-1m", split="train[:700]")

conversations = []

for entry in dataset:
    for turn in entry.get("conversation", []):
        role = turn.get("role")
        if role == "assistant":
            continue
        content = turn.get("content")
        if role and content:
            conversations.append({"prompt": content})

# Save data into a json file
with open("datasets/lmsys_chat_1m_subset_conversations.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

print(f"Saved {len(conversations)} structured conversations to file.")
