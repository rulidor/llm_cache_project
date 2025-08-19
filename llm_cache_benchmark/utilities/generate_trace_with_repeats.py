import random
import json

def generate_repeated_prompts_from_json(input_file: str, output_file: str,
                                         n_unique=300, repeats=2, total=450):
    """
    Load prompts from a trace JSON file, select a subset, and create repeated entries.

    Parameters:
    - input_file (str): Path to the input JSON file containing prompts in {"prompt": "..."} format.
    - output_file (str): Path where the repeated prompts will be saved.
    - n_unique (int): Number of unique prompts to repeat.
    - repeats (int): How many times each unique prompt should be repeated.
    - total (int): Total number of prompts to save after shuffling.

    The final result will be written to `output_file` as a list of dictionaries with "prompt" keys.
    """
    # Load the original prompt file
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Extract all "prompt" values
    all_prompts = [entry["prompt"] for entry in raw_data if "prompt" in entry]

    if len(all_prompts) < n_unique:
        raise ValueError(f"Only {len(all_prompts)} prompts found, but n_unique={n_unique} requested.")

    # Take the first n_unique prompts
    unique_prompts = all_prompts[:n_unique]

    # Repeat each prompt `repeats` times
    repeated = []
    for prompt in unique_prompts:
        repeated.extend([prompt] * repeats)

    # Shuffle and truncate to total number of prompts
    random.shuffle(repeated)
    final = repeated[:total]

    # Write the result to output JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([{"prompt": p} for p in final], f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final)} repeated prompts to {output_file}")


# Example usage
generate_repeated_prompts_from_json(
    input_file="../datasets/lmsys_chat_1m_subset_conversations.json",
    output_file="../datasets/lmsys_trace_with_repeats.json",
    n_unique=700,
    repeats=3,
    total=1000
)
