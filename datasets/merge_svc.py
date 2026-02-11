"""
SVC Dataset Merger

Merges instruct_svc_semantic.jsonl and conversations_svc_cleaned.jsonl
into a single shuffled training file.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "_data"

INSTRUCT_FILE = DATA_DIR / "instruct_svc_semantic.jsonl"
CONVERSATIONS_FILE = DATA_DIR / "conversations_svc_cleaned.jsonl"
OUTPUT_FILE = DATA_DIR / "svc_training_merged.jsonl"

SEED = 42


def load_jsonl(filepath: Path) -> list[str]:
    """Load all lines from a JSONL file as raw strings."""
    lines = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("Loading instruct data...")
    instruct_lines = load_jsonl(INSTRUCT_FILE)
    print(f"  Loaded {len(instruct_lines):,} entries")

    print("Loading cleaned conversations data...")
    conv_lines = load_jsonl(CONVERSATIONS_FILE)
    print(f"  Loaded {len(conv_lines):,} entries")

    # Merge
    all_lines = instruct_lines + conv_lines
    total = len(all_lines)
    print(f"\nTotal: {total:,} entries")

    # Shuffle with deterministic seed
    print(f"Shuffling with seed={SEED}...")
    random.seed(SEED)
    random.shuffle(all_lines)

    # Write output
    print(f"Writing to {OUTPUT_FILE.name}...")
    realm_counter = Counter()
    source_counter = Counter()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")
            try:
                entry = json.loads(line)
                realm_counter[entry.get("realm", "?")] += 1
                source_counter[entry.get("source", "?")] += 1
            except json.JSONDecodeError:
                pass

    print("\n--- Merged Stats ---")
    print(f"  Total entries:  {total:,}")
    print(f"  Source split:   {dict(source_counter)}")
    print(f"  Realm dist:     {dict(realm_counter.most_common())}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
