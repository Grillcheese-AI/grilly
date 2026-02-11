"""
Conversations SVC Cleaner

Cleans the conversations_svc_semantic.jsonl file by removing:
- Leaked filenames (``*.py``, ``*.pt``, ``*.txt``, etc.)
- File paths (Unix/Windows)
- Code artifacts (&&, backticks, arrows, shell commands)
- Technical numbers (checkpoint IDs, dimension specs)
- Leaked project names (AURA, STDP, etc.)
- Entries that become too short or invalid after cleaning

Output: conversations_svc_cleaned.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "_data"
INPUT_FILE = DATA_DIR / "conversations_svc_semantic.jsonl"
OUTPUT_FILE = DATA_DIR / "conversations_svc_cleaned.jsonl"

# Patterns to clean
# File extensions
FILE_EXT_PATTERN = re.compile(
    r"\b\w+\.(py|pt|txt|json|jsonl|js|ts|sh|yaml|yml|toml|cfg|ini|md|csv|log|bin|pkl|pth|ckpt|safetensors|onnx|h5|npz|npy)\b",
    re.IGNORECASE,
)

# File paths (Unix and Windows)
UNIX_PATH_PATTERN = re.compile(r"(?:/[\w._-]+){2,}/?")
WINDOWS_PATH_PATTERN = re.compile(r"[A-Z]:\\(?:[\w._-]+\\)*[\w._-]+")

# Code artifacts
CODE_PATTERNS = [
    re.compile(r"&&"),  # shell chaining
    re.compile(r"`[^`]+`"),  # backtick code
    re.compile(r"python3?\s"),  # python commands
    re.compile(r"\bpip\s+install\b"),  # pip install
    re.compile(r"\bcd\s+\S+"),  # cd commands
    re.compile(r"\bgit\s+\w+"),  # git commands
    re.compile(r"\bnpm\s+\w+"),  # npm commands
    re.compile(r"\bsudo\s"),  # sudo
    re.compile(r"\$\{?\w+\}?"),  # shell variables
    re.compile(r"import\s+\w+"),  # python imports
    re.compile(r"from\s+\w+\s+import"),  # python from imports
    re.compile(r"def\s+\w+\s*\("),  # function defs
    re.compile(r"class\s+\w+[:\(]"),  # class defs
]

# Dimension specs and technical numbers
DIMENSION_PATTERN = re.compile(r"\b\d+\s*[\u2192\u2190\u2194→←]\s*\d+")  # 64→128
CHECKPOINT_PATTERN = re.compile(
    r"\b(?:checkpoint|ckpt|epoch|step|line|iteration)[\s_]*[\d,._]+", re.IGNORECASE
)
LARGE_TECHNICAL_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")  # 257,450 etc.

# Leaked project/model names
LEAKED_NAMES = re.compile(
    r"\b(?:AURA|STDP|stdp|phasic|neuromorphic|Vulkan|vulkan|GLSL|glsl|GrillCheese|grilly|grillcheese)\b",
    re.IGNORECASE,
)

# Bracket placeholders already in data - keep these
PLACEHOLDER_PATTERN = re.compile(r"\[(?:PROJECT_NAME|FILE_PATH|VOLUME_PATH|ID_\d+|FILE|PATH|NUM)\]")


def clean_text(text: str) -> str:
    """Apply all cleaning rules to a text string."""
    # Replace file extensions with [FILE]
    text = FILE_EXT_PATTERN.sub("[FILE]", text)

    # Replace file paths
    text = UNIX_PATH_PATTERN.sub("[PATH]", text)
    text = WINDOWS_PATH_PATTERN.sub("[PATH]", text)

    # Remove code artifacts entirely
    for pattern in CODE_PATTERNS:
        text = pattern.sub("", text)

    # Replace dimension specs
    text = DIMENSION_PATTERN.sub("[DIM]", text)

    # Replace checkpoint references
    text = CHECKPOINT_PATTERN.sub("[CHECKPOINT]", text)

    # Replace large technical numbers
    text = LARGE_TECHNICAL_NUMBER.sub("[NUM]", text)

    # Replace leaked project names
    text = LEAKED_NAMES.sub("[PROJECT_NAME]", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Collapse multiple placeholders in a row
    text = re.sub(
        r"(\[(?:FILE|PATH|NUM|DIM|CHECKPOINT|PROJECT_NAME)\]\s*){3,}", "[REDACTED] ", text
    )

    return text


def clean_svc(svc: dict) -> dict:
    """Clean SVC fields."""
    return {
        "s": clean_text(svc.get("s", "")),
        "v": clean_text(svc.get("v", "")),
        "c": clean_text(svc.get("c", "")),
    }


def is_valid_after_cleaning(entry: dict) -> bool:
    """Check if an entry is still valid after cleaning."""
    text = entry.get("text", "")

    # Text must have at least 3 real words (not just placeholders)
    real_words = [w for w in text.split() if not w.startswith("[") and len(w) > 1]
    if len(real_words) < 3:
        return False

    # Verb must still be present
    svc = entry.get("svc", {})
    verb = svc.get("v", "").strip()
    if not verb:
        return False

    return True


def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"Cleaning: {INPUT_FILE.name}")
    print(f"Output:   {OUTPUT_FILE.name}")

    total = 0
    kept = 0
    dropped = 0
    cleaned_count = 0  # entries where text was modified

    realm_counter = Counter()
    drop_reasons = Counter()

    with (
        open(INPUT_FILE, encoding="utf-8") as fin,
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                drop_reasons["json_error"] += 1
                dropped += 1
                continue

            original_text = entry.get("text", "")

            # Clean text and SVC
            entry["text"] = clean_text(entry["text"])
            entry["svc"] = clean_svc(entry.get("svc", {}))

            # Clean lemmas too (may contain leaked names)
            if "lemmas" in entry:
                entry["lemmas"] = [
                    clean_text(l) if LEAKED_NAMES.search(l) else l for l in entry["lemmas"]
                ]

            # Track if we changed anything
            if entry["text"] != original_text:
                cleaned_count += 1

            # Validate after cleaning
            if not is_valid_after_cleaning(entry):
                drop_reasons["too_short_or_invalid"] += 1
                dropped += 1
                continue

            realm_counter[entry.get("realm", "?")] += 1
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1

            if total % 20000 == 0:
                print(f"  Processed {total:,}... kept {kept:,}, dropped {dropped:,}")

    print("\n--- Results ---")
    print(f"  Total input:   {total:,}")
    print(f"  Kept:          {kept:,} ({kept / total * 100:.1f}%)")
    print(f"  Dropped:       {dropped:,} ({dropped / total * 100:.1f}%)")
    print(f"  Modified:      {cleaned_count:,} ({cleaned_count / total * 100:.1f}%)")
    print(f"  Drop reasons:  {dict(drop_reasons)}")
    print(f"  Realms:        {dict(realm_counter.most_common())}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
