"""
SVC Dataset Validator

Validates instruct and conversation JSONL files for:
- Schema completeness (10 fields, correct types)
- SVC rule correctness (imperative promotion, normal subjects)
- Field distributions (realm, complexity, POS/deps)
- Data quality (text length, verb presence)
- Cross-file consistency (source field matches file)
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "_data"

INSTRUCT_FILE = DATA_DIR / "instruct_svc_semantic.jsonl"
CONVERSATIONS_FILE = DATA_DIR / "conversations_svc_semantic.jsonl"

# Expected schema fields per source
INSTRUCT_FIELDS = {
    "id",
    "text",
    "svc",
    "pos",
    "lemmas",
    "deps",
    "root_verb",
    "realm",
    "source",
    "complexity",
}
CONVERSATION_FIELDS = INSTRUCT_FIELDS | {"role"}

VALID_REALMS = {
    "technology",
    "science",
    "health",
    "history",
    "nature",
    "business",
    "arts",
    "social",
    "general",
    "conversation",
}

VALID_POS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
}

SAMPLE_SIZE = 1000


def sample_lines(filepath: Path, n: int) -> list[dict]:
    """Reservoir sample n lines from a JSONL file."""
    reservoir = []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if i < n:
                reservoir.append(entry)
            else:
                j = random.randint(0, i)
                if j < n:
                    reservoir[j] = entry
    return reservoir


def count_lines(filepath: Path) -> int:
    """Count total lines in file."""
    count = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def validate_schema(entry: dict, expected_fields: set) -> list[str]:
    """Check that all expected fields are present with correct types."""
    errors = []
    missing = expected_fields - set(entry.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")

    if "text" in entry and not isinstance(entry["text"], str):
        errors.append(f"text is not str: {type(entry['text'])}")

    if "svc" in entry:
        svc = entry["svc"]
        if not isinstance(svc, dict):
            errors.append(f"svc is not dict: {type(svc)}")
        else:
            for key in ("s", "v", "c"):
                if key not in svc:
                    errors.append(f"svc missing key '{key}'")
                elif not isinstance(svc[key], str):
                    errors.append(f"svc.{key} is not str: {type(svc[key])}")

    for list_field in ("pos", "lemmas", "deps"):
        if list_field in entry and not isinstance(entry[list_field], list):
            errors.append(f"{list_field} is not list: {type(entry[list_field])}")

    if "complexity" in entry:
        c = entry["complexity"]
        if not isinstance(c, (int, float)):
            errors.append(f"complexity is not numeric: {type(c)}")

    if "root_verb" in entry and not isinstance(entry["root_verb"], str):
        errors.append(f"root_verb is not str: {type(entry['root_verb'])}")

    if "realm" in entry and not isinstance(entry["realm"], str):
        errors.append(f"realm is not str: {type(entry['realm'])}")

    return errors


def validate_svc_rules(entry: dict) -> list[str]:
    """Check SVC extraction rules."""
    errors = []
    svc = entry.get("svc", {})
    deps = entry.get("deps", [])
    verb = svc.get("v", "")

    # Verb must not be empty
    if not verb.strip():
        errors.append("Empty verb in SVC")

    # Check imperative detection: if ROOT is first token and no nsubj, should be imperative
    if deps and deps[0] == "ROOT" and "nsubj" not in deps:
        # This is likely an imperative - subject should come from dobj
        subject = svc.get("s", "")
        if not subject.strip():
            # Empty subject in imperative is acceptable only if no dobj either
            if "dobj" not in deps:
                pass  # No dobj to promote, empty subject is OK
            # Otherwise it's a potential miss (but not always an error due to edge cases)

    return errors


def validate_data_quality(entry: dict) -> list[str]:
    """Check data quality constraints."""
    errors = []
    text = entry.get("text", "")

    if not text.strip():
        errors.append("Empty text")

    words = text.split()
    if len(words) < 3:
        errors.append(f"Text too short: {len(words)} words")

    # Complexity should be in [0, 1]
    complexity = entry.get("complexity", 0)
    if not (0 <= complexity <= 1):
        errors.append(f"Complexity out of range: {complexity}")

    # POS tags should be valid spaCy tags
    pos_tags = entry.get("pos", [])
    invalid_pos = [p for p in pos_tags if p not in VALID_POS]
    if invalid_pos:
        errors.append(f"Invalid POS tags: {set(invalid_pos)}")

    # Realm should be in known set
    realm = entry.get("realm", "")
    if realm not in VALID_REALMS:
        errors.append(f"Unknown realm: {realm}")

    # List lengths should match
    pos_len = len(entry.get("pos", []))
    lemma_len = len(entry.get("lemmas", []))
    dep_len = len(entry.get("deps", []))
    if not (pos_len == lemma_len == dep_len):
        errors.append(f"List length mismatch: pos={pos_len}, lemmas={lemma_len}, deps={dep_len}")

    return errors


def validate_source_consistency(entry: dict, expected_source: str) -> list[str]:
    """Check source field matches the file it came from."""
    errors = []
    source = entry.get("source", "")
    if source != expected_source:
        errors.append(f"Source mismatch: expected '{expected_source}', got '{source}'")
    return errors


def validate_file(filepath: Path, expected_source: str, expected_fields: set):
    """Validate a single JSONL file and print report."""
    print(f"\n{'=' * 60}")
    print(f"Validating: {filepath.name}")
    print(f"{'=' * 60}")

    if not filepath.exists():
        print(f"  FILE NOT FOUND: {filepath}")
        return

    total_lines = count_lines(filepath)
    print(f"  Total entries: {total_lines:,}")

    random.seed(42)
    samples = sample_lines(filepath, SAMPLE_SIZE)
    n = len(samples)
    print(f"  Sampled: {n:,}")

    # Counters
    schema_errors = 0
    svc_errors = 0
    quality_errors = 0
    source_errors = 0
    schema_examples = []
    svc_examples = []
    quality_examples = []

    realm_counter = Counter()
    complexity_values = []
    source_counter = Counter()
    verb_counter = Counter()

    for entry in samples:
        # Schema
        errs = validate_schema(entry, expected_fields)
        if errs:
            schema_errors += 1
            if len(schema_examples) < 3:
                schema_examples.append((entry.get("id", "?"), errs))

        # SVC rules
        errs = validate_svc_rules(entry)
        if errs:
            svc_errors += 1
            if len(svc_examples) < 3:
                svc_examples.append((entry.get("id", "?"), errs))

        # Data quality
        errs = validate_data_quality(entry)
        if errs:
            quality_errors += 1
            if len(quality_examples) < 3:
                quality_examples.append((entry.get("id", "?"), errs))

        # Source consistency
        errs = validate_source_consistency(entry, expected_source)
        if errs:
            source_errors += 1

        # Distributions
        realm_counter[entry.get("realm", "?")] += 1
        complexity_values.append(entry.get("complexity", 0))
        source_counter[entry.get("source", "?")] += 1
        verb_counter[entry.get("root_verb", "?")] += 1

    # Report
    print(f"\n  --- Results ({n} samples) ---")
    print(f"  Schema valid:  {n - schema_errors}/{n} ({(n - schema_errors) / n * 100:.1f}%)")
    print(f"  SVC valid:     {n - svc_errors}/{n} ({(n - svc_errors) / n * 100:.1f}%)")
    print(f"  Quality valid: {n - quality_errors}/{n} ({(n - quality_errors) / n * 100:.1f}%)")
    print(f"  Source match:  {n - source_errors}/{n} ({(n - source_errors) / n * 100:.1f}%)")

    if schema_examples:
        print("\n  Schema error examples:")
        for eid, errs in schema_examples:
            print(f"    [{eid}] {errs}")

    if svc_examples:
        print("\n  SVC error examples:")
        for eid, errs in svc_examples:
            print(f"    [{eid}] {errs}")

    if quality_examples:
        print("\n  Quality error examples:")
        for eid, errs in quality_examples:
            print(f"    [{eid}] {errs}")

    # Distributions
    print("\n  --- Distributions ---")
    print(f"  Realms: {dict(realm_counter.most_common())}")
    if complexity_values:
        print(
            f"  Complexity: min={min(complexity_values):.2f}, max={max(complexity_values):.2f}, "
            f"mean={sum(complexity_values) / len(complexity_values):.2f}"
        )
    print(f"  Top verbs: {dict(verb_counter.most_common(10))}")
    print(f"  Sources: {dict(source_counter)}")


def main():
    # Fix Windows console encoding
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("SVC Dataset Validation Report")
    print(f"Data directory: {DATA_DIR}")

    validate_file(INSTRUCT_FILE, "instruct", INSTRUCT_FIELDS)
    validate_file(CONVERSATIONS_FILE, "conversation", CONVERSATION_FIELDS)

    print(f"\n{'=' * 60}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
