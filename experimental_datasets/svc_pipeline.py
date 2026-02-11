"""
Lean SVC Pipeline for Grilly Training Data
Uses spaCy for sentence splitting and POS/dependency parsing
Output: ~10 fields per entry instead of 50+
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


@dataclass
class SVCResult:
    """Represent svcresult behavior."""

    subject: str
    verb: str
    complement: str
    valid: bool


def extract_svc(doc) -> SVCResult:
    """Extract Subject-Verb-Complement from spaCy Doc."""
    subject_parts = []
    verb_parts = []
    complement_parts = []

    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if not root or root.pos_ not in ("VERB", "AUX"):
        return SVCResult("", "", "", False)

    # Collect verb phrase
    for token in doc:
        if token == root or token.head == root:
            if token.dep_ in ("aux", "auxpass", "neg") or token == root:
                verb_parts.append((token.i, token.text))

    # Collect subject
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass", "expl"):
            for t in token.subtree:
                if t.i < root.i:
                    subject_parts.append((t.i, t.text))

    # Complement = everything after verb
    verb_end = max([i for i, _ in verb_parts]) if verb_parts else root.i
    for token in doc:
        if token.i > verb_end and token.dep_ != "punct":
            complement_parts.append((token.i, token.text))

    subject = " ".join(t for _, t in sorted(subject_parts))
    verb = " ".join(t for _, t in sorted(verb_parts))
    complement = " ".join(t for _, t in sorted(complement_parts))

    return SVCResult(subject, verb, complement, bool(verb and len(doc) >= 3))


def classify_realm(text: str) -> str:
    """Simple keyword-based realm classification."""
    text_lower = text.lower()
    realms = {
        "technology": [
            "computer",
            "software",
            "code",
            "algorithm",
            "data",
            "ai",
            "machine",
            "digital",
        ],
        "science": ["atom", "molecule", "cell", "energy", "physics", "chemistry", "biology"],
        "health": ["health", "medical", "disease", "exercise", "diet", "body", "sleep"],
        "history": ["war", "century", "king", "empire", "ancient", "historical"],
        "nature": ["animal", "plant", "environment", "climate", "ocean", "forest"],
        "business": ["company", "market", "economy", "money", "trade", "business"],
        "arts": ["music", "art", "paint", "novel", "film", "creative"],
    }
    for realm, keywords in realms.items():
        if any(kw in text_lower for kw in keywords):
            return realm
    return "general"


def compute_complexity(doc) -> float:
    """Syntactic complexity 0-1."""
    if len(doc) == 0:
        return 0.0
    length_score = min(len(doc) / 30, 1.0)
    sub_clauses = sum(1 for t in doc if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp", "acl"))
    clause_score = min(sub_clauses / 3, 1.0)
    return round((length_score + clause_score) / 2, 2)


def process_sentence(doc, sent_id: str) -> dict[str, Any] | None:
    """Process single sentence to lean SVC format."""
    if len(doc) < 3 or len(doc) > 100:
        return None

    has_verb = any(t.pos_ in ("VERB", "AUX") and t.dep_ == "ROOT" for t in doc)
    if not has_verb:
        return None

    svc = extract_svc(doc)
    if not svc.valid:
        return None

    pos = [t.pos_ for t in doc if t.pos_ != "PUNCT"]
    lemmas = [t.lemma_.lower() for t in doc if t.pos_ != "PUNCT"]
    deps = [t.dep_ for t in doc if t.pos_ != "PUNCT"]

    root_verb = next((t.lemma_.lower() for t in doc if t.dep_ == "ROOT"), "")

    return {
        "id": sent_id,
        "text": doc.text.strip(),
        "svc": {"s": svc.subject, "v": svc.verb, "c": svc.complement},
        "pos": pos,
        "lemmas": lemmas,
        "deps": deps,
        "root_verb": root_verb,
        "realm": classify_realm(doc.text),
        "complexity": compute_complexity(doc),
    }


def process_instruct_file(input_path: Path, output_path: Path, max_entries: int = None):
    """Process instruct JSONL to lean SVC format."""
    stats = Counter()
    entry_count = 0

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_num, line in enumerate(fin):
            if max_entries and entry_count >= max_entries:
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            texts = []
            if "prompt" in data:
                texts.append(("p", data["prompt"]))
            if "response" in data:
                texts.append(("r", data["response"]))

            for text_type, text in texts:
                doc = nlp(text)
                for sent_num, sent in enumerate(doc.sents):
                    if len(sent) < 3:
                        continue
                    sent_id = f"i{line_num}_{text_type}{sent_num}"

                    try:
                        result = process_sentence(sent.as_doc(), sent_id)
                        if result:
                            fout.write(json.dumps(result) + "\n")
                            stats["success"] += 1
                            entry_count += 1
                        else:
                            stats["filtered"] += 1
                    except Exception:
                        stats["errors"] += 1

            if (line_num + 1) % 1000 == 0:
                print(f"Processed {line_num + 1} entries, {stats['success']} sentences...")

    return stats


def main():
    """Run main."""

    base = Path(r"E:\Grillcheese Inc\grilly\grilly\experimental_datasets")
    input_path = base / "instruct_55k_clean.jsonl"
    output_path = base / "instruct_svc_lean.jsonl"

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    stats = process_instruct_file(input_path, output_path)

    print("\n=== Done ===")
    print(f"Success: {stats['success']}")
    print(f"Filtered: {stats['filtered']}")
    print(f"Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
