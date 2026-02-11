"""
Lean SVC Pipeline for Grilly Training Data
Converts instruct_55k_clean.jsonl to sentence-level SEMANTIC SVC format

SEMANTIC SVC: Subject = what the sentence is ABOUT (not grammatical subject)
- Imperatives: "Discuss the challenges" → S="the challenges", V="Discuss"
- Questions: "What causes rain?" → S="rain", V="causes"
- Passives: "The ball was kicked" → S="The ball" (already correct)
- Normal: "Dogs eat food" → S="Dogs", V="eat", C="food"
"""

import json
import re
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


@dataclass
class SVCResult:
    subject: str
    verb: str
    complement: str
    valid: bool


def extract_svc(doc) -> SVCResult:
    """Extract SEMANTIC Subject-Verb-Complement from a spaCy Doc."""
    subject_parts = []
    verb_parts = []
    complement_parts = []
    dobj_parts = []

    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if not root or root.pos_ not in ("VERB", "AUX"):
        return SVCResult("", "", "", False)

    # Collect verb phrase (aux + main verb)
    for token in doc:
        if token == root or token.head == root:
            if token.dep_ in ("aux", "auxpass", "neg") or token == root:
                verb_parts.append((token.i, token.text))

    # Collect grammatical subject
    has_grammatical_subject = False
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass", "expl"):
            has_grammatical_subject = True
            subtree = list(token.subtree)
            for t in subtree:
                if t.i < root.i:
                    subject_parts.append((t.i, t.text))

    # Collect direct object (for semantic promotion in imperatives)
    for token in doc:
        if token.dep_ in ("dobj", "attr", "oprd"):
            subtree = list(token.subtree)
            for t in subtree:
                if t.dep_ != "punct":
                    dobj_parts.append((t.i, t.text))

    # SEMANTIC RULE: No grammatical subject + has direct object = imperative
    # Promote direct object to subject position
    is_imperative = not has_grammatical_subject and dobj_parts

    if is_imperative:
        subject_parts = dobj_parts
        dobj_parts = []

    # Complement = everything after verb, excluding promoted subject parts
    verb_end = max([i for i, _ in verb_parts]) if verb_parts else root.i
    subject_indices = {i for i, _ in subject_parts}

    for token in doc:
        if token.i > verb_end and token.dep_ != "punct" and token.i not in subject_indices:
            complement_parts.append((token.i, token.text))

    # Sort and join
    subject = " ".join(t for _, t in sorted(subject_parts))
    verb = " ".join(t for _, t in sorted(verb_parts))
    complement = " ".join(t for _, t in sorted(complement_parts))

    valid = bool(verb and len(doc) >= 3)
    return SVCResult(subject, verb, complement, valid)


def classify_realm(text: str, pos_tags: list) -> str:
    """Simple realm classification based on keywords."""
    text_lower = text.lower()

    realms = {
        "technology": [
            "computer",
            "software",
            "code",
            "programming",
            "algorithm",
            "data",
            "ai",
            "machine",
            "digital",
        ],
        "science": [
            "atom",
            "molecule",
            "cell",
            "energy",
            "physics",
            "chemistry",
            "biology",
            "experiment",
        ],
        "health": ["health", "medical", "disease", "exercise", "diet", "body", "sleep", "doctor"],
        "history": ["war", "century", "king", "empire", "ancient", "historical", "revolution"],
        "nature": ["animal", "plant", "environment", "climate", "ocean", "forest", "species"],
        "business": ["company", "market", "economy", "money", "trade", "invest", "business"],
        "arts": ["music", "art", "paint", "write", "novel", "film", "creative"],
        "social": ["people", "society", "culture", "family", "community", "relationship"],
    }

    for realm, keywords in realms.items():
        if any(kw in text_lower for kw in keywords):
            return realm
    return "general"


def compute_complexity(doc) -> float:
    """Compute syntactic complexity score 0-1."""
    if len(doc) == 0:
        return 0.0

    length_score = min(len(doc) / 30, 1.0)
    sub_clauses = sum(1 for t in doc if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp", "acl"))
    clause_score = min(sub_clauses / 3, 1.0)
    depths = [len(list(t.ancestors)) for t in doc]
    avg_depth = sum(depths) / len(depths) if depths else 0
    depth_score = min(avg_depth / 5, 1.0)

    return round((length_score + clause_score + depth_score) / 3, 2)


def process_sentence(doc, sent_id: str, source: str) -> dict[str, Any] | None:
    """Process a single sentence into lean SVC format."""
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

    root_verb = ""
    for t in doc:
        if t.dep_ == "ROOT":
            root_verb = t.lemma_.lower()
            break

    return {
        "id": sent_id,
        "text": doc.text.strip(),
        "svc": {"s": svc.subject, "v": svc.verb, "c": svc.complement},
        "pos": pos,
        "lemmas": lemmas,
        "deps": deps,
        "root_verb": root_verb,
        "realm": classify_realm(doc.text, pos),
        "source": source,
        "complexity": compute_complexity(doc),
    }


def split_into_sentences(text: str) -> Generator[str, None, None]:
    """Split text into clean sentences."""
    doc = nlp(text)
    for sent in doc.sents:
        cleaned = sent.text.strip()
        if re.match(r"^\d+\.\s*$", cleaned):
            continue
        if len(cleaned.split()) < 3:
            continue
        yield cleaned


def process_instruct_file(input_path: Path, output_path: Path, max_entries: int = None):
    """Process instruct_55k_clean.jsonl to lean SVC format."""

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
                texts.append(("prompt", data["prompt"]))
            if "response" in data:
                texts.append(("response", data["response"]))

            for text_type, text in texts:
                for sent_num, sentence in enumerate(split_into_sentences(text)):
                    sent_id = f"i{line_num}_{text_type[0]}{sent_num}"

                    try:
                        sent_doc = nlp(sentence)
                        result = process_sentence(sent_doc, sent_id, "instruct")

                        if result:
                            fout.write(json.dumps(result) + "\n")
                            stats["success"] += 1
                            entry_count += 1
                        else:
                            stats["filtered"] += 1
                    except Exception:
                        stats["process_errors"] += 1

            if (line_num + 1) % 1000 == 0:
                print(f"Processed {line_num + 1} entries, {stats['success']} sentences...")

    return stats


def main():
    input_path = Path(
        r"C:\Users\grill\Desktop\GrillCheese\data_learning\jsonl\instruct_55k_clean.jsonl"
    )
    output_path = Path(
        r"C:\Users\grill\Desktop\GrillCheese\data_learning\instruct_svc_semantic.jsonl"
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Using SEMANTIC SVC extraction...")
    print()

    stats = process_instruct_file(input_path, output_path)

    print("\n=== Processing Complete ===")
    print(f"Successful: {stats['success']}")
    print(f"Filtered: {stats['filtered']}")
    print(f"Errors: {stats['json_errors'] + stats['process_errors']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
