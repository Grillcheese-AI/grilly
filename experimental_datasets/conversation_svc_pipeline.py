"""
Conversation SVC Pipeline
Cleans human-assistant conversations and extracts semantic SVC

Input: conversations_dataset_anonymized_cleaned.jsonl
Output: conversations_svc_semantic.jsonl
"""

import json
import re
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


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================


def remove_code_blocks(text: str) -> str:
    """Remove markdown code blocks"""
    # Remove ``` ... ``` blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code `...`
    text = re.sub(r"`[^`]+`", " ", text)
    return text


def remove_paths_and_urls(text: str) -> str:
    """Remove file paths, URLs, and technical artifacts"""
    # URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Windows paths
    text = re.sub(r"[A-Z]:\\[\w\\/\-\.]+", " ", text)
    # Unix paths
    text = re.sub(r"/[\w/\-\.]+\.\w+", " ", text)
    # Remaining path-like patterns
    text = re.sub(r"\b\w+\.\w+\.\w+\b", " ", text)  # file.ext.ext
    return text


def remove_technical_artifacts(text: str) -> str:
    """Remove code-like artifacts and technical noise"""
    # Remove lines that look like code output
    text = re.sub(r"^[\s]*[\-\=\*]{3,}[\s]*$", "", text, flags=re.MULTILINE)
    # Remove shell prompts
    text = re.sub(r"\$\s*$", "", text, flags=re.MULTILINE)
    # Remove hex/binary patterns
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", " ", text)
    # Remove variable assignments
    text = re.sub(r"\b\w+\s*=\s*[\d\.]+\b", " ", text)
    # Remove emoji unicode escapes
    text = re.sub(r"\\x[0-9a-fA-F]{2}", "", text)
    # Remove markdown formatting remnants
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold** -> bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic* -> italic
    text = re.sub(r"#{1,6}\s*", "", text)  # headers
    return text


def remove_placeholder_artifacts(text: str) -> str:
    """Clean up placeholder formatting artifacts"""
    # Already has placeholders like [PROJECT_NAME], keep them but clean formatting
    # Remove numeric suffixes from placeholders for consistency
    text = re.sub(r"\[([A-Z_]+)_\d+\]", r"[\1]", text)
    return text


def clean_whitespace(text: str) -> str:
    """Normalize whitespace"""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline"""
    text = remove_code_blocks(text)
    text = remove_paths_and_urls(text)
    text = remove_technical_artifacts(text)
    text = remove_placeholder_artifacts(text)
    text = clean_whitespace(text)
    return text


def is_valid_sentence(sent: str) -> bool:
    """Filter out non-sentence content"""
    sent = sent.strip()
    if len(sent) < 10:
        return False
    if len(sent.split()) < 3:
        return False
    # Skip if mostly placeholders
    placeholder_count = len(re.findall(r"\[[A-Z_]+\]", sent))
    word_count = len(sent.split())
    if placeholder_count > word_count / 2:
        return False
    # Skip if starts with bullet/number only
    if re.match(r"^[\d\.\-\*\•]+\s*$", sent):
        return False
    # Skip if looks like a file listing
    if re.match(r"^[\w\-]+\.\w+\s*$", sent):
        return False
    return True


# =============================================================================
# SVC EXTRACTION (Semantic)
# =============================================================================


def extract_svc(doc) -> SVCResult:
    """Extract SEMANTIC Subject-Verb-Complement"""
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

    # Collect verb phrase
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

    # Collect direct object
    for token in doc:
        if token.dep_ in ("dobj", "attr", "oprd"):
            subtree = list(token.subtree)
            for t in subtree:
                if t.dep_ != "punct":
                    dobj_parts.append((t.i, t.text))

    # SEMANTIC: Promote direct object in imperatives
    is_imperative = not has_grammatical_subject and dobj_parts
    if is_imperative:
        subject_parts = dobj_parts
        dobj_parts = []

    # Complement = rest after verb
    verb_end = max([i for i, _ in verb_parts]) if verb_parts else root.i
    subject_indices = {i for i, _ in subject_parts}

    for token in doc:
        if token.i > verb_end and token.dep_ != "punct" and token.i not in subject_indices:
            complement_parts.append((token.i, token.text))

    subject = " ".join(t for _, t in sorted(subject_parts))
    verb = " ".join(t for _, t in sorted(verb_parts))
    complement = " ".join(t for _, t in sorted(complement_parts))

    valid = bool(verb and len(doc) >= 3)
    return SVCResult(subject, verb, complement, valid)


def classify_realm(text: str) -> str:
    """Classify into domain"""
    text_lower = text.lower()
    realms = {
        "technology": [
            "code",
            "programming",
            "software",
            "algorithm",
            "data",
            "ai",
            "machine",
            "neural",
            "model",
            "training",
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
        "health": ["health", "medical", "disease", "exercise", "diet", "body", "sleep"],
        "business": ["company", "market", "economy", "money", "trade", "business", "project"],
        "conversation": ["help", "please", "thanks", "question", "understand", "explain", "think"],
    }
    for realm, keywords in realms.items():
        if any(kw in text_lower for kw in keywords):
            return realm
    return "general"


def compute_complexity(doc) -> float:
    """Syntactic complexity 0-1"""
    if len(doc) == 0:
        return 0.0
    length_score = min(len(doc) / 30, 1.0)
    sub_clauses = sum(1 for t in doc if t.dep_ in ("advcl", "relcl", "ccomp", "xcomp", "acl"))
    clause_score = min(sub_clauses / 3, 1.0)
    depths = [len(list(t.ancestors)) for t in doc]
    avg_depth = sum(depths) / len(depths) if depths else 0
    depth_score = min(avg_depth / 5, 1.0)
    return round((length_score + clause_score + depth_score) / 3, 2)


def process_sentence(doc, sent_id: str, role: str) -> dict[str, Any] | None:
    """Process single sentence into SVC format"""
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
        "realm": classify_realm(doc.text),
        "role": role,  # "user" or "assistant"
        "source": "conversation",
        "complexity": compute_complexity(doc),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def process_conversations(input_path: Path, output_path: Path):
    """Process conversation dataset"""
    stats = Counter()

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_num, line in enumerate(fin):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            messages = data.get("messages", [])
            conv_id = data.get("conversation_id", f"conv_{line_num}")

            for msg_idx, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

                # Clean the content
                cleaned = clean_text(content)
                if not cleaned:
                    continue

                # Split into sentences
                try:
                    doc = nlp(cleaned)
                    for sent_num, sent in enumerate(doc.sents):
                        sent_text = sent.text.strip()

                        if not is_valid_sentence(sent_text):
                            stats["filtered_invalid"] += 1
                            continue

                        sent_id = f"{conv_id}_m{msg_idx}_s{sent_num}"

                        try:
                            sent_doc = nlp(sent_text)
                            result = process_sentence(sent_doc, sent_id, role)

                            if result:
                                fout.write(json.dumps(result) + "\n")
                                stats["success"] += 1
                                stats[f"role_{role}"] += 1
                            else:
                                stats["filtered_no_svc"] += 1
                        except Exception:
                            stats["process_errors"] += 1

                except Exception:
                    stats["nlp_errors"] += 1

            if (line_num + 1) % 100 == 0:
                print(f"Processed {line_num + 1} conversations, {stats['success']} sentences...")

    return stats


def main():
    """Run main."""

    input_path = Path(
        r"E:\Grillcheese Inc\grilly\grilly\experimental_datasets\conversations_dataset_anonymized_cleaned.jsonl"
    )
    output_path = Path(
        r"C:\Users\grill\Desktop\GrillCheese\data_learning\conversations_svc_semantic.jsonl"
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("Processing conversations with semantic SVC extraction...")
    print()

    stats = process_conversations(input_path, output_path)

    print("\n=== Processing Complete ===")
    print(f"Successful sentences: {stats['success']}")
    print(f"  - From user: {stats.get('role_user', 0)}")
    print(f"  - From assistant: {stats.get('role_assistant', 0)}")
    print(f"Filtered (invalid): {stats.get('filtered_invalid', 0)}")
    print(f"Filtered (no SVC): {stats.get('filtered_no_svc', 0)}")
    print(
        f"Errors: {stats.get('json_errors', 0) + stats.get('process_errors', 0) + stats.get('nlp_errors', 0)}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
