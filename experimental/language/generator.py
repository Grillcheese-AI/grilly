"""
SentenceGenerator for instant sentence generation.

Generates sentences from templates and relations.
"""

import numpy as np

from grilly.experimental.language.encoder import SentenceEncoder
from grilly.experimental.vsa.ops import HolographicOps


class SentenceGenerator:
    """
    Generate sentences by:
    1. Template filling (unbind template slots, bind with content)
    2. Relation chaining (A causes B, express as sentence)
    3. Role-based construction (given subject, verb, object -> sentence)

    No training needed!
    """

    def __init__(self, sentence_encoder: SentenceEncoder):
        """Initialize the instance."""

        self.encoder = sentence_encoder
        self.word_encoder = sentence_encoder.word_encoder
        self.dim = sentence_encoder.dim

        # Common templates
        self.templates: dict[str, dict] = {}
        self._init_templates()

        # Learned sentence patterns
        self.patterns: dict[str, np.ndarray] = {}

    def _init_templates(self):
        """Initialize common sentence templates."""
        self.templates = {
            "simple_transitive": {
                "pattern": ["SUBJ", "VERB", "OBJ"],
                "example": ["The dog", "chased", "the cat"],
            },
            "copula": {
                "pattern": ["SUBJ", "AUX", "ADJ"],
                "example": ["The sky", "is", "blue"],
            },
            "ditransitive": {
                "pattern": ["SUBJ", "VERB", "IOBJ", "OBJ"],
                "example": ["She", "gave", "him", "a book"],
            },
            "prepositional": {
                "pattern": ["SUBJ", "VERB", "PREP", "POBJ"],
                "example": ["The cat", "sat", "on", "the mat"],
            },
            "passive": {
                "pattern": ["OBJ", "AUX", "VERB", "PREP", "SUBJ"],
                "example": ["The ball", "was", "kicked", "by", "John"],
            },
            "question_what": {
                "pattern": ["OBJ", "AUX", "SUBJ", "VERB"],
                "example": ["What", "did", "you", "see"],
            },
            "question_who": {
                "pattern": ["SUBJ", "VERB", "OBJ"],
                "example": ["Who", "ate", "the pizza"],
            },
            "causal": {
                "pattern": ["SUBJ", "VERB", "OBJ", "COMP", "SUBJ", "VERB"],
                "example": ["Lightning", "causes", "thunder", "because", "it", "heats"],
            },
        }

    def generate_from_roles(
        self, role_fillers: dict[str, str], template_name: str = "simple_transitive"
    ) -> list[str]:
        """
        Generate sentence by filling template roles.

        Args:
            role_fillers: {"SUBJ": "The dog", "VERB": "chased", "OBJ": "the cat"}
            template_name: Which template to use

        Returns:
            List of words forming the sentence
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.templates[template_name]
        pattern = template["pattern"]

        words = []
        for role in pattern:
            if role in role_fillers:
                # Add the filler (may be multi-word)
                filler = role_fillers[role]
                if isinstance(filler, str):
                    words.extend(filler.split())
                else:
                    words.append(str(filler))
            else:
                # Try to find a default or placeholder
                words.append(f"[{role}]")

        return words

    def generate_from_relation(self, subject: str, relation: str, object_: str) -> list[str]:
        """
        Generate sentence expressing a relation.

        Given: subject="lightning", relation="causes", object="thunder"
        Output: ["Lightning", "causes", "thunder"]
        """
        # Map relations to sentence patterns
        relation_templates = {
            "causes": ("simple_transitive", {"VERB": "causes"}),
            "before": ("simple_transitive", {"VERB": "precedes"}),
            "after": ("simple_transitive", {"VERB": "follows"}),
            "synonym": ("copula", {"AUX": "is similar to"}),
            "antonym": ("copula", {"AUX": "is opposite of"}),
            "hypernym": ("copula", {"AUX": "is a type of"}),
            "hyponym": ("copula", {"AUX": "is an example of"}),
            "part_of": ("copula", {"AUX": "is part of"}),
            "contains": ("simple_transitive", {"VERB": "contains"}),
        }

        if relation in relation_templates:
            template_name, extras = relation_templates[relation]
            fillers = {"SUBJ": subject.capitalize(), "OBJ": object_, **extras}
            return self.generate_from_roles(fillers, template_name)
        else:
            # Default: simple statement
            return [subject.capitalize(), relation, object_]

    def learn_pattern(self, sentences: list[list[str]], pattern_name: str):
        """
        Learn a sentence pattern from examples.

        Extract the common structure from multiple sentences.
        """
        pattern_vecs = []

        for words in sentences:
            sent_vec = self.encoder.encode_sentence(words)
            pattern_vecs.append(sent_vec)

        # Average to get prototype pattern
        self.patterns[pattern_name] = HolographicOps.bundle(pattern_vecs)

    def learn_svc_templates(
        self,
        word_lists: list[list[str]],
        template_name: str,
    ) -> None:
        """
        Learn a sentence template from SVC data.

        Groups sentences that share the same dependency structure and
        creates a prototype vector in VSA space, plus a role pattern
        derived from the majority role assignment.

        Args:
            word_lists: List of tokenized sentences sharing the same
                        dependency pattern.
            template_name: Name for the template (e.g. the dep-label key).
        """
        if not word_lists:
            return

        # Build prototype via pattern learning
        self.learn_pattern(word_lists, template_name)

        # Also infer a role pattern from the first sentence
        # (all sentences in a group share the same dep structure)
        first_words = word_lists[0]
        auto_roles = self.encoder._auto_assign_roles(first_words)

        # Derive a compact role pattern (unique ordered roles)
        seen: dict[str, bool] = {}
        role_pattern: list[str] = []
        for role in auto_roles:
            if role not in seen:
                seen[role] = True
                role_pattern.append(role)

        self.templates[template_name] = {
            "pattern": role_pattern,
            "example": first_words,
        }

    def complete_sentence(
        self, partial: list[str], partial_roles: list[str], missing_role: str
    ) -> list[tuple[str, float]]:
        """
        Complete a partial sentence by finding what fits a missing role.

        "The dog ___ the cat" -> find best VERB
        """
        # Encode partial sentence
        partial_vec = self.encoder.encode_sentence(partial, partial_roles)

        # Query what should fill the missing role
        query = self.encoder.query_role(partial_vec, missing_role)

        # Find closest words
        return self.word_encoder.find_closest(query)
