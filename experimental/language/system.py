"""
InstantLanguage - Complete instant language learning system.

Combines word encoding, sentence encoding, parsing, and generation.
"""

import numpy as np
import re
from typing import Dict, List, Tuple
from collections import defaultdict
from grilly.experimental.language.encoder import WordEncoder, SentenceEncoder
from grilly.experimental.language.generator import SentenceGenerator
from grilly.experimental.language.parser import ResonatorParser
from grilly.experimental.vsa.ops import HolographicOps


class InstantLanguage:
    """
    Complete instant language learning system.
    
    Combines:
    - Word encoding (n-grams, no training)
    - Relation extraction (O(1))
    - Sentence encoding (composition)
    - Parsing (resonator factorization)
    - Generation (template filling)
    
    Everything is instant - no gradient descent!
    """
    
    DEFAULT_DIM = 4096
    
    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim
        
        # Components
        self.word_encoder = WordEncoder(dim=dim)
        self.sentence_encoder = SentenceEncoder(self.word_encoder)
        self.generator = SentenceGenerator(self.sentence_encoder)
        self.parser = ResonatorParser(self.sentence_encoder)
        
        # Memory
        self.sentence_memory: List[Tuple[np.ndarray, List[str]]] = []
        self.relation_memory: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    
    def learn_sentence(self, sentence: str) -> np.ndarray:
        """
        Learn a sentence instantly.
        
        Encodes and stores in memory. No training loop!
        """
        words = self._tokenize(sentence)
        
        # Encode all words (builds vocabulary on the fly)
        for word in words:
            self.word_encoder.encode_word(word)
        
        # Encode sentence
        sent_vec = self.sentence_encoder.encode_sentence(words)
        
        # Store
        self.sentence_memory.append((sent_vec, words))
        
        return sent_vec
    
    def learn_relation(self, word_a: str, word_b: str, relation: str):
        """
        Learn a word relation from a single example.
        """
        self.relation_memory[relation].append((word_a, word_b))
        
        # If we have enough examples, create a relation prototype
        if len(self.relation_memory[relation]) >= 2:
            pairs = self.relation_memory[relation]
            self.word_encoder.learn_relation(pairs, relation)
    
    def query_relation(self, word: str, relation: str) -> List[Tuple[str, float]]:
        """
        Query: "What is the [relation] of [word]?"
        
        E.g., "What is the antonym of hot?" -> cold
        """
        if relation not in self.word_encoder.relations:
            return []
        
        rel_vec = self.word_encoder.relations[relation]
        result_vec = self.word_encoder.apply_relation(word, rel_vec)
        return self.word_encoder.find_closest(result_vec)
    
    def express_relation(self, word_a: str, relation: str, word_b: str) -> str:
        """
        Generate a sentence expressing a relation.
        """
        words = self.generator.generate_from_relation(word_a, relation, word_b)
        return " ".join(words)
    
    def parse_sentence(self, sentence: str) -> List[Tuple[str, str, float]]:
        """
        Parse a sentence into word-role pairs.
        """
        words = self._tokenize(sentence)
        
        # First encode
        for word in words:
            self.word_encoder.encode_word(word)
        
        sent_vec = self.sentence_encoder.encode_sentence(words)
        
        return self.parser.parse(sent_vec, num_slots=len(words))
    
    def find_similar_sentences(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[List[str], float]]:
        """
        Find similar sentences in memory.
        """
        query_words = self._tokenize(query)
        query_vec = self.sentence_encoder.encode_sentence(query_words)
        
        results = []
        for sent_vec, words in self.sentence_memory:
            sim = HolographicOps.similarity(query_vec, sent_vec)
            results.append((words, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def complete(self, partial: str, role: str = "OBJ") -> List[Tuple[str, float]]:
        """
        Complete a partial sentence.
        
        "The dog chased the ___" -> find best OBJ
        """
        words = self._tokenize(partial)
        
        # Encode what we have
        for word in words:
            self.word_encoder.encode_word(word)
        
        sent_vec = self.sentence_encoder.encode_sentence(words)
        
        return self.sentence_encoder.find_role_filler(sent_vec, role)
    
    def analogy(
        self,
        word_a: str,
        word_b: str,
        word_c: str
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy: A is to B as C is to ?
        
        E.g., king:queen :: man:? -> woman
        """
        # Extract A:B relation
        relation = self.word_encoder.extract_relation(word_a, word_b)
        
        # Apply to C
        c_vec = self.word_encoder.encode_word(word_c)
        d_vec = HolographicOps.convolve(c_vec, relation)
        
        return self.word_encoder.find_closest(d_vec)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation, split on whitespace
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
