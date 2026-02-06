#!/usr/bin/env python3
"""
SVC Enhanced Dataset Converter for Grilly

Converts raw text data to structured SVC (Subject-Verb-Complement) format
with full linguistic annotations for training Grilly's language models.

Uses Stanza for tokenization, POS, lemma, dependency parsing and NER.

Usage:
    python convert_to_svc.py --input /path/to/data --output /path/to/output --gpu 0
"""

import json
import argparse
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Generator, Tuple, Any
from collections import Counter
import logging
from datetime import datetime
import sys

# Progress tracking
from tqdm import tqdm

# NLP
import stanza

# Adapters so existing code can use Stanza output like spaCy Doc/Span/Token
Doc = Any  # type alias for doc-like object
Span = Any  # type alias for sentence-like object

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('svc_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Stanza adapters (spaCy-like Doc / Span / Token interface)
# =============================================================================

class _MorphView:
    """Mimic spaCy token.morph for Stanza word.feats."""
    def __init__(self, feats: Optional[str]):
        self._feats = feats or ""

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for part in self._feats.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k] = [v] if k not in out else out[k] + [v]
        return out

    def __str__(self) -> str:
        return self._feats


class _StanzaToken:
    """Wrap a Stanza Word to look like a spaCy Token."""
    __slots__ = ("_word", "_sent", "_global_i", "_head_ref", "_sentence_tokens")

    def __init__(self, word, sent: "_StanzaSent", global_i: int, sentence_tokens: List["_StanzaToken"]):
        self._word = word
        self._sent = sent
        self._global_i = global_i
        self._sentence_tokens = sentence_tokens
        self._head_ref: Optional[_StanzaToken] = None

    def _set_head(self, head_token: Optional["_StanzaToken"]):
        self._head_ref = head_token

    @property
    def text(self) -> str:
        return self._word.text

    @property
    def lemma_(self) -> str:
        return self._word.lemma

    @property
    def pos_(self) -> str:
        return self._word.upos

    @property
    def tag_(self) -> str:
        return getattr(self._word, "xpos", "") or self._word.upos

    @property
    def dep_(self) -> str:
        return self._word.deprel

    @property
    def head(self) -> "_StanzaToken":
        if self._head_ref is not None:
            return self._head_ref
        return self

    @property
    def i(self) -> int:
        return self._global_i

    @property
    def is_punct(self) -> bool:
        return self._word.upos == "PUNCT"

    @property
    def morph(self) -> _MorphView:
        return _MorphView(getattr(self._word, "feats", None))

    @property
    def children(self) -> List["_StanzaToken"]:
        return [t for t in self._sentence_tokens if t._word.head == self._word.id]

    @property
    def subtree(self) -> List["_StanzaToken"]:
        out = [self]
        for t in self._sentence_tokens:
            if t._word.head == self._word.id:
                out.extend(t.subtree)
        return out


class _StanzaSent:
    """Wrap a Stanza Sentence to look like a spaCy Span."""
    def __init__(self, sent, doc: "_StanzaDoc", start_i: int):
        self._sent = sent
        self._doc = doc
        self._start_i = start_i
        self._tokens: List[_StanzaToken] = []
        self._build_tokens()

    def _build_tokens(self):
        for i, w in enumerate(self._sent.words):
            global_i = self._start_i + i
            t = _StanzaToken(w, self, global_i, self._tokens)
            self._tokens.append(t)
        for t in self._tokens:
            head_id = t._word.head
            if head_id == 0 or head_id < 1 or head_id > len(self._tokens):
                t._set_head(t)
            else:
                t._set_head(self._tokens[head_id - 1])

    def __iter__(self):
        return iter(self._tokens)

    def __getitem__(self, i: int) -> _StanzaToken:
        return self._tokens[i]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self._sent.words)

    @property
    def ents(self) -> List[Any]:
        return self._doc._entities_for_sent(self)

    def __len__(self) -> int:
        return len(self._tokens)


class _StanzaDoc:
    """Wrap a Stanza Document to look like a spaCy Doc."""
    def __init__(self, doc, text: str):
        self._doc = doc
        self._text = text
        self._sents: List[_StanzaSent] = []
        idx = 0
        for sent in doc.sentences:
            s = _StanzaSent(sent, self, idx)
            self._sents.append(s)
            idx += len(s._tokens)
        self._entities = getattr(doc, "entities", []) or []

    @property
    def text(self) -> str:
        return self._text

    @property
    def sents(self):
        return self._sents

    def _entities_for_sent(self, sent: _StanzaSent) -> List[Any]:
        """Return entities overlapping this sentence (character span)."""
        sent_start = sum(len(s._sent.words) for s in self._sents if s is not sent)
        # Approximate: use token index; Stanza Entity has start_char, end_char
        result = []
        for ent in self._entities:
            e_start = getattr(ent, "start_char", 0)
            e_end = getattr(ent, "end_char", 0)
            sent_text = sent.text
            # Simple overlap: if entity text appears in sentence
            if hasattr(ent, "text") and ent.text and ent.text in sent_text:
                result.append(_EntitySpan(ent.text, getattr(ent, "type", "ENTITY"), sent, self))
        return result

    def __iter__(self):
        for s in self._sents:
            for t in s._tokens:
                yield t

    @property
    def noun_chunks(self) -> List[Any]:
        """Stanza does not provide noun chunks; return empty list."""
        return []


class _EntitySpan:
    """Minimal entity span for sent.ents compatibility."""
    def __init__(self, text: str, label: str, sent: _StanzaSent, doc: _StanzaDoc):
        self.text = text
        self.label_ = label
        self._sent = sent
        self._doc = doc

    @property
    def start(self) -> int:
        return self._sent._start_i

    @property
    def end(self) -> int:
        return self._sent._start_i + len(self._sent._tokens)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SVCComponents:
    subject: str
    verb: str
    complement: str


@dataclass  
class TokenInfo:
    text: str
    sentence_id: int
    token_id: int


@dataclass
class POSTag:
    token: str
    pos: str
    xpos: str
    feats: Optional[str]
    sentence_id: int
    token_id: int


@dataclass
class Lemma:
    token: str
    lemma: str
    sentence_id: int
    token_id: int


@dataclass
class Dependency:
    token: str
    head: str
    deprel: str
    sentence_id: int
    token_id: int


@dataclass
class NamedEntity:
    text: str
    label: str
    start: int
    end: int
    sentence_id: int


# =============================================================================
# SVC Extractor
# =============================================================================

class SVCExtractor:
    """Extract Subject-Verb-Complement structure from parsed sentences."""
    
    # Dependency relations that indicate subjects
    SUBJECT_DEPS = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'expl'}
    
    # Dependency relations that indicate objects/complements
    OBJECT_DEPS = {'dobj', 'obj', 'iobj', 'pobj', 'attr', 'oprd', 'dative'}
    COMPLEMENT_DEPS = {'xcomp', 'ccomp', 'acomp', 'prep', 'obl', 'advmod', 'advcl'}
    
    def extract(self, sent: Span) -> SVCComponents:
        """Extract SVC from a sentence (Stanza span-like object)."""
        subject_tokens = []
        verb_tokens = []
        complement_tokens = []
        
        root = None
        for token in sent:
            if token.dep_ == 'ROOT':
                root = token
                break
        
        if root is None:
            # Fallback: first verb or first token
            for token in sent:
                if token.pos_ == 'VERB':
                    root = token
                    break
            if root is None and len(sent) > 0:
                root = sent[0]
        
        if root is None:
            return SVCComponents(
                subject=sent.text,
                verb="",
                complement=""
            )
        
        # Collect verb phrase
        verb_tokens = self._get_verb_phrase(root)
        
        # Collect subject
        for token in sent:
            if token.dep_ in self.SUBJECT_DEPS and token.head == root:
                subject_tokens = self._get_noun_phrase(token)
                break
        
        # Collect complement (everything else after verb)
        complement_start = None
        for token in sent:
            if token.dep_ in self.OBJECT_DEPS | self.COMPLEMENT_DEPS:
                if token.head == root or token.head.head == root:
                    if complement_start is None or token.i < complement_start:
                        complement_start = token.i
        
        if complement_start is not None:
            complement_tokens = [t for t in sent if t.i >= complement_start and t not in verb_tokens]
        
        # Build strings
        subject = ' '.join(t.text for t in sorted(subject_tokens, key=lambda x: x.i)) if subject_tokens else ""
        verb = ' '.join(t.text for t in sorted(verb_tokens, key=lambda x: x.i)) if verb_tokens else root.text if root else ""
        complement = ' '.join(t.text for t in sorted(complement_tokens, key=lambda x: x.i)) if complement_tokens else ""
        
        # Fallback: if no subject found, use text before verb
        if not subject and root:
            pre_verb = [t for t in sent if t.i < root.i]
            subject = ' '.join(t.text for t in pre_verb) if pre_verb else ""
        
        # Fallback: if no complement, use text after verb
        if not complement and root:
            post_verb = [t for t in sent if t.i > root.i and t not in verb_tokens]
            complement = ' '.join(t.text for t in post_verb) if post_verb else ""
        
        return SVCComponents(
            subject=subject.strip(),
            verb=verb.strip(),
            complement=complement.strip()
        )
    
    def _get_verb_phrase(self, verb_token) -> List:
        """Get the full verb phrase including auxiliaries."""
        tokens = [verb_token]
        for child in verb_token.children:
            if child.dep_ in {'aux', 'auxpass', 'neg', 'prt'}:
                tokens.append(child)
        return tokens
    
    def _get_noun_phrase(self, head_token) -> List:
        """Get the full noun phrase including modifiers."""
        tokens = [head_token]
        for child in head_token.subtree:
            if child.dep_ in {'det', 'amod', 'compound', 'poss', 'nummod', 'nmod'}:
                tokens.append(child)
            elif child.dep_ == 'prep':
                # Include prepositional phrases in the noun phrase
                for subchild in child.subtree:
                    tokens.append(subchild)
        return list(set(tokens))


# =============================================================================
# Linguistic Feature Extractor
# =============================================================================

class LinguisticFeatureExtractor:
    """Extract comprehensive linguistic features from parsed text."""
    
    def __init__(self):
        self.svc_extractor = SVCExtractor()
    
    def extract_all(self, doc: Doc, text_id: str) -> Dict[str, Any]:
        """Extract all linguistic features from a doc-like object."""
        
        tokens = []
        pos_tags = []
        lemmas = []
        dependencies = []
        named_entities = []
        
        for sent_id, sent in enumerate(doc.sents):
            for token in sent:
                token_id = token.i + 1  # 1-indexed
                
                tokens.append({
                    'text': token.text,
                    'sentence_id': sent_id,
                    'token_id': token_id
                })
                
                pos_tags.append({
                    'token': token.text,
                    'pos': token.pos_,
                    'xpos': token.tag_,
                    'feats': str(token.morph) if token.morph else None,
                    'sentence_id': sent_id,
                    'token_id': token_id
                })
                
                lemmas.append({
                    'token': token.text,
                    'lemma': token.lemma_,
                    'sentence_id': sent_id,
                    'token_id': token_id
                })
                
                dependencies.append({
                    'token': token.text,
                    'head': token.head.text if token.head != token else 'ROOT',
                    'deprel': token.dep_,
                    'sentence_id': sent_id,
                    'token_id': token_id
                })
            
            # Named entities for this sentence
            for ent in sent.ents:
                named_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start,
                    'end': ent.end,
                    'sentence_id': sent_id
                })
        
        # Morphological features
        verb_forms = []
        noun_forms = []
        adjective_forms = []
        
        for token in doc:
            if token.pos_ == 'VERB':
                verb_forms.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'feats': str(token.morph) if token.morph else None
                })
            elif token.pos_ == 'NOUN':
                noun_forms.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'feats': str(token.morph) if token.morph else None
                })
            elif token.pos_ == 'ADJ':
                adjective_forms.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'feats': str(token.morph) if token.morph else None
                })
        
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'lemmas': lemmas,
            'named_entities': named_entities,
            'dependencies': dependencies,
            'sentence_count': len(list(doc.sents)),
            'word_count': len([t for t in doc if not t.is_punct]),
            'morphological_features': {
                'verb_forms': verb_forms,
                'noun_forms': noun_forms,
                'adjective_forms': adjective_forms,
                'complexity_metrics': {}
            }
        }
    
    def extract_svc_linguistics(self, doc: Doc) -> Dict[str, Any]:
        """Extract SVC-specific linguistic analysis."""
        
        # Get first sentence for primary SVC
        sents = list(doc.sents)
        if not sents:
            return {}
        
        main_sent = sents[0]
        svc = self.svc_extractor.extract(main_sent)
        
        # Analyze each component
        result = {}
        
        # Subject analysis
        if svc.subject:
            result['subject_analysis'] = self._analyze_component(
                svc.subject, 'SUBJ', '[SUBJ]', doc
            )
        
        # Verb analysis
        if svc.verb:
            result['verb_analysis'] = self._analyze_component(
                svc.verb, 'VERB', '[VERB]', doc
            )
        
        # Complement analysis
        if svc.complement:
            result['complement_analysis'] = self._analyze_component(
                svc.complement, 'COMP', '[COMP]', doc
            )
        
        return result
    
    def _analyze_component(self, text: str, svc_type: str, tag: str, full_doc: Doc) -> Dict:
        """Analyze a single SVC component."""
        # Find tokens in original doc that match this component
        tokens = text.split()
        
        return {
            'component': text,
            'svc_type': svc_type,
            'structural_tag': tag,
            'tokens': tokens,
            'complexity_score': 1.0  # Placeholder
        }
    
    def generate_tagged_versions(self, doc: Doc, svc: SVCComponents, realm: str) -> Dict[str, Any]:
        """Generate various tagged representations."""
        
        text = doc.text
        
        # SVC full tagged
        svc_full_tagged = ""
        if svc.subject:
            svc_full_tagged += f"[SUBJ]{svc.subject}[/SUBJ] "
        if svc.verb:
            svc_full_tagged += f"[VERB]{svc.verb}[/VERB] "
        if svc.complement:
            svc_full_tagged += f"[COMP]{svc.complement}[/COMP]"
        
        # SVC simple tagged
        svc_simple_tagged = f"{svc.subject} [VERB] {svc.complement}" if svc.subject else text
        
        # SVC pattern
        pattern_parts = []
        if svc.subject:
            pattern_parts.append('[SUBJ]')
        if svc.verb:
            pattern_parts.append('[VERB]')
        if svc.complement:
            pattern_parts.append('[COMP]')
        svc_pattern = '-'.join(pattern_parts)
        
        # POS tagged text
        pos_tagged = ' '.join(f"{t.text}/{t.pos_}" for t in doc)
        
        # Lemma tagged text
        lemma_tagged = ' '.join(f"{t.text}→{t.lemma_}" for t in doc)
        
        # NER tagged text (just the raw text with entities marked)
        ner_tagged = text  # Could enhance this
        
        # Dependency pattern
        dep_pattern = [' '.join(f"{t.pos_}:{t.dep_}" for t in sent) for sent in doc.sents]
        
        # Semantic roles
        semantic_roles = {
            'agent': svc.subject,
            'action': svc.verb,
            'theme': svc.complement,
            'domain': realm.split('/')[0].title() if '/' in realm else realm.title(),
            'realm': realm
        }
        
        # Structural representation
        structural = {
            'svc_pattern': svc_pattern,
            'main_verbs': [t.lemma_ for t in doc if t.pos_ == 'VERB'][:3],
            'sentence_structure': '-'.join(f"[{t.pos_}]" for t in doc if t.pos_ in {'NOUN', 'VERB', 'PUNCT'})
        }
        
        return {
            'svc_full_tagged': svc_full_tagged.strip(),
            'svc_simple_tagged': svc_simple_tagged.strip(),
            'svc_pattern': svc_pattern,
            'pos_tagged_text': pos_tagged,
            'lemma_tagged_text': lemma_tagged,
            'ner_tagged_text': ner_tagged,
            'dependency_pattern': dep_pattern,
            'semantic_roles': semantic_roles,
            'structural_representation': json.dumps(structural)
        }
    
    def extract_structural_features(self, doc: Doc) -> Dict[str, Any]:
        """Extract structural features from the document."""
        
        # Verb tense info
        verb_tense_info = []
        for token in doc:
            if token.pos_ == 'VERB':
                morph = token.morph.to_dict()
                verb_tense_info.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tense': morph.get('Tense', [None])[0] if isinstance(morph.get('Tense'), list) else morph.get('Tense'),
                    'voice': morph.get('Voice', [None])[0] if isinstance(morph.get('Voice'), list) else morph.get('Voice'),
                    'mood': morph.get('Mood', [None])[0] if isinstance(morph.get('Mood'), list) else morph.get('Mood')
                })
        
        # Noun phrase structure
        noun_phrases = []
        for i, chunk in enumerate(doc.noun_chunks):
            noun_phrases.append(f"NOUN({i})")
        
        # Syntactic complexity
        depths = []
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 50:  # Prevent infinite loops
                    break
            depths.append(depth)
        
        avg_depth = sum(depths) / len(depths) if depths else 0
        clause_count = sum(1 for t in doc if t.dep_ in {'ccomp', 'xcomp', 'advcl', 'relcl', 'acl'}) + 1
        
        # SVC balance
        sents = list(doc.sents)
        if sents:
            svc = self.svc_extractor.extract(sents[0])
            total_len = len(svc.subject) + len(svc.verb) + len(svc.complement)
            if total_len > 0:
                svc_balance = {
                    'subject_ratio': len(svc.subject) / total_len,
                    'verb_ratio': len(svc.verb) / total_len,
                    'complement_ratio': len(svc.complement) / total_len,
                    'balance_score': 1.0 - abs(len(svc.subject) - len(svc.complement)) / total_len
                }
            else:
                svc_balance = {'subject_ratio': 0, 'verb_ratio': 0, 'complement_ratio': 0, 'balance_score': 0}
        else:
            svc_balance = {}
        
        # Discourse markers
        discourse_markers = []
        marker_words = {'however', 'therefore', 'furthermore', 'moreover', 'thus', 'hence', 
                       'consequently', 'nevertheless', 'nonetheless', 'meanwhile', 'otherwise'}
        for token in doc:
            if token.text.lower() in marker_words:
                discourse_markers.append(token.text.lower())
        
        return {
            'verb_tense_info': verb_tense_info,
            'noun_phrase_structure': noun_phrases,
            'syntactic_complexity': {
                'avg_dependency_depth': avg_depth,
                'clause_count': clause_count,
                'subordination_ratio': (clause_count - 1) / clause_count if clause_count > 0 else 0
            },
            'svc_balance': svc_balance,
            'discourse_markers': discourse_markers
        }


# =============================================================================
# Realm Classifier
# =============================================================================

class RealmClassifier:
    """Classify text into domain/realm categories."""
    
    REALM_KEYWORDS = {
        'science/biology': ['cell', 'organism', 'species', 'evolution', 'gene', 'protein', 
                           'photosynthesis', 'ecosystem', 'bacteria', 'virus', 'dna', 'rna'],
        'science/physics': ['energy', 'force', 'mass', 'particle', 'quantum', 'wave', 
                           'electron', 'atom', 'gravity', 'velocity', 'momentum'],
        'science/chemistry': ['molecule', 'compound', 'reaction', 'acid', 'base', 'element',
                             'bond', 'ion', 'catalyst', 'solution'],
        'technology/computing': ['algorithm', 'software', 'computer', 'data', 'program',
                                'code', 'network', 'server', 'database', 'api'],
        'technology/ai': ['artificial intelligence', 'machine learning', 'neural network',
                         'deep learning', 'model', 'training', 'inference'],
        'finance/economics': ['market', 'economy', 'price', 'trade', 'inflation', 'gdp',
                             'investment', 'stock', 'bond', 'currency'],
        'world/history': ['century', 'war', 'empire', 'king', 'queen', 'revolution',
                         'civilization', 'ancient', 'medieval', 'dynasty'],
        'culture/art': ['painting', 'sculpture', 'artist', 'museum', 'exhibition',
                       'masterpiece', 'canvas', 'portrait'],
        'culture/literature': ['novel', 'poem', 'author', 'writer', 'story', 'narrative',
                              'character', 'fiction', 'prose'],
        'culture/music': ['composer', 'symphony', 'melody', 'rhythm', 'instrument',
                         'orchestra', 'concert', 'song'],
        'linguistics': ['language', 'grammar', 'syntax', 'semantics', 'phonology',
                       'morphology', 'word', 'sentence', 'linguistic'],
    }
    
    def classify(self, text: str) -> str:
        """Classify text into a realm."""
        text_lower = text.lower()
        scores = Counter()
        
        for realm, keywords in self.REALM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[realm] += 1
        
        if scores:
            return scores.most_common(1)[0][0]
        return 'general'


# =============================================================================
# Main Converter
# =============================================================================

class SVCConverter:
    """Main converter class that orchestrates the conversion process."""
    
    def __init__(self, model_name: str = "en", lang: Optional[str] = None, gpu_id: int = 0):
        """
        Initialize the converter.

        Args:
            model_name: Stanza language code (e.g. 'en' for English).
            lang: Optional alias for model_name (ignored if model_name set).
            gpu_id: GPU device ID to use (-1 for CPU).
        """
        lang = lang or model_name
        logger.info(f"Loading Stanza pipeline: {lang}")

        use_gpu = gpu_id >= 0
        if use_gpu:
            logger.info(f"Using GPU {gpu_id}")

        processors = "tokenize,pos,lemma,depparse,ner"
        self.nlp = stanza.Pipeline(
            lang=lang,
            processors=processors,
            use_gpu=use_gpu,
            verbose=False,
        )

        self.feature_extractor = LinguisticFeatureExtractor()
        self.realm_classifier = RealmClassifier()
        self.svc_extractor = SVCExtractor()

        # Stats
        self.stats = Counter()
    
    def convert_text(self, text: str, text_id: str, realm: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Convert a single text to SVC enhanced format."""
        
        # Parse with Stanza and wrap as doc-like
        stanza_doc = self.nlp(text)
        doc = _StanzaDoc(stanza_doc, text)
        
        # Auto-classify realm if not provided
        if realm is None:
            realm = self.realm_classifier.classify(text)
        
        # Extract SVC from first sentence
        sents = list(doc.sents)
        if sents:
            svc = self.svc_extractor.extract(sents[0])
        else:
            svc = SVCComponents(subject=text, verb="", complement="")
        
        # Build the full structured entry
        entry = {
            'id': text_id,
            'text': text,
            'realm': realm,
            'language': 'en',
            'metadata': {
                'svc': {
                    'subject': svc.subject,
                    'verb': svc.verb,
                    'complement': svc.complement
                },
                'domain': realm.split('/')[0].title() if '/' in realm else realm.title(),
                'source': metadata.get('source', 'converted') if metadata else 'converted',
                'difficulty': 0.5  # Placeholder
            },
            'linguistic_features': self.feature_extractor.extract_all(doc, text_id),
            'svc_linguistics': self.feature_extractor.extract_svc_linguistics(doc),
            'tagged_versions': self.feature_extractor.generate_tagged_versions(doc, svc, realm),
            'structural_features': self.feature_extractor.extract_structural_features(doc)
        }
        
        # Merge additional metadata
        if metadata:
            entry['metadata'].update({k: v for k, v in metadata.items() if k not in entry['metadata']})
        
        return entry
    
    def process_batch(self, texts: List[Tuple[str, str, Optional[str], Optional[Dict]]],
                     batch_size: int = 100) -> Generator[Dict, None, None]:
        """
        Process a batch of texts with the Stanza pipeline.

        Args:
            texts: List of (text, text_id, realm, metadata) tuples
            batch_size: Unused (kept for API compatibility)

        Yields:
            Converted entries
        """
        # Separate texts and metadata
        text_list = [t[0] for t in texts]
        id_list = [t[1] for t in texts]
        realm_list = [t[2] for t in texts]
        meta_list = [t[3] for t in texts]
        
        # Process with Stanza (batch by calling pipeline on each text)
        for i, raw_text in enumerate(text_list):
            stanza_doc = self.nlp(raw_text)
            doc = _StanzaDoc(stanza_doc, raw_text)
            text_id = id_list[i]
            realm = realm_list[i]
            metadata = meta_list[i]
            
            # Auto-classify realm if not provided
            if realm is None:
                realm = self.realm_classifier.classify(text_list[i])
            
            # Extract SVC
            sents = list(doc.sents)
            if sents:
                svc = self.svc_extractor.extract(sents[0])
            else:
                svc = SVCComponents(subject=text_list[i], verb="", complement="")
            
            # Build entry
            entry = {
                'id': text_id,
                'text': text_list[i],
                'realm': realm,
                'language': 'en',
                'metadata': {
                    'svc': {
                        'subject': svc.subject,
                        'verb': svc.verb,
                        'complement': svc.complement
                    },
                    'domain': realm.split('/')[0].title() if '/' in realm else realm.title(),
                    'source': metadata.get('source', 'converted') if metadata else 'converted',
                    'difficulty': 0.5
                },
                'linguistic_features': self.feature_extractor.extract_all(doc, text_id),
                'svc_linguistics': self.feature_extractor.extract_svc_linguistics(doc),
                'tagged_versions': self.feature_extractor.generate_tagged_versions(doc, svc, realm),
                'structural_features': self.feature_extractor.extract_structural_features(doc)
            }
            
            if metadata:
                entry['metadata'].update({k: v for k, v in metadata.items() if k not in entry['metadata']})
            
            self.stats['processed'] += 1
            yield entry


# =============================================================================
# Input Parsers
# =============================================================================

def parse_temporal_dataset(filepath: Path) -> Generator[Tuple[str, str, str, Dict], None, None]:
    """Parse temporal_dataset.jsonl format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if data.get('type') == 'event':
                    event = data['data']
                    text = event.get('text', '')
                    if text and len(text) > 20:  # Skip very short texts
                        text_id = f"temporal_{event.get('id', i)}"
                        realm = 'world/history'  # NYT historical data
                        metadata = {
                            'source': 'nyt_archive',
                            'date': event.get('date'),
                            'year': event.get('year'),
                            'location': event.get('location'),
                            'section': event.get('section')
                        }
                        yield (text, text_id, realm, metadata)
            except json.JSONDecodeError:
                continue


def parse_instruct_dataset(filepath: Path) -> Generator[Tuple[str, str, str, Dict], None, None]:
    """Parse instruct.json format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        # Process input
        input_text = item.get('input', '')
        if input_text and len(input_text) > 20:
            text_id = f"instruct_input_{i}"
            realm = 'technology/computing'  # Technical instructions
            metadata = {
                'source': 'instruct',
                'type': 'input',
                'conversation_id': item.get('conversation_id')
            }
            yield (input_text, text_id, realm, metadata)
        
        # Process output (split into sentences for better granularity)
        output_text = item.get('output', '')
        if output_text:
            # Split by sentences (rough)
            sentences = re.split(r'(?<=[.!?])\s+', output_text)
            for j, sent in enumerate(sentences):
                if len(sent) > 30 and len(sent) < 500:  # Reasonable sentence length
                    text_id = f"instruct_output_{i}_{j}"
                    metadata = {
                        'source': 'instruct',
                        'type': 'output',
                        'conversation_id': item.get('conversation_id')
                    }
                    yield (sent, text_id, realm, metadata)


def parse_conversations_dataset(filepath: Path) -> Generator[Tuple[str, str, str, Dict], None, None]:
    """Parse conversations_dataset.jsonl format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                for j, msg in enumerate(messages):
                    content = msg.get('content', '')
                    role = msg.get('role', 'unknown')
                    
                    # Split long content into sentences
                    if content:
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        for k, sent in enumerate(sentences):
                            if len(sent) > 30 and len(sent) < 500:
                                text_id = f"conv_{i}_{j}_{k}"
                                realm = 'technology/computing'
                                metadata = {
                                    'source': 'conversations',
                                    'role': role,
                                    'conversation_id': str(i)
                                }
                                yield (sent, text_id, realm, metadata)
            except json.JSONDecodeError:
                continue


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_file(converter: SVCConverter, input_path: Path, output_path: Path,
                parser_func, batch_size: int = 100, max_entries: Optional[int] = None,
                checkpoint_interval: int = 10000):
    """Process a single input file."""
    
    logger.info(f"Processing: {input_path}")
    
    # Load checkpoint if exists
    checkpoint_file = output_path.with_suffix('.checkpoint')
    processed_ids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            processed_ids = set(line.strip() for line in f)
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")
    
    # Open output file in append mode
    mode = 'a' if processed_ids else 'w'
    
    batch = []
    total_processed = 0
    
    with open(output_path, mode, encoding='utf-8') as out_f, \
         open(checkpoint_file, 'a') as ckpt_f:
        
        for text, text_id, realm, metadata in tqdm(parser_func(input_path), desc=str(input_path.name)):
            # Skip if already processed
            if text_id in processed_ids:
                continue
            
            # Check max entries
            if max_entries and total_processed >= max_entries:
                break
            
            batch.append((text, text_id, realm, metadata))
            
            # Process batch when full
            if len(batch) >= batch_size:
                for entry in converter.process_batch(batch, batch_size=batch_size):
                    out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    ckpt_f.write(entry['id'] + '\n')
                    total_processed += 1
                    
                    # Flush periodically
                    if total_processed % checkpoint_interval == 0:
                        out_f.flush()
                        ckpt_f.flush()
                        logger.info(f"Checkpoint: {total_processed} entries processed")
                
                batch = []
        
        # Process remaining batch
        if batch:
            for entry in converter.process_batch(batch, batch_size=batch_size):
                out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                ckpt_f.write(entry['id'] + '\n')
                total_processed += 1
    
    logger.info(f"Completed: {total_processed} entries written to {output_path}")
    return total_processed


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to SVC enhanced format')
    parser.add_argument('--input', type=str, required=True, help='Input directory or file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--max-entries', type=int, default=None, help='Max entries per file')
    parser.add_argument('--model', type=str, default='en', help='Stanza language code (e.g. en)')
    parser.add_argument('--files', type=str, nargs='*', help='Specific files to process')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize converter
    converter = SVCConverter(model_name=args.model, gpu_id=args.gpu)
    
    # File type to parser mapping
    parsers = {
        'temporal_dataset.jsonl': parse_temporal_dataset,
        'temporal_dataset_events.jsonl': parse_temporal_dataset,
        'temporal_dataset_associations.jsonl': parse_temporal_dataset,
        'instruct_anonymized_cleaned.json': parse_instruct_dataset,
        'conversations_dataset_anonymized_cleaned.jsonl': parse_conversations_dataset,
    }
    
    # Process files
    if input_path.is_file():
        # Single file
        parser_func = parsers.get(input_path.name)
        if parser_func is None:
            logger.error(f"Unknown file format: {input_path.name}")
            sys.exit(1)
        
        out_file = output_path / f"{input_path.stem}_svc_enhanced.jsonl"
        process_file(converter, input_path, out_file, parser_func,
                    batch_size=args.batch_size, max_entries=args.max_entries)
    
    elif input_path.is_dir():
        # Directory - process all known files
        files_to_process = args.files if args.files else parsers.keys()
        
        for filename in files_to_process:
            filepath = input_path / filename
            if filepath.exists() and filename in parsers:
                out_file = output_path / f"{filepath.stem}_svc_enhanced.jsonl"
                process_file(converter, filepath, out_file, parsers[filename],
                            batch_size=args.batch_size, max_entries=args.max_entries)
    
    # Print final stats
    logger.info(f"Conversion complete. Stats: {dict(converter.stats)}")


if __name__ == '__main__':
    main()
