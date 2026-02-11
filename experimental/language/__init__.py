"""
grilly.experimental.language - Instant language learning with VSA.

Provides instant word encoding, sentence composition, parsing, and generation
using Vector Symbolic Architectures. No training required!

Submodules:
    - encoder: WordEncoder and SentenceEncoder
    - generator: SentenceGenerator
    - parser: ResonatorParser
    - system: InstantLanguage (unified API)
    - svc_loader: SVC data loading utilities
"""

from .encoder import SentenceEncoder, WordEncoder
from .generator import SentenceGenerator
from .parser import ResonatorParser
from .svc_loader import (
    SVCBatch,
    SVCEntry,
    SVCIngestionEngine,
    load_svc_batch,
    load_svc_entries,
    load_svc_entries_from_dicts,
)
from .system import InstantLanguage, SVCIngestionResult

__all__ = [
    "WordEncoder",
    "SentenceEncoder",
    "SentenceGenerator",
    "ResonatorParser",
    "InstantLanguage",
    "SVCIngestionResult",
    "SVCEntry",
    "SVCBatch",
    "SVCIngestionEngine",
    "load_svc_entries",
    "load_svc_batch",
    "load_svc_entries_from_dicts",
]
