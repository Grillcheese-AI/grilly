"""
grilly.experimental.language - Instant language learning with VSA.

Provides instant word encoding, sentence composition, parsing, and generation
using Vector Symbolic Architectures. No training required!

Submodules:
    - encoder: WordEncoder and SentenceEncoder
    - generator: SentenceGenerator
    - parser: ResonatorParser
    - system: InstantLanguage (unified API)
"""

from .encoder import WordEncoder, SentenceEncoder
from .generator import SentenceGenerator
from .parser import ResonatorParser
from .system import InstantLanguage

__all__ = [
    "WordEncoder",
    "SentenceEncoder",
    "SentenceGenerator",
    "ResonatorParser",
    "InstantLanguage",
]
