"""Text preparation shared by the retrievers."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

from rag_ablation_bench.schema import Chunk

# Function words that carry no retrieval signal. Negations ("no", "sin") are kept on purpose:
# in legal text they flip the meaning of a provision.
SPANISH_STOPWORDS = frozenset(
    """
    a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde
    donde durante e el ella ellas ellos en entre era eran es esa esas ese eso esos esta
    estas este esto estos fue fueron ha han hasta la las le les lo los mas me mi mis muy
    nos o os otra otras otro otros para pero por que quien quienes se sea sean ser si
    sido sobre su sus tambien te tiene tienen u un una unas uno unos y ya yo
    """.split()
)
TOKEN = re.compile(r"\w+", re.UNICODE)
_stemmer = snowballstemmer.stemmer("spanish")


def fold_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def tokenize(text: str, *, stem: bool = True, stopwords: bool = True) -> list[str]:
    """Lowercase word tokens; accents are folded because users often type without them."""
    tokens = TOKEN.findall(text.lower())
    if stopwords:
        tokens = [t for t in tokens if fold_accents(t) not in SPANISH_STOPWORDS]
    if stem:  # the Snowball stemmer expects accented Spanish, so fold afterwards
        tokens = [_stem(t) for t in tokens]
    return [fold_accents(t) for t in tokens]


def chunk_text(chunk: Chunk, *, header: bool) -> str:
    """What gets indexed: optionally prefix the chunk with its location in the document."""
    return f"{chunk.header}\n{chunk.text}" if header else chunk.text
