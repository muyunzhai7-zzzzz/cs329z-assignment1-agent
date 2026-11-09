"""BM25 retrieval over an inverted index, without model or search libraries."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
import math

import numpy as np

from cs329z_hw1.tokenizer import tokenize


K1 = 1.5
B = 0.75


@dataclass(frozen=True)
class BM25Index:
    """Built once; searches only read these statistics and postings."""

    document_count: int
    average_length: float
    # Each term maps to its [start, end) range in the two compact arrays.
    postings: dict[str, tuple[int, int]]
    doc_ids: np.ndarray
    term_scores: np.ndarray


def build_index(docs: list[str]) -> BM25Index:
    """Read each document once, keeping empty documents in corpus statistics."""
    frequencies: dict[str, list[int]] = defaultdict(list)
    lengths: list[int] = []
    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        lengths.append(len(tokens))
        # One posting per document/term; repeated tokens increase its frequency.
        for term, frequency in Counter(tokens).items():
            frequencies[term].extend((doc_id, frequency))

    document_count = len(lengths)
    average_length = sum(lengths) / document_count if document_count else 0.0
    if not frequencies:
        return BM25Index(document_count, average_length, {}, np.array([], dtype=np.intp), np.array([], dtype=float))

    postings: dict[str, tuple[int, int]] = {}
    counts: list[int] = []
    weights: list[float] = []
    offset = 0
    for term, entries in frequencies.items():
        count = len(entries) // 2
        postings[term] = (offset, offset + count)
        offset += count
        counts.append(count)
        weights.append(math.log1p((document_count - count + 0.5) / (count + 0.5)) * (K1 + 1))

    # Flatten postings once, then evaluate the same formula in bulk. The corpus
    # itself is still read/tokenized only once. NumPy is provided by the course.
    pairs = np.fromiter(chain.from_iterable(frequencies.values()), dtype=np.intp, count=2 * offset).reshape(-1, 2)
    doc_ids = pairs[:, 0].copy()
    term_frequencies = pairs[:, 1]
    length_norms = K1 * (1 - B + B * np.asarray(lengths) / average_length)
    term_scores = (
        np.repeat(weights, counts) * term_frequencies
        / (term_frequencies + length_norms[doc_ids])
    )
    return BM25Index(document_count, average_length, postings, doc_ids, term_scores)


def search_index(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    """Score only matching documents, then select top-k with stable tie breaks."""
    if k <= 0:
        return []

    # Preserve token order while counting every distinct query term just once.
    matching_ids: list[np.ndarray] = []
    matching_scores: list[np.ndarray] = []
    for term in dict.fromkeys(tokenize(query)):
        bounds = index.postings.get(term)
        if bounds is None:
            continue
        start, end = bounds
        matching_ids.append(index.doc_ids[start:end])
        matching_scores.append(index.term_scores[start:end])
    if not matching_ids:
        return []

    # Aggregate only these postings, never allocate or scan one entry per corpus
    # document. Even a huge corpus is cheap to search for a genuinely rare term.
    doc_ids, groups = np.unique(np.concatenate(matching_ids), return_inverse=True)
    scores = np.bincount(groups, weights=np.concatenate(matching_scores))
    best = np.lexsort((doc_ids, -scores))[:k]
    return [(int(doc_ids[position]), float(scores[position])) for position in best]
