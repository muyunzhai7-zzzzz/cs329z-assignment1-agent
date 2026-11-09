"""Q9: chunked BM25 retrieval with source lines and explicitly stated dates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from cs329z_hw1.pipelines.bm25 import BM25Index, build_index, search_index
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_tokens


CHUNK_TOKENS = 320
OVERLAP_TOKENS = 60
SNIPPET_TOKENS = 150
READ_TOKENS = 800
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
SNIPPET_CHARS = SNIPPET_TOKENS * CHARS_PER_TOKEN
HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
WORD = re.compile(r"[a-z0-9]+")
HEADER_DATE = re.compile(
    r"^\s*(Effective|Posted|Published|Date|Updated|Last updated|As of):\s*(\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


@dataclass
class Document:
    doc_id: str
    title: str
    lines: list[str]
    dates: list[dict]


@dataclass
class Chunk:
    doc_id: str
    section: str
    start_line: int
    end_line: int
    text: str


@dataclass
class DocumentIndex:
    documents: dict[str, Document]
    chunks: list[Chunk]
    bm25: BM25Index


def _source_dates(lines: list[str]) -> list[dict]:
    """Keep date labels and provenance; never use filenames or filesystem dates."""
    dates = []
    for number, line in enumerate(lines, 1):
        if line.startswith("##"):
            break
        match = HEADER_DATE.match(line)
        if match:
            label, value = match.groups()
            try:
                date.fromisoformat(value)
            except ValueError:
                continue
            label = label.lower().replace(" ", "_")
            if label == "last_updated":
                label = "updated"
            dates.append({"kind": label, "value": value, "line": number})

    # Some announcements state their effective date in a sentence, rather
    # than an Effective header. Only recognize explicit forward-applicability
    # wording, not dates embedded in references to the superseded policy.
    if not any(item["kind"] == "effective" for item in dates):
        patterns = (
            re.compile(r"^\s*(?:Starting|From)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
            re.compile(r"\bappl(?:y|ies)\b[^.]*?\bon or after\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        )
        for number, line in enumerate(lines, 1):
            match = next((match for pattern in patterns if (match := pattern.search(line))), None)
            if match:
                value = match.group(1)
                try:
                    date.fromisoformat(value)
                except ValueError:
                    continue
                dates.append({"kind": "effective", "value": value, "line": number})
                break
    return dates


def _sections(lines: list[str]):
    start = 0
    path: list[tuple[int, str]] = []
    for position, line in enumerate(lines):
        match = HEADING.match(line)
        if match:
            if position > start:
                yield start, position, " > ".join(title for _, title in path)
            level = len(match.group(1))
            path = [(depth, title) for depth, title in path if depth < level]
            path.append((level, match.group(2)))
            start = position
    if start < len(lines):
        yield start, len(lines), " > ".join(title for _, title in path)


def _chunks(document: Document) -> list[Chunk]:
    result = []
    lines = document.lines
    for section_start, section_end, section in _sections(lines):
        start = section_start
        while start < section_end:
            if len(lines[start]) > CHUNK_CHARS:
                # Preserve source line identity even for an unusually long
                # paragraph; prefer splitting at a word boundary.
                text = lines[start]
                offset = 0
                while offset < len(text):
                    end = min(offset + CHUNK_CHARS, len(text))
                    if end < len(text):
                        space = text.rfind(" ", offset, end)
                        if space > offset:
                            end = space
                    result.append(Chunk(document.doc_id, section, start + 1, start + 1, text[offset:end]))
                    offset = end
                    while offset < len(text) and text[offset].isspace():
                        offset += 1
                start += 1
                continue

            end, size = start, 0
            while end < section_end:
                added = len(lines[end]) + (1 if end > start else 0)
                if size + added > CHUNK_CHARS:
                    break
                size += added
                end += 1
            text = "\n".join(lines[start:end])
            if text.strip():
                result.append(Chunk(document.doc_id, section, start + 1, end, text))

            following = end
            if end < section_end:
                overlap = end - 1
                while overlap > start and not lines[overlap].strip():
                    overlap -= 1
                overlap_size = len("\n".join(lines[overlap:end]))
                if (
                    overlap > start and overlap_size <= OVERLAP_TOKENS * CHARS_PER_TOKEN
                    and overlap_size + 1 + len(lines[end]) <= CHUNK_CHARS
                ):
                    following = overlap
            start = following
    return result


def build_document_index(docs: list[dict]) -> DocumentIndex:
    documents = {}
    chunks = []
    texts = []
    for source in docs:
        lines = source["text"].splitlines()
        document = Document(source["doc_id"], source["title"], lines, _source_dates(lines))
        documents[document.doc_id] = document
        for chunk in _chunks(document):
            chunks.append(chunk)
            texts.append(document.title + "\n" + chunk.section + "\n" + chunk.text)
    return DocumentIndex(documents, chunks, build_index(texts))


def _snippet(text: str, query_terms: set[str]) -> tuple[str, int]:
    if len(text) <= SNIPPET_CHARS:
        return text.strip(), len(text) - len(text.lstrip())
    hits = [match for match in WORD.finditer(text.lower()) if match.group() in query_terms]
    best = None
    for hit in hits:
        left = max(0, min(hit.start() - 100, len(text) - SNIPPET_CHARS))
        while left < hit.start() and left > 0 and text[left - 1].isalnum() and text[left].isalnum():
            left += 1
        right = min(len(text), left + SNIPPET_CHARS)
        while right > hit.end() and right < len(text) and text[right - 1].isalnum() and text[right].isalnum():
            right -= 1
        window = text[left:right]
        score = len(query_terms.intersection(tokenize(window)))
        candidate = (score, -left, window, left)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    _, _, window, left = best if best is not None else (0, 0, text[:SNIPPET_CHARS], 0)
    return window.strip(), left + len(window) - len(window.lstrip())


def search_documents(index: DocumentIndex, query: str, k: int = 5) -> list[dict]:
    if k <= 0:
        return []
    terms = set(tokenize(query))
    if not terms:
        return []
    results, seen = [], set()
    for position, _ in search_index(index.bm25, query, len(index.chunks)):
        chunk = index.chunks[position]
        if chunk.doc_id in seen or not terms.intersection(tokenize(chunk.text)):
            continue
        document = index.documents[chunk.doc_id]
        snippet, offset = _snippet(chunk.text, terms)
        start = chunk.start_line + chunk.text[:offset].count("\n")
        results.append({
            "doc_id": document.doc_id, "title": document.title, "snippet": snippet,
            "start_line": start, "end_line": start + snippet.count("\n"),
            "section": chunk.section, "chunk_id": position,
            "chunk_start_line": chunk.start_line, "chunk_end_line": chunk.end_line,
            "dates": [dict(item) for item in document.dates],
        })
        seen.add(document.doc_id)
        if len(results) == k:
            break
    return results


def read_document(index: DocumentIndex, doc_id: str, start_line: int = 1) -> dict:
    if doc_id not in index.documents:
        raise LookupError(f"Unknown document ID {doc_id!r}; use a doc_id returned by search.")
    document = index.documents[doc_id]
    total = len(document.lines)
    if not 1 <= start_line <= total:
        raise ValueError(f"start_line must be between 1 and {total} for {doc_id!r}.")
    end = start_line - 1
    size = 0
    while end < total:
        added = len(document.lines[end]) + (1 if end >= start_line else 0)
        if size + added > READ_TOKENS * CHARS_PER_TOKEN:
            break
        size += added
        end += 1
    if end < start_line:
        raise ValueError("The requested source line is too large for a bounded, whole-line read.")
    text = "\n".join(document.lines[start_line - 1:end])
    assert count_tokens(text) <= READ_TOKENS
    return {
        "doc_id": doc_id, "title": document.title, "start_line": start_line,
        "end_line": end, "total_lines": total, "text": text,
        "dates": [dict(item) for item in document.dates],
    }
