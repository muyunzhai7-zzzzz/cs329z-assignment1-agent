"""Lazy shared document index and a bounded observation budget per request."""

from __future__ import annotations

import json
from threading import RLock

from cs329z_hw1 import data
from cs329z_hw1.tokens import count_tokens


RESULT_BUDGET = 6000
BUDGET_NOTICE = "Document result budget reached. Answer using the evidence already retrieved, or explain what remains unknown."


class DocumentTools:
    def __init__(self) -> None:
        self._lock = RLock()
        self._index = None
        self.reset_budget()

    def reset_budget(self) -> None:
        self._remaining = RESULT_BUDGET
        self._notice_sent = False

    def _get_index(self):
        from cs329z_hw1.pipelines.search_docs import build_document_index

        with self._lock:
            if self._index is None:
                self._index = build_document_index(data.load_docs())
            return self._index

    @staticmethod
    def _encode(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _available(self) -> int:
        return max(0, self._remaining - count_tokens(BUDGET_NOTICE))

    def _emit(self, text: str) -> str:
        size = count_tokens(text)
        if size <= self._available():
            self._remaining -= size
            return text
        return self._budget_notice()

    def _budget_notice(self) -> str:
        if self._notice_sent:
            return ""
        self._notice_sent = True
        self._remaining -= count_tokens(BUDGET_NOTICE)
        return BUDGET_NOTICE

    def search(self, query: str, k: int = 5) -> str:
        from cs329z_hw1.pipelines.search_docs import search_documents

        if self._notice_sent:
            return ""
        results = search_documents(self._get_index(), query, k)
        had_results = bool(results)
        while results and count_tokens(self._encode(results)) > self._available():
            results.pop()
        if had_results and not results:
            return self._budget_notice()
        return self._emit(self._encode(results))

    def read(self, doc_id: str, start_line: int = 1) -> str:
        from cs329z_hw1.pipelines.search_docs import read_document

        if self._notice_sent:
            return ""
        index = self._get_index()
        window = read_document(index, doc_id, start_line)
        while window["end_line"] > start_line and count_tokens(self._encode(window)) > self._available():
            window["end_line"] -= 1
            window["text"] = "\n".join(index.documents[doc_id].lines[start_line - 1:window["end_line"]])
        return self._emit(self._encode(window))
