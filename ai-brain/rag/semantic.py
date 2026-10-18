"""Local embedding-based semantic retrieval with source metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SemanticChunk:
    """A text chunk linked to its original document and optional page."""

    source: str
    page: Optional[int]
    text: str


@dataclass(frozen=True)
class SemanticSearchResult:
    """A retrieved chunk and its cosine similarity score."""

    source: str
    page: Optional[int]
    text: str
    score: float


class SemanticVectorRAG:
    """Local semantic search; model loading is deferred until needed."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model: Any = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.model_name = model_name
        self._model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[SemanticChunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers. "
                    "Install it with: python -m pip install sentence-transformers"
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model

    def _split_text(self, text: str) -> List[str]:
        """Keep financial labels with values and normalise simple financial tables."""
        import re

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalised_lines: List[str] = []
        i = 0

        # Handle explicit two-column financial tables such as:
        # Metric / Value (Crore) / Revenue / 100 / Total Debt / 90.
        while i < len(lines):
            if (
                lines[i].casefold() == "metric"
                and i + 1 < len(lines)
                and lines[i + 1].casefold().startswith("value")
            ):
                value_header = lines[i + 1]
                unit = (
                    " Crore"
                    if re.search(r"\bcrore\b", value_header, re.IGNORECASE)
                    else ""
                )
                i += 2

                while i < len(lines):
                    if lines[i].casefold().startswith("notes"):
                        break

                    # Consume only clear label/value pairs.
                    if i + 1 >= len(lines):
                        normalised_lines.append(lines[i])
                        i += 1
                        continue

                    label, value = lines[i], lines[i + 1]

                    # A section heading or separator ends this table.
                    if (
                        set(label) <= {"-", "—", "_", " "}
                        or label.endswith(":")
                    ):
                        break

                    if (
                        value.casefold().startswith("notes")
                        or set(value) <= {"-", "—", "_", " "}
                    ):
                        normalised_lines.append(label)
                        i += 1
                        continue

                    suffix = unit if unit and not re.search(
                        r"\bcrore\b", value, re.IGNORECASE
                    ) else ""
                    normalised_lines.append(f"{label}: {value}{suffix}")
                    i += 2

                continue

            normalised_lines.append(lines[i])
            i += 1

        # Pair a financial heading with the following numeric value line,
        # but do not merge arbitrary adjacent financial metrics.
        paired_lines: List[str] = []
        numeric_start = re.compile(
            r"^(?:~\s*)?(?:(?:INR|RS\.?)\s*)?"
            r"[\d,]+(?:\.\d+)?"
            r"(?:\s*(?:-|–|to)\s*[\d,]+(?:\.\d+)?)?\b",
            re.IGNORECASE,
        )

        i = 0
        while i < len(normalised_lines):
            current = normalised_lines[i]

            metric_heading = bool(
                re.match(
                    r"^(?:total\s+)?(?:revenue|turnover|sales|net\s+profit|"
                    r"gross\s+profit|ebitda|total\s+debt|debt|borrowings|"
                    r"receivables|payables|bank\s+deposits|operating\s+cash\s+flow|"
                    r"cash\s+flow\s+from\s+operations)\s*:?$",
                    current,
                    re.IGNORECASE,
                )
            )

            if (
                (current.endswith(":") or metric_heading)
                and i + 1 < len(normalised_lines)
                and numeric_start.match(normalised_lines[i + 1])
            ):
                heading = current.rstrip(":").strip()
                value = normalised_lines[i + 1].strip()
                paired_lines.append(f"{heading}: {value}")
                i += 2
            else:
                paired_lines.append(current)
                i += 1

        # Preserve short lines; split oversized lines at sentence boundaries
        # and then at whitespace when necessary.
        chunks: List[str] = []

        for line in paired_lines:
            if len(line) <= self.chunk_size:
                chunks.append(line)
                continue

            protected = re.sub(
                r"\bRs\.\s*",
                "Rs<RS_ABBREVIATION> ",
                line,
                flags=re.IGNORECASE,
            )
            sentences = [
                part.replace("<RS_ABBREVIATION>", ".").strip()
                for part in re.split(r"(?<=[.!?])\s+", protected)
                if part.strip()
            ]

            for sentence in sentences:
                while len(sentence) > self.chunk_size:
                    end = self.chunk_size
                    boundary = sentence.rfind(
                        " ", self.chunk_size // 2, self.chunk_size
                    )
                    if boundary > 0:
                        end = boundary

                    part = sentence[:end].strip()
                    if part:
                        chunks.append(part)

                    sentence = sentence[
                        max(1, end - self.chunk_overlap):
                    ].strip()

                if sentence:
                    chunks.append(sentence)

        return chunks

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Replace the index only after new embeddings validate successfully.

        Each record should contain 'text' and 'source'. 'page' is optional.
        Page numbers should be 1-based when known, otherwise None.
        """
        chunks: List[SemanticChunk] = []

        for document in documents:
            text = str(document.get("text") or "")
            source = str(document.get("source") or "unknown")
            page_value = document.get("page")

            page: Optional[int] = None
            if page_value is not None:
                try:
                    parsed_page = int(page_value)
                    page = parsed_page if parsed_page > 0 else None
                except (TypeError, ValueError):
                    page = None

            for part in self._split_text(text):
                chunks.append(
                    SemanticChunk(source=source, page=page, text=part)
                )

        # An intentionally empty re-index clears the previous index.
        if not chunks:
            self._chunks = []
            self._embeddings = None
            return 0

        # Build and validate locally. Any exception leaves the old index intact.
        vectors = self._get_model().encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(vectors, dtype=np.float32)

        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(chunks)
            or embeddings.shape[1] == 0
        ):
            raise ValueError("Embedding model returned an invalid matrix shape")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding model returned non-finite values")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Embedding model returned a zero-length vector")

        # Normalize defensively so dot products represent cosine similarity.
        embeddings = embeddings / norms

        # Commit the new index only after every validation passes.
        self._chunks = chunks
        self._embeddings = embeddings
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SemanticSearchResult]:
        """Return the most semantically similar indexed chunks."""
        if top_k <= 0:
            return []
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")
        if not query or not query.strip():
            return []
        if not self._chunks or self._embeddings is None:
            return []

        # Expand common financial metric queries with relevant synonyms.
        metric_queries = {
            "revenue": [
                "Revenue",
                "Total Turnover",
                "Sales",
                "GST declared revenue",
            ],
            "turnover": [
                "Turnover",
                "Revenue",
                "Sales",
                "GST declared revenue",
            ],
            "sales": [
                "Sales",
                "Revenue",
                "Total Turnover",
            ],
            "debt": [
                "Debt",
                "Total Borrowings",
                "Loans and borrowings",
            ],
            "borrowings": [
                "Borrowings",
                "Total Debt",
                "Loans and debt",
            ],
            "operating cash flow": [
                "Operating Cash Flow",
                "Cash Flow from Operations",
            ],
            "receivables": [
                "Receivables",
                "Trade Receivables",
                "Outstanding customer payments",
            ],
        }

        normalized_query = query.strip().lower()
        query_variants = metric_queries.get(
            normalized_query, [query.strip()]
        )

        query_vectors = self._get_model().encode(
            query_variants,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embeddings = np.asarray(query_vectors, dtype=np.float32)

        if (
            query_embeddings.ndim != 2
            or query_embeddings.shape[0] != len(query_variants)
            or query_embeddings.shape[1] != self._embeddings.shape[1]
        ):
            raise ValueError("Query embedding has an incompatible shape")

        if not np.isfinite(query_embeddings).all():
            raise ValueError("Embedding model returned non-finite values")

        query_norms = np.linalg.norm(query_embeddings, axis=1)
        if np.any(query_norms <= 1e-12):
            raise ValueError(
                "Embedding model returned a zero-length query vector"
            )

        query_embeddings = query_embeddings / query_norms[:, None]

        # Preserve raw cosine similarities for returned result scores.
        scores = (self._embeddings @ query_embeddings.T).max(axis=1)

        # Prefer explicit numeric labels, while distinguishing
        # reported revenue from GST-declared revenue.
        ranking_scores = scores.copy()
        normalized_query = " ".join(query.casefold().split())

        numeric_value = (
            r"\s*(?:\([^)]*\))?\s*[:~—-]?\s*"
            r"(?:approx(?:imately)?\.?\s*)?"
            r"(?:INR\s*|Rs\.?\s*|₹\s*)?"
            r"[-+]?\d[\d,]*(?:\.\d+)?"
        )

        label_pattern = None

        if normalized_query in {
            "gst revenue",
            "gst declared revenue",
            "gst-declared revenue",
        }:
            label_pattern = (
                r"(?i)^\s*GST\s+Declared\s+Revenue"
                + numeric_value
            )

        elif normalized_query in {"revenue", "turnover", "sales"}:
            # Match a general reported-revenue label only at the
            # beginning of the chunk. This prevents
            # "GST Declared Revenue" from receiving this boost.
            label_pattern = (
                r"(?i)^\s*(?:(?:total)\s+)?"
                r"(?:revenue|turnover|sales)"
                + numeric_value
            )

        else:
            metric_labels = {
                "total debt": r"total debt|debt|borrowings",
                "debt": r"total debt|debt|borrowings",
                "borrowings": r"borrowings|debt",
                "operating cash flow": (
                    r"operating cash flow|cash flow from operations|"
                    r"cash flow from ops"
                ),
                "receivables": r"receivables|trade receivables",
            }

            metric_pattern = metric_labels.get(normalized_query)

            if metric_pattern:
                label_pattern = (
                    rf"(?i)\b(?:{metric_pattern})\b"
                    + numeric_value
                )

        if label_pattern:
            for index, chunk in enumerate(self._chunks):
                if re.search(label_pattern, chunk.text.strip()):
                    ranking_scores[index] += 0.25

        ranked_indices = np.argsort(-ranking_scores)

        results: List[SemanticSearchResult] = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < min_score:
                continue

            chunk = self._chunks[int(index)]
            results.append(
                SemanticSearchResult(
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text,
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results
