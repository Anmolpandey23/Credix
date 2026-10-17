"""Controlled vector-style retrieval for text insights only."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import List

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, MAX_INSIGHTS
from models.schemas import InsightResult
from utils.text_utils import clean_extracted_text, sentence_split, tokenize

RISK_TERMS = {
    "delay",
    "default",
    "overdue",
    "litigation",
    "tax",
    "demand",
    "penalty",
    "insolvency",
    "fraud",
    "stress",
    "collection",
    "cash",
    "receivable",
    "debt",
}


class ControlledVectorRAG:
    def extract_insights(self, raw_text: str) -> InsightResult:
        cleaned_text = clean_extracted_text(raw_text)
        chunks = self._chunk_text(cleaned_text)
        if not chunks:
            return InsightResult(insights=[], source_chunks=[])

        query_vector = self._vectorize(" ".join(sorted(RISK_TERMS)))
        ranked = sorted(
            ((self._cosine_similarity(query_vector, self._vectorize(chunk)), chunk) for chunk in chunks),
            key=lambda item: item[0],
            reverse=True,
        )

        selected_chunks = [chunk for score, chunk in ranked if score > 0][:6]
        insights: List[str] = []

        for chunk in selected_chunks:
            for sentence in sentence_split(chunk):
                if len(insights) >= MAX_INSIGHTS:
                    break
                if self._is_risk_related_sentence(sentence) and self._is_complete_sentence(sentence):
                    insights.append(sentence)

        deduped_insights = list(dict.fromkeys(insights))[:MAX_INSIGHTS]
        deduped_chunks = list(dict.fromkeys(selected_chunks))

        return InsightResult(insights=deduped_insights, source_chunks=deduped_chunks)

    def _chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)

        return [chunk for chunk in chunks if chunk]

    def _vectorize(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        return Counter(tokens)

    def _cosine_similarity(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0

        common = set(left.keys()) & set(right.keys())
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return numerator / (left_norm * right_norm)

    def _is_risk_related_sentence(self, sentence: str) -> bool:
        tokens = set(tokenize(sentence))
        return bool(tokens & RISK_TERMS)

    def _is_complete_sentence(self, sentence: str) -> bool:
        stripped = sentence.strip()
        if len(stripped) < 20:
            return False
        if stripped[0].islower() and not stripped.endswith((".", "!", "?")):
            return False
        return True
