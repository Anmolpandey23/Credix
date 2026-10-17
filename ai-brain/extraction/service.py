"""Deterministic document extraction without numeric transformation."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, List

from utils.text_utils import clean_extracted_text, normalize_whitespace


class ExtractionService:
    def extract_documents(
        self, file_paths: Iterable[Path]
    ) -> List[dict[str, str | int | None]]:
        """Extract document segments while retaining available source locations.

        PDF segments retain their 1-based page number when text is available.
        CSV and JSON segments retain their filename; page is not applicable.
        If PDF page extraction fails, fall back to the existing extraction
        path and explicitly leave the page unknown.
        """
        documents: List[dict[str, str | int | None]] = []

        for file_path in file_paths:
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                try:
                    from pypdf import PdfReader  # type: ignore

                    logging.getLogger("pypdf").setLevel(logging.ERROR)
                    reader = PdfReader(str(file_path))
                    pages = [
                        (page_number, clean_extracted_text(page.extract_text() or ""))
                        for page_number, page in enumerate(reader.pages, start=1)
                    ]

                    usable_pages = [
                        (page_number, text)
                        for page_number, text in pages
                        if normalize_whitespace(text)
                    ]

                    if usable_pages:
                        documents.extend(
                            {
                                "source": file_path.name,
                                "page": page_number,
                                "text": text,
                            }
                            for page_number, text in usable_pages
                        )
                        continue
                except Exception:
                    pass

                # Existing extraction may use OCR. Do not guess its page
                # mapping if the fallback returns combined document text.
                fallback_text = self.extract(file_path)
                if normalize_whitespace(fallback_text):
                    documents.append(
                        {
                            "source": file_path.name,
                            "page": None,
                            "text": fallback_text,
                        }
                    )
                continue

            text = self.extract(file_path)
            if normalize_whitespace(text):
                documents.append(
                    {
                        "source": file_path.name,
                        "page": None,
                        "text": text,
                    }
                )

        return documents

    def extract_many(self, file_paths: Iterable[Path]) -> str:
        documents: List[str] = []
        for file_path in file_paths:
            documents.append(self.extract(file_path))
        return clean_extracted_text("\n".join(part for part in documents if part.strip()))

    def extract(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        if suffix == ".csv":
            return self._extract_csv(file_path)
        if suffix == ".json":
            return self._extract_json(file_path)

        return ""

    def _extract_pdf(self, file_path: Path) -> str:
        text = ""
        try:
            from pypdf import PdfReader  # type: ignore

            # Silence noisy non-fatal parser warnings (e.g., wrong pointing object).
            logging.getLogger("pypdf").setLevel(logging.ERROR)
            reader = PdfReader(str(file_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""

        extracted = clean_extracted_text(text)
        if normalize_whitespace(extracted):
            # If primary extraction appears incomplete on key anchors, attempt OCR and merge.
            anchors = ("revenue", "profit", "debt", "receivables", "cash flow", "gst", "bank")
            lower_text = extracted.lower()
            anchor_hits = sum(1 for anchor in anchors if anchor in lower_text)
            if anchor_hits >= 4:
                return extracted

        # OCR fallback is attempted when text extraction is empty or anchor coverage is weak.
        try:
            import pytesseract  # type: ignore
            from pdf2image import convert_from_path  # type: ignore

            images = convert_from_path(str(file_path))
            ocr_text = "\n".join(pytesseract.image_to_string(image) for image in images)
            ocr_clean = clean_extracted_text(ocr_text)

            if normalize_whitespace(extracted) and normalize_whitespace(ocr_clean):
                return clean_extracted_text(extracted + "\n" + ocr_clean)
            if normalize_whitespace(ocr_clean):
                return ocr_clean
            return extracted
        except Exception:
            return extracted

    def _extract_csv(self, file_path: Path) -> str:
        lines: List[str] = []
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if headers:
                lines.append("headers: " + ", ".join(headers))
            for row in reader:
                row_parts = [f"{key}: {value}" for key, value in row.items()]
                lines.append("; ".join(row_parts))
        return clean_extracted_text("\n".join(lines))

    def _extract_json(self, file_path: Path) -> str:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        flattened_lines: List[str] = []

        def walk(node: object, prefix: str = "") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    next_prefix = f"{prefix}.{key}" if prefix else str(key)
                    walk(value, next_prefix)
                return

            if isinstance(node, list):
                for index, value in enumerate(node):
                    next_prefix = f"{prefix}[{index}]"
                    walk(value, next_prefix)
                return

            flattened_lines.append(f"{prefix}: {node}")

        walk(payload)
        return clean_extracted_text("\n".join(flattened_lines))
