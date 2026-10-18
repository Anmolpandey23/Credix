"""Document extraction with page-level provenance and local OCR."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, List

from utils.text_utils import clean_extracted_text, normalize_whitespace

logger = logging.getLogger(__name__)


class ExtractionService:
    def extract_documents(
        self, file_paths: Iterable[Path]
    ) -> List[dict[str, str | int | None]]:
        """Extract document segments and preserve source locations where known."""
        documents: List[dict[str, str | int | None]] = []

        for file_path in file_paths:
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                try:
                    from pypdf import PdfReader  # type: ignore

                    logging.getLogger("pypdf").setLevel(logging.ERROR)
                    reader = PdfReader(str(file_path))

                    for page_number, page in enumerate(reader.pages, start=1):
                        text = clean_extracted_text(page.extract_text() or "")

                        # OCR only pages where selectable text is unavailable.
                        if not normalize_whitespace(text):
                            text = self._ocr_pdf_page(file_path, page_number)

                        if normalize_whitespace(text):
                            documents.append(
                                {
                                    "source": file_path.name,
                                    "page": page_number,
                                    "text": text,
                                }
                            )

                    # Page extraction succeeded. Do not merge the document
                    # into one segment or guess page numbers.
                    continue

                except Exception as exc:
                    logger.warning(
                        "Page-level PDF extraction failed for %s: %s",
                        file_path.name,
                        exc,
                    )

                # Last-resort combined extraction. Page is explicitly unknown.
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
        parts = [self.extract(path) for path in file_paths]
        return clean_extracted_text(
            "\n".join(part for part in parts if part.strip())
        )

    def extract(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        if suffix == ".csv":
            return self._extract_csv(file_path)
        if suffix == ".json":
            return self._extract_json(file_path)

        return ""

    def _ocr_pdf_page(self, file_path: Path, page_number: int) -> str:
        """OCR one 1-based PDF page using PyMuPDF and local Tesseract."""
        try:
            import pymupdf  # type: ignore

            with pymupdf.open(str(file_path)) as document:
                page_index = page_number - 1

                if page_index < 0 or page_index >= len(document):
                    logger.warning(
                        "Invalid OCR page %s for %s",
                        page_number,
                        file_path.name,
                    )
                    return ""

                page = document[page_index]
                text_page = page.get_textpage_ocr(
                    language="eng",
                    dpi=300,
                    full=True,
                )
                text = page.get_text("text", textpage=text_page)

            return clean_extracted_text(text)

        except Exception as exc:
            logger.warning(
                "OCR unavailable or failed for %s page %s: %s",
                file_path.name,
                page_number,
                exc,
            )
            return ""

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract all PDF pages, OCRing pages without selectable text."""
        try:
            from pypdf import PdfReader  # type: ignore

            logging.getLogger("pypdf").setLevel(logging.ERROR)
            reader = PdfReader(str(file_path))
            page_texts: List[str] = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = clean_extracted_text(page.extract_text() or "")

                if not normalize_whitespace(text):
                    text = self._ocr_pdf_page(file_path, page_number)

                if normalize_whitespace(text):
                    page_texts.append(text)

            return clean_extracted_text("\n".join(page_texts))

        except Exception as exc:
            logger.warning(
                "pypdf extraction failed for %s; trying PyMuPDF: %s",
                file_path.name,
                exc,
            )

        # Fallback when pypdf cannot parse the document.
        try:
            import pymupdf  # type: ignore

            page_texts = []
            with pymupdf.open(str(file_path)) as document:
                for page_number, page in enumerate(document, start=1):
                    text = clean_extracted_text(page.get_text("text") or "")

                    if not normalize_whitespace(text):
                        text_page = page.get_textpage_ocr(
                            language="eng",
                            dpi=300,
                            full=True,
                        )
                        text = clean_extracted_text(
                            page.get_text("text", textpage=text_page)
                        )

                    if normalize_whitespace(text):
                        page_texts.append(text)

            return clean_extracted_text("\n".join(page_texts))

        except Exception as exc:
            logger.warning(
                "PyMuPDF extraction/OCR failed for %s: %s",
                file_path.name,
                exc,
            )
            return ""

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
