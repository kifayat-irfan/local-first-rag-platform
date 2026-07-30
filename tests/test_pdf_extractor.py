from __future__ import annotations

from pathlib import Path

import pytest

from rag_platform.pdf_extractor import PDFExtractionError, extract_pdf


def test_extracts_two_pages(sample_pdf: Path):
    doc = extract_pdf(sample_pdf)
    assert doc.page_count == 2
    assert len(doc.pages) == 2
    assert doc.is_scanned is False


def test_heading_detected_and_marked_with_markdown_hash(sample_pdf: Path):
    doc = extract_pdf(sample_pdf)
    assert "## Deployment Guide" in doc.markdown
    assert "## Rollback Procedure" in doc.markdown


def test_code_block_detected_and_fenced(sample_pdf: Path):
    doc = extract_pdf(sample_pdf)
    assert "```" in doc.markdown
    assert "def deploy():" in doc.markdown
    # Code fences must be balanced.
    assert doc.markdown.count("```") % 2 == 0


def test_page_markers_present_in_order(sample_pdf: Path):
    doc = extract_pdf(sample_pdf)
    p1_idx = doc.markdown.index("<!-- page:1 -->")
    p2_idx = doc.markdown.index("<!-- page:2 -->")
    assert p1_idx < p2_idx


def test_scanned_pdf_is_flagged(blank_pdf: Path):
    doc = extract_pdf(blank_pdf)
    assert doc.is_scanned is True
    assert doc.metadata["scanned_page_count"] == doc.page_count


def test_missing_file_raises():
    with pytest.raises(PDFExtractionError):
        extract_pdf(Path("/nonexistent/path/to/nothing.pdf"))


def test_body_text_preserved(sample_pdf: Path):
    doc = extract_pdf(sample_pdf)
    assert "deploy the service to production" in doc.markdown
    assert "rollback script immediately" in doc.markdown
