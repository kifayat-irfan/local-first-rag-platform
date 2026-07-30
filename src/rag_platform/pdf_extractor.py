"""
PDF text extraction with page numbers, heading detection, and code-block
preservation.

Design decisions (with reasoning):

1. PDF library choice: pypdf is the first pass on every page — it's fast and
   handles the vast majority of well-formed, digitally-authored PDFs fine.
   pdfplumber is the fallback, used only when pypdf's output looks sparse
   (likely a scanned page, an encoding pypdf mishandled, or a complex
   multi-column layout). pdfplumber is slower but exposes per-character
   font metadata (size, fontname), which is what heading and code-block
   detection actually need — pypdf doesn't expose this reliably. So
   pdfplumber does double duty here: text-quality fallback AND the only
   source of structural (heading/code) information.

2. Page number preservation: inline HTML-comment markers
   (`<!-- page:N -->`) embedded directly in the returned markdown, rather
   than page numbers living only in a separate metadata list. Reasoning:
   once text is handed to the chunker, paragraphs get merged and re-split
   — a page-number list keyed by paragraph index doesn't survive that.
   An inline marker does: the chunker can regex-scan for the last marker
   before a chunk's start offset and attach that page number to the
   chunk's metadata directly, with no separate alignment step.

3. Scanned-PDF detection: if, after both extractors, a page still has
   fewer than MIN_CHARS_PER_PAGE characters, it's flagged as likely
   scanned/image-only. If more than half a document's pages are flagged,
   the whole document is marked `is_scanned=True`. Per the spec, this
   platform does NOT run OCR — callers (ingest_company_pdfs.py) are
   expected to skip scanned documents and log a clear warning rather than
   silently indexing near-empty text.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

MIN_CHARS_PER_PAGE = 20
HEADING_SIZE_RATIO = 1.15          # a line's font must be >=15% larger than body text to count as a heading
HEADING_MAX_CHARS = 120            # headings are short; long "big text" lines are probably not headings
MONOSPACE_HINTS = ("mono", "courier", "consolas", "menlo", "code")
PDFPLUMBER_FONT_SAMPLE_PAGES = 20  # cap how many pages we sample to establish the body font size, for perf on 100+ page docs

PAGE_MARKER = "<!-- page:{n} -->"


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened at all (corrupt, encrypted, zero pages)."""


class PageContent(BaseModel):
    page_number: int  # 1-indexed
    text: str
    char_count: int
    is_likely_scanned: bool = False


class ExtractedDocument(BaseModel):
    source_path: str
    markdown: str
    pages: list[PageContent]
    extractor_used: str  # "pypdf" | "pypdf+pdfplumber"
    is_scanned: bool
    page_count: int
    metadata: dict = {}


def extract_pdf(path: Path) -> ExtractedDocument:
    """
    Extract text, headings, code blocks, and page numbers from a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        An ExtractedDocument. If the document looks scanned, `is_scanned`
        is True and `markdown` will be sparse/near-empty — callers should
        check this flag and skip indexing rather than embedding near-empty
        chunks.

    Raises:
        PDFExtractionError: if the file doesn't exist or can't be opened
            by either backend at all (corrupt file, unsupported encryption).

    Example:
        >>> doc = extract_pdf(Path("handbook.pdf"))
        >>> if doc.is_scanned:
        ...     print("skipping, looks scanned")
        >>> doc.page_count
        12
    """
    path = Path(path)
    if not path.exists():
        raise PDFExtractionError(f"PDF not found: {path}")

    pypdf_pages = _extract_with_pypdf(path)  # may raise PDFExtractionError

    sparse_page_indices = [i for i, text in enumerate(pypdf_pages) if len(text.strip()) < MIN_CHARS_PER_PAGE]
    needs_structure_pass = True  # we always want heading/code detection when possible
    extractor_used = "pypdf"
    plumber_pages: Optional[list[str]] = None

    if sparse_page_indices or needs_structure_pass:
        try:
            plumber_pages = _extract_with_pdfplumber(path)
            extractor_used = "pypdf+pdfplumber"
        except Exception:
            # pdfplumber failing is not fatal — we degrade to plain pypdf
            # text with no heading/code structure, rather than losing the document.
            plumber_pages = None

    final_pages: list[str] = []
    for i, pypdf_text in enumerate(pypdf_pages):
        if plumber_pages is not None and i < len(plumber_pages):
            plumber_text = plumber_pages[i]
            # Prefer whichever extractor actually found more content on this page.
            final_pages.append(plumber_text if len(plumber_text.strip()) >= len(pypdf_text.strip()) else pypdf_text)
        else:
            final_pages.append(pypdf_text)

    pages: list[PageContent] = []
    markdown_parts: list[str] = []
    scanned_count = 0
    for idx, text in enumerate(final_pages, start=1):
        char_count = len(text.strip())
        is_scanned_page = char_count < MIN_CHARS_PER_PAGE
        if is_scanned_page:
            scanned_count += 1
        pages.append(PageContent(page_number=idx, text=text, char_count=char_count, is_likely_scanned=is_scanned_page))
        markdown_parts.append(PAGE_MARKER.format(n=idx))
        markdown_parts.append(text)

    doc_is_scanned = len(pages) > 0 and (scanned_count / len(pages)) > 0.5

    return ExtractedDocument(
        source_path=str(path),
        markdown="\n\n".join(p for p in markdown_parts if p),
        pages=pages,
        extractor_used=extractor_used,
        is_scanned=doc_is_scanned,
        page_count=len(pages),
        metadata={"scanned_page_count": scanned_count},
    )


# ---------------------------------------------------------------------- #
# pypdf backend
# ---------------------------------------------------------------------- #
def _extract_with_pypdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover
        raise PDFExtractionError("pypdf is not installed") from exc

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            # Try an empty password — some PDFs are "encrypted" with owner
            # restrictions only (no user password), which pypdf can bypass this way.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise PDFExtractionError(f"PDF is encrypted and could not be decrypted: {path}") from exc
    except PdfReadError as exc:
        raise PDFExtractionError(f"pypdf could not open {path}: {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # A single malformed page must not take down the whole document.
            pages.append("")

    if not pages:
        raise PDFExtractionError(f"PDF has zero pages: {path}")

    return pages


# ---------------------------------------------------------------------- #
# pdfplumber backend — text quality fallback + heading/code structure
# ---------------------------------------------------------------------- #
def _extract_with_pdfplumber(path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        body_size = _estimate_body_font_size(pdf.pages[:PDFPLUMBER_FONT_SAMPLE_PAGES])
        heading_threshold = body_size * HEADING_SIZE_RATIO

        pages_out: list[str] = []
        for page in pdf.pages:
            pages_out.append(_page_to_markdown(page, heading_threshold))
        return pages_out


def _estimate_body_font_size(sample_pages) -> float:
    sizes: list[float] = []
    for page in sample_pages:
        for ch in page.chars:
            size = ch.get("size")
            if size:
                sizes.append(round(size, 1))
    return statistics.median(sizes) if sizes else 10.0


def _page_to_markdown(page, heading_threshold: float) -> str:
    lines = _group_chars_into_lines(page.chars)

    md_lines: list[str] = []
    in_code_block = False
    for line_chars in lines:
        text = "".join(c["text"] for c in line_chars).strip()
        if not text:
            continue

        avg_size = statistics.mean(c["size"] for c in line_chars if c.get("size"))
        is_heading = avg_size >= heading_threshold and len(text) < HEADING_MAX_CHARS
        is_code = _is_monospace_line(line_chars) and not is_heading

        if is_code:
            if not in_code_block:
                md_lines.append("```")
                in_code_block = True
            md_lines.append(text)
            continue

        if in_code_block:
            md_lines.append("```")
            in_code_block = False

        md_lines.append(f"## {text}" if is_heading else text)

    if in_code_block:
        md_lines.append("```")

    return "\n".join(md_lines)


def _group_chars_into_lines(chars) -> list[list[dict]]:
    """Group pdfplumber chars into visual lines by their vertical ('top') position."""
    lines: dict[float, list[dict]] = {}
    for c in chars:
        key = round(c["top"], 1)
        lines.setdefault(key, []).append(c)
    return [sorted(lines[k], key=lambda c: c["x0"]) for k in sorted(lines.keys())]


def _is_monospace_line(line_chars: list[dict]) -> bool:
    if not line_chars:
        return False
    mono_count = sum(1 for c in line_chars if any(h in c.get("fontname", "").lower() for h in MONOSPACE_HINTS))
    return (mono_count / len(line_chars)) > 0.7
