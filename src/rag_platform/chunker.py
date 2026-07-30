"""
Recursive character chunker for technical documentation.

Two behaviors beyond a plain text splitter, both needed for PDF-sourced
technical docs specifically:

1. Code-block awareness: a fenced ```code block``` from pdf_extractor.py
   is never split across a chunk boundary if it fits within chunk_size.
   Splitting a code sample mid-line makes it useless for both a human
   reader and an LLM trying to answer questions about it, so code blocks
   are treated as atomic units and only force-split if they alone exceed
   chunk_size.

2. Page-number carrying: pdf_extractor.py embeds `<!-- page:N -->` markers
   inline. This chunker strips them from the visible chunk text but
   records the page number of the *last marker seen before each chunk's
   start* as chunk metadata — so every chunk can cite "page 7" without a
   separate alignment pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

PAGE_MARKER_RE = re.compile(r"<!--\s*page:(\d+)\s*-->")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    text: str
    index: int
    page_number: Optional[int] = None
    is_code: bool = False
    metadata: dict = field(default_factory=dict)


class RecursiveCharacterChunker:
    """
    Args:
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap carried between consecutive chunks.
        min_chunk_length: Chunks shorter than this (after stripping) are dropped —
            filters out near-empty fragments left over from splitting (e.g. a
            lone page marker or a stray heading with no body).
        separators: Ordered from most to least semantic; see _split_text.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_length: int = 0,
        separators: Optional[list[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.separators = separators or DEFAULT_SEPARATORS

    def split(self, markdown: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """
        Args:
            markdown: Text to chunk, possibly containing `<!-- page:N -->`
                markers and fenced code blocks (as produced by pdf_extractor.py).
            metadata: Extra key/value pairs copied onto every resulting chunk
                (e.g. source filename, sha256).

        Returns:
            List of Chunk, each carrying its own page_number where determinable.
        """
        markdown = markdown.strip()
        if not markdown:
            return []

        page_positions = self._locate_page_markers(markdown)
        text_no_markers = PAGE_MARKER_RE.sub("", markdown)

        segments = self._split_preserving_code_blocks(text_no_markers)
        merged = self._merge_with_overlap(segments)

        chunks: list[Chunk] = []
        cursor = 0
        for piece in merged:
            is_code = piece.strip().startswith("```") and piece.strip().endswith("```")
            # Use the offset of the piece's first non-whitespace character, not
            # the raw piece boundary. A page marker like <!-- page:2 --> leaves
            # behind a few characters of blank-line padding when stripped, and
            # a chunk boundary can land inside that padding — a couple chars
            # "before" the marker's own offset even though the chunk's real
            # content is entirely on the later page. Skipping leading
            # whitespace before the lookup avoids attributing that chunk to
            # the earlier page by a false margin of just a few characters.
            leading_ws = len(piece) - len(piece.lstrip())
            page_number = self._page_for_offset(cursor + leading_ws, page_positions)
            stripped = piece.strip()
            if len(stripped) < self.min_chunk_length:
                cursor += len(piece)
                continue
            chunks.append(
                Chunk(
                    text=stripped,
                    index=len(chunks),
                    page_number=page_number,
                    is_code=is_code,
                    metadata=dict(metadata or {}),
                )
            )
            cursor += len(piece)

        return chunks

    # ------------------------------------------------------------------ #
    # Page number tracking
    # ------------------------------------------------------------------ #
    def _locate_page_markers(self, markdown: str) -> list[tuple[int, int]]:
        """Returns [(char_offset_in_marker_stripped_text, page_number), ...] in order."""
        positions: list[tuple[int, int]] = []
        stripped_offset = 0
        last_end = 0
        for m in PAGE_MARKER_RE.finditer(markdown):
            stripped_offset += m.start() - last_end
            positions.append((stripped_offset, int(m.group(1))))
            last_end = m.end()
        return positions

    def _page_for_offset(self, offset: int, page_positions: list[tuple[int, int]]) -> Optional[int]:
        page = None
        for marker_offset, page_number in page_positions:
            if marker_offset <= offset:
                page = page_number
            else:
                break
        return page

    # ------------------------------------------------------------------ #
    # Code-block-aware splitting
    # ------------------------------------------------------------------ #
    def _split_preserving_code_blocks(self, text: str) -> list[str]:
        """
        Split `text` into pieces, treating each fenced code block as one
        atomic unit (further split internally only if it alone exceeds
        chunk_size) and recursively splitting everything else normally.
        """
        pieces: list[str] = []
        cursor = 0
        for m in CODE_BLOCK_RE.finditer(text):
            before = text[cursor : m.start()]
            if before.strip():
                pieces.extend(self._split_text(before, self.separators))

            code_block = m.group(0)
            if len(code_block) <= self.chunk_size:
                pieces.append(code_block)
            else:
                pieces.extend(self._split_long_code_block(code_block))

            cursor = m.end()

        remainder = text[cursor:]
        if remainder.strip():
            pieces.extend(self._split_text(remainder, self.separators))

        return pieces

    def _split_long_code_block(self, code_block: str) -> list[str]:
        inner = code_block.strip("`\n")
        lines = inner.split("\n")
        parts: list[str] = []
        buffer = "```\n"
        for line in lines:
            candidate = buffer + line + "\n"
            if len(candidate) + 4 > self.chunk_size and buffer != "```\n":
                parts.append(buffer + "```")
                buffer = "```\n" + line + "\n"
            else:
                buffer = candidate
        if buffer != "```\n":
            parts.append(buffer + "```")
        return parts

    # ------------------------------------------------------------------ #
    # Plain-text recursive splitting (paragraph -> sentence -> word -> char)
    # ------------------------------------------------------------------ #
    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        rejoined = [p + sep for p in parts[:-1]] + parts[-1:]

        results: list[str] = []
        buffer = ""
        for part in rejoined:
            candidate = buffer + part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    results.append(buffer)
                if len(part) > self.chunk_size:
                    results.extend(self._split_text(part, rest))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            results.append(buffer)

        return [r for r in results if r.strip()]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Note the code-block branch flushes `current` immediately and pushes
        the code block directly to `merged`, then resets current="" — it
        never sits in `current` waiting to combine with the next piece.
        That's deliberate: if a code block were left in `current`, the next
        overflow's `overlap_text = current[-self.chunk_overlap:]` would
        slice a trailing fragment (backticks included) out of the code
        block and leak it into the following chunk. Flushing immediately
        means overlap_text is only ever computed from plain-text `current`.
        """
        if not pieces:
            return []

        merged: list[str] = []
        current = ""
        for piece in pieces:
            is_atomic_code = piece.strip().startswith("```") and piece.strip().endswith("```")

            if is_atomic_code:
                if current:
                    merged.append(current)
                merged.append(piece)
                current = ""
                continue

            candidate = (current + piece) if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                overlap_text = current[-self.chunk_overlap :] if current else ""
                current = (overlap_text + piece) if overlap_text else piece
                if len(current) > self.chunk_size * 1.5:
                    merged.append(current[: self.chunk_size])
                    current = current[self.chunk_size - self.chunk_overlap :]
        if current:
            merged.append(current)

        return merged
