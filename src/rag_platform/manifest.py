"""
Append-only JSONL manifest — the source of truth for "has this PDF already
been processed, and is the result still trustworthy?"

Local-first rule for PDF ingestion: skip re-processing a PDF only if the
manifest has an entry for it AND the file's current SHA-256 matches what's
recorded AND the corresponding clean/{sha256}.md file still exists. Any one
of those failing means re-extract, re-chunk, re-embed.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class ManifestStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    SKIPPED_SCANNED = "skipped_scanned"  # detected as a scanned/image-only PDF, not indexed


class ManifestEntry(BaseModel):
    source_path: str
    file_sha256: str
    clean_path: Optional[str] = None
    company_id: str
    status: ManifestStatus = ManifestStatus.OK
    error: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    extractor_used: Optional[str] = None
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_jsonl(self) -> str:
        return self.model_dump_json()


class ManifestStore:
    """
    In-memory index over a per-company append-only manifest.jsonl.

    Args:
        manifest_path: Path to this tenant's manifest.jsonl (from Settings.manifest_path).
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self._index: dict[str, ManifestEntry] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    def load(self) -> None:
        """Replay the JSONL file; last entry per source_path wins. Corrupt lines are skipped, not fatal."""
        self._index.clear()
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = ManifestEntry.model_validate_json(line)
                    except Exception:
                        continue
                    self._index[entry.source_path] = entry
        self._loaded = True

    async def ensure_loaded(self) -> None:
        if not self._loaded:
            async with self._lock:
                if not self._loaded:
                    self.load()

    def get(self, source_path: str) -> Optional[ManifestEntry]:
        return self._index.get(source_path)

    def __len__(self) -> int:
        return len(self._index)

    def all_entries(self) -> list[ManifestEntry]:
        return list(self._index.values())

    def needs_processing(self, pdf_path: Path) -> bool:
        """
        Local-first check: True if this PDF must be (re-)processed.

        Args:
            pdf_path: Path to the source PDF on disk.

        Returns:
            False only if all three local-first conditions hold: a manifest
            entry exists, its recorded hash matches the file's current hash,
            and the clean markdown file it points to still exists on disk.
        """
        entry = self.get(str(pdf_path))
        if entry is None or entry.status != ManifestStatus.OK:
            return True
        if not pdf_path.exists():
            return True
        if sha256_file(pdf_path) != entry.file_sha256:
            return True
        if entry.clean_path and not Path(entry.clean_path).exists():
            return True
        return False

    async def add(self, entry: ManifestEntry) -> None:
        async with self._lock:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "a", encoding="utf-8") as fh:
                fh.write(entry.to_jsonl() + "\n")
                fh.flush()
            self._index[entry.source_path] = entry

    async def compact(self) -> None:
        """Rewrite manifest.jsonl keeping only the latest entry per source_path."""
        async with self._lock:
            tmp_path = self.manifest_path.with_suffix(".jsonl.tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                for entry in self._index.values():
                    fh.write(entry.to_jsonl() + "\n")
            tmp_path.replace(self.manifest_path)
