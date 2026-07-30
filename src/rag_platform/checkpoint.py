"""
Checkpoint: tracks progress of an ingestion *run* for one company, separate
from the manifest (which tracks per-file trustworthiness). Lets
`ingest_company_pdfs.py --company X` be killed mid-run and resumed by
re-running the identical command — completed files are skipped via the
manifest's local-first check anyway, but the checkpoint additionally
remembers which files failed, so a summary can distinguish "not yet
attempted" from "tried and failed."
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    company_id: str
    pending: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    skipped_scanned: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def load(cls, path: Path, company_id: str) -> "Checkpoint":
        path = Path(path)
        if not path.exists():
            return cls(company_id=company_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            return cls(company_id=company_id)

    def save(self, path: Path) -> None:
        """Atomic write: write to .tmp then rename, so a crash mid-write never corrupts the checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def seed(self, paths: list[str]) -> None:
        seen = set(self.completed) | set(self.pending) | set(self.skipped_scanned)
        for p in paths:
            if p not in seen:
                self.pending.append(p)
                seen.add(p)

    def mark_completed(self, path: str) -> None:
        self._remove_from_pending(path)
        if path not in self.completed:
            self.completed.append(path)

    def mark_failed(self, path: str) -> None:
        self._remove_from_pending(path)
        if path not in self.failed:
            self.failed.append(path)

    def mark_skipped_scanned(self, path: str) -> None:
        self._remove_from_pending(path)
        if path not in self.skipped_scanned:
            self.skipped_scanned.append(path)

    def _remove_from_pending(self, path: str) -> None:
        if path in self.pending:
            self.pending.remove(path)

    def is_complete(self) -> bool:
        return len(self.pending) == 0
