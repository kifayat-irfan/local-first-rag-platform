#!/usr/bin/env python3
"""
Audit a company's manifest against what's actually on disk.

Usage:
    python scripts/check_manifest.py --company company_a

Reports OK / MISSING_CLEAN_FILE / HASH_MISMATCH / ERROR_ENTRY / SCANNED per
PDF. Exits non-zero if anything but SCANNED needs attention — CI-friendly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.config import get_settings
from rag_platform.manifest import ManifestStatus, ManifestStore, sha256_file


def main(company_id: str) -> int:
    settings = get_settings(company_id)
    manifest = ManifestStore(settings.manifest_path)
    manifest.load()

    if len(manifest) == 0:
        print(f"Manifest is empty or missing at {settings.manifest_path}")
        return 0

    counts = {"OK": 0, "MISSING_SOURCE": 0, "MISSING_CLEAN_FILE": 0, "HASH_MISMATCH": 0, "ERROR_ENTRY": 0, "SCANNED": 0}

    for entry in manifest.all_entries():
        if entry.status == ManifestStatus.ERROR:
            counts["ERROR_ENTRY"] += 1
            print(f"[ERROR_ENTRY]         {entry.source_path}  ({entry.error})")
            continue
        if entry.status == ManifestStatus.SKIPPED_SCANNED:
            counts["SCANNED"] += 1
            print(f"[SCANNED]             {entry.source_path}")
            continue

        source_path = Path(entry.source_path)
        if not source_path.exists():
            counts["MISSING_SOURCE"] += 1
            print(f"[MISSING_SOURCE]      {entry.source_path}")
            continue

        if sha256_file(source_path) != entry.file_sha256:
            counts["HASH_MISMATCH"] += 1
            print(f"[HASH_MISMATCH]       {entry.source_path}  (will be re-processed on next ingest)")
            continue

        if entry.clean_path and not Path(entry.clean_path).exists():
            counts["MISSING_CLEAN_FILE"] += 1
            print(f"[MISSING_CLEAN_FILE]  {entry.source_path}  (expected at {entry.clean_path})")
            continue

        counts["OK"] += 1

    print("\n--- Summary ---")
    for k, v in counts.items():
        print(f"{k:20s} {v}")

    failures = counts["MISSING_SOURCE"] + counts["MISSING_CLEAN_FILE"] + counts["HASH_MISMATCH"] + counts["ERROR_ENTRY"]
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.company))
