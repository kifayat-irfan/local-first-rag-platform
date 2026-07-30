#!/usr/bin/env python3
"""
Wipe a company's local ingestion state (clean files, manifest, checkpoint,
vector store). Raw PDFs are never touched — only derived state.

Usage:
    python scripts/reset_data.py --company company_a
    python scripts/reset_data.py --company company_a --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.config import get_settings


def main(company_id: str, skip_confirm: bool) -> int:
    settings = get_settings(company_id)

    targets = [settings.clean_dir, settings.manifest_path, settings.checkpoint_path, settings.vector_db_dir]

    print(f"This will delete derived ingestion state for {company_id!r} (raw_pdfs/ is NOT touched):")
    for t in targets:
        print(f"  - {t}")

    if not skip_confirm:
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return 1

    for t in targets:
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
        elif t.exists():
            t.unlink()

    settings.ensure_dirs()
    print("Reset complete.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.company, args.yes))
