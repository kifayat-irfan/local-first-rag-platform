#!/usr/bin/env python3
"""
Manage registered companies.

Usage:
    python scripts/manage_companies.py list
    python scripts/manage_companies.py add company_a --name "Acme Corp"
    python scripts/manage_companies.py info company_a
    python scripts/manage_companies.py remove company_a [--delete-data]
    python scripts/manage_companies.py update company_a
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_platform.company_registry import CompanyAlreadyExistsError, CompanyNotFoundError, CompanyRegistry
from rag_platform.config import get_settings
from rag_platform.manifest import ManifestStore


def cmd_list(registry: CompanyRegistry) -> int:
    companies = registry.list_companies()
    if not companies:
        print("No companies registered yet. Use 'add' to register one.")
        return 0
    print(f"{'ID':<20} {'Name':<25} {'PDFs':>6} {'Chunks':>8}  Last ingestion")
    print("-" * 90)
    for c in companies:
        print(f"{c.company_id:<20} {c.name:<25} {c.total_pdfs:>6} {c.total_chunks:>8}  {c.last_ingestion or 'never'}")
    return 0


def cmd_add(registry: CompanyRegistry, company_id: str, name: str) -> int:
    settings = get_settings(company_id)
    settings.ensure_dirs()
    try:
        record = registry.add(company_id, name, str(settings.config_path), str(settings.data_dir))
    except CompanyAlreadyExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not settings.config_path.exists():
        print(f"Note: {settings.config_path} does not exist yet.")
        print(f"      Copy configs/template.yaml -> {settings.config_path.name} and set company.id: {company_id}")

    print(f"Registered {company_id!r} ({name}).")
    print(f"  Data dir:    {record.data_path}")
    print(f"  Config path: {record.config_path}")
    print(f"  Raw PDFs go: {settings.raw_pdfs_dir}")
    return 0


def cmd_info(registry: CompanyRegistry, company_id: str) -> int:
    try:
        record = registry.get(company_id)
    except CompanyNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    settings = get_settings(company_id)
    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    ok = sum(1 for e in manifest.all_entries() if e.status == "ok")
    scanned = sum(1 for e in manifest.all_entries() if e.status == "skipped_scanned")
    errored = sum(1 for e in manifest.all_entries() if e.status == "error")

    print(f"Company: {record.company_id}")
    print(f"  Name:            {record.name}")
    print(f"  Created:         {record.created_at}")
    print(f"  Last ingestion:  {record.last_ingestion or 'never'}")
    print(f"  Registered PDFs: {record.total_pdfs}  Chunks: {record.total_chunks}")
    print(f"  Manifest:        {ok} ok, {scanned} scanned-skipped, {errored} errored ({len(manifest)} total entries)")
    print(f"  Data dir:        {settings.data_dir}")
    print(f"  Vector store:    {settings.vector_db_dir}")
    return 0


def cmd_remove(registry: CompanyRegistry, company_id: str, delete_data: bool) -> int:
    try:
        registry.remove(company_id)
    except CompanyNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Unregistered {company_id!r}.")
    if delete_data:
        settings = get_settings(company_id)
        if settings.data_dir.exists():
            shutil.rmtree(settings.data_dir)
            print(f"Deleted data directory: {settings.data_dir}")
    else:
        print("Data directory left in place (pass --delete-data to remove it too).")
    return 0


def cmd_update(registry: CompanyRegistry, company_id: str) -> int:
    settings = get_settings(company_id)
    manifest = ManifestStore(settings.manifest_path)
    manifest.load()
    total_chunks = sum(e.chunk_count or 0 for e in manifest.all_entries())
    try:
        registry.update_stats(company_id, total_pdfs=len(manifest.all_entries()), total_chunks=total_chunks)
    except CompanyNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Refreshed stats for {company_id!r}: {len(manifest.all_entries())} PDFs, {total_chunks} chunks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list")

    p_add = sub.add_parser("add")
    p_add.add_argument("company_id")
    p_add.add_argument("--name", required=True)

    p_info = sub.add_parser("info")
    p_info.add_argument("company_id")

    p_remove = sub.add_parser("remove")
    p_remove.add_argument("company_id")
    p_remove.add_argument("--delete-data", action="store_true")

    p_update = sub.add_parser("update")
    p_update.add_argument("company_id")

    args = parser.parse_args()

    settings = get_settings()  # registry path is global, "default" tenant is fine here
    registry = CompanyRegistry(settings.registry_path)

    if args.command == "list":
        return cmd_list(registry)
    if args.command == "add":
        return cmd_add(registry, args.company_id, args.name)
    if args.command == "info":
        return cmd_info(registry, args.company_id)
    if args.command == "remove":
        return cmd_remove(registry, args.company_id, args.delete_data)
    if args.command == "update":
        return cmd_update(registry, args.company_id)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
