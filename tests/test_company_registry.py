from __future__ import annotations

import pytest

from rag_platform.company_registry import (
    CompanyAlreadyExistsError,
    CompanyNotFoundError,
    CompanyRegistry,
)


def test_add_and_list(registry: CompanyRegistry):
    registry.add("acme", "Acme Corp", "configs/acme.yaml", "data/companies/acme")
    registry.add("globex", "Globex Inc", "configs/globex.yaml", "data/companies/globex")

    ids = {c.company_id for c in registry.list_companies()}
    assert ids == {"acme", "globex"}


def test_duplicate_add_raises(registry: CompanyRegistry):
    registry.add("acme", "Acme Corp", "configs/acme.yaml", "data/companies/acme")
    with pytest.raises(CompanyAlreadyExistsError):
        registry.add("acme", "Dup", "x", "y")


def test_get_missing_raises(registry: CompanyRegistry):
    with pytest.raises(CompanyNotFoundError):
        registry.get("does_not_exist")


def test_update_stats(registry: CompanyRegistry):
    registry.add("acme", "Acme Corp", "configs/acme.yaml", "data/companies/acme")
    record = registry.update_stats("acme", total_pdfs=10, total_chunks=250, mark_ingestion_now=True)

    assert record.total_pdfs == 10
    assert record.total_chunks == 250
    assert record.last_ingestion is not None


def test_remove(registry: CompanyRegistry):
    registry.add("acme", "Acme Corp", "configs/acme.yaml", "data/companies/acme")
    registry.remove("acme")
    assert registry.exists("acme") is False

    with pytest.raises(CompanyNotFoundError):
        registry.remove("acme")


def test_registry_survives_reload(registry: CompanyRegistry, tmp_settings):
    registry.add("acme", "Acme Corp", "configs/acme.yaml", "data/companies/acme")

    fresh = CompanyRegistry(tmp_settings.registry_path)
    record = fresh.get("acme")
    assert record.name == "Acme Corp"
