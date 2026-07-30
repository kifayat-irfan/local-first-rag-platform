from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_platform.config import Settings
from rag_platform.config_loader import ConfigValidationError, load_company_config


TEMPLATE_YAML = """
company:
  id: company_xxx
  name: "Company Name"
sources:
  - type: pdf
    path: ./data/companies/company_xxx/raw_pdfs/
    glob: "*.pdf"
processing:
  chunk_size: 512
  chunk_overlap: 50
  min_chunk_length: 100
embedding:
  model: BAAI/bge-small-en-v1.5
  batch_size: 32
vector_store:
  type: chromadb
  path: ./data/companies/company_xxx/vector_store/
  collection_name: company_xxx_docs
retrieval:
  top_k: 10
  rerank_top_k: 5
"""


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    (d / "template.yaml").write_text(TEMPLATE_YAML)
    return d


def test_company_config_inherits_template_defaults(configs_dir: Path):
    company_yaml = {
        "company": {"id": "acme", "name": "Acme Corp"},
        "vector_store": {"type": "chromadb", "path": "x", "collection_name": "acme_docs"},
    }
    (configs_dir / "acme.yaml").write_text(yaml.dump(company_yaml))

    cfg = load_company_config("acme", configs_dir)

    assert cfg.company.name == "Acme Corp"
    assert cfg.processing.chunk_size == 512  # inherited from template
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"  # inherited


def test_company_config_overrides_template(configs_dir: Path):
    company_yaml = {
        "company": {"id": "acme", "name": "Acme Corp"},
        "processing": {"chunk_size": 400, "chunk_overlap": 60},
        "vector_store": {"type": "chromadb", "path": "x", "collection_name": "acme_docs"},
    }
    (configs_dir / "acme.yaml").write_text(yaml.dump(company_yaml))

    cfg = load_company_config("acme", configs_dir)

    assert cfg.processing.chunk_size == 400  # overridden
    assert cfg.processing.min_chunk_length == 100  # still inherited (not overridden)


def test_missing_company_file_raises(configs_dir: Path):
    with pytest.raises(ConfigValidationError):
        load_company_config("nonexistent", configs_dir)


def test_invalid_yaml_raises(configs_dir: Path):
    (configs_dir / "broken.yaml").write_text("company:\n  id: [unclosed")
    with pytest.raises(ConfigValidationError):
        load_company_config("broken", configs_dir)


def test_missing_required_field_raises(tmp_path: Path):
    # No template.yaml at all here, and the company file omits vector_store
    # (which has no default) — this must fail validation, not silently pass.
    configs_dir_no_template = tmp_path / "configs_no_template"
    configs_dir_no_template.mkdir()
    (configs_dir_no_template / "incomplete.yaml").write_text(
        yaml.dump({"company": {"id": "incomplete", "name": "X"}})
    )
    with pytest.raises(ConfigValidationError):
        load_company_config("incomplete", configs_dir_no_template)


def test_id_mismatch_raises(configs_dir: Path):
    company_yaml = {
        "company": {"id": "wrong_id", "name": "X"},
        "vector_store": {"type": "chromadb", "path": "x", "collection_name": "x"},
    }
    (configs_dir / "mismatched.yaml").write_text(yaml.dump(company_yaml))

    with pytest.raises(ConfigValidationError):
        load_company_config("mismatched", configs_dir)


def test_settings_company_id_validator_rejects_bad_chars():
    with pytest.raises(ValueError):
        Settings(company_id="Not Valid!")


def test_settings_default_tenant_vs_company_tenant_isolated(tmp_path: Path):
    default = Settings(company_id="default", project_root=tmp_path)
    acme = Settings(company_id="acme_corp", project_root=tmp_path)

    assert default.data_dir != acme.data_dir
    assert default.vector_collection_name != acme.vector_collection_name
    assert default.registry_path == acme.registry_path  # registry is global
