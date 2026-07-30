"""
Loads and validates a company's YAML config, with inheritance from
configs/template.yaml.

Design decision — template inheritance via deep merge: a company YAML file
only needs to specify what differs from the template (usually just
`company.id`, `company.name`, and `sources[].path`). Everything else
(chunking, embedding model, retrieval params) falls back to the template's
defaults. This keeps company configs small and means a platform-wide
default change (e.g. bumping chunk_size) is a one-line edit to
template.yaml instead of N edits across every company file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class ConfigValidationError(Exception):
    """Raised with a human-readable message when a company config fails validation."""


class SourceConfig(BaseModel):
    type: str = "pdf"
    path: str
    glob: str = "*.pdf"


class CompanyInfo(BaseModel):
    id: str
    name: str
    description: str = ""


class ProcessingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 100
    preserve_code_blocks: bool = True


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    type: str = "chromadb"
    path: str
    collection_name: str
    distance_metric: str = "cosine"


class RetrievalConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 5
    use_bm25: bool = True
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CompanyConfig(BaseModel):
    company: CompanyInfo
    sources: list[SourceConfig] = Field(default_factory=list)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`; dicts merge key-by-key, everything else is replaced outright."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigValidationError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigValidationError(f"{path} must contain a YAML mapping at the top level, got {type(data).__name__}")
    return data


def load_company_config(
    company_id: str, configs_dir: Path, template_name: str = "template.yaml"
) -> CompanyConfig:
    """
    Load configs/{company_id}.yaml, deep-merged onto configs/template.yaml, and validate it.

    Args:
        company_id: Which company's config to load.
        configs_dir: Directory containing template.yaml and {company_id}.yaml.
        template_name: Template filename, overridable for tests.

    Returns:
        A validated CompanyConfig.

    Raises:
        ConfigValidationError: with a human-readable explanation — missing
            file, invalid YAML syntax, or a field that failed pydantic
            validation (required field missing, wrong type, etc).

    Example:
        >>> cfg = load_company_config("acme_corp", Path("configs"))
        >>> cfg.processing.chunk_size
        512
    """
    configs_dir = Path(configs_dir)
    company_path = configs_dir / f"{company_id}.yaml"
    template_path = configs_dir / template_name

    template_data = load_yaml(template_path) if template_path.exists() else {}
    company_data = load_yaml(company_path)

    merged = _deep_merge(template_data, company_data)

    # The template ships with placeholder values like "company_xxx" meant to
    # be substituted — if a company file forgets to override `company.id`,
    # fail loudly here rather than silently indexing under the wrong id.
    declared_id = merged.get("company", {}).get("id")
    if declared_id and declared_id != company_id and not declared_id.endswith("_xxx"):
        raise ConfigValidationError(
            f"{company_path} declares company.id={declared_id!r} but was loaded as "
            f"company_id={company_id!r} — these must match."
        )
    merged.setdefault("company", {})["id"] = company_id

    try:
        return CompanyConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigValidationError(f"Invalid config for company {company_id!r} ({company_path}):\n{exc}") from exc
