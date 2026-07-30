from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from rag_platform.company_registry import CompanyRegistry
from rag_platform.config import Settings


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """A Settings instance fully isolated under tmp_path, for a throwaway 'test_co' tenant."""
    settings = Settings(company_id="test_co", project_root=tmp_path)
    settings.ensure_dirs()
    return settings


@pytest.fixture
def registry(tmp_settings: Settings) -> CompanyRegistry:
    return CompanyRegistry(tmp_settings.registry_path)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A 2-page synthetic PDF: heading + body + code block on page 1, heading + body on page 2."""
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 720, "Deployment Guide")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "This document explains how to deploy the service to production.")
    c.drawString(72, 675, "Follow the steps below carefully before releasing.")
    c.setFont("Courier", 10)
    c.drawString(72, 640, "def deploy():")
    c.drawString(90, 625, 'run("make build")')
    c.drawString(90, 610, 'run("make push")')
    c.showPage()

    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "Rollback Procedure")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "If deployment fails, run the rollback script immediately.")
    c.drawString(72, 675, "Contact the on-call engineer if issues persist.")
    c.showPage()

    c.save()
    return pdf_path


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF with no extractable text at all — simulates a scanned/image-only document."""
    pdf_path = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.showPage()
    c.showPage()
    c.save()
    return pdf_path
