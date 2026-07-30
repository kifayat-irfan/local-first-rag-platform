"""
Structured JSON logging for the RAG platform.

Every log line in a multi-tenant system needs to answer "which company was
this?" without the caller having to remember to pass company_id on every
single call. structlog's contextvars support solves this: bind company_id
once per request/run via bind_company(), and every log emitted afterwards
on that async task (including from library code deep in the call stack)
automatically carries it — until you clear it or the task ends.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO, json_output: bool = True) -> None:
    """
    Configure structlog + stdlib logging for the whole process.

    Args:
        level: Minimum log level to emit.
        json_output: If True, render logs as JSON (production/CI). If False,
            render as human-readable colored console output (local dev).

    Example:
        >>> configure_logging()
        >>> log = get_logger("ingest")
        >>> log.info("pdf_processed", filename="handbook.pdf", pages=42)
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger. Call configure_logging() once at process start first."""
    return structlog.get_logger(name)


def bind_company(company_id: str) -> None:
    """
    Bind company_id into every subsequent log line on the current context
    (thread/async task), without having to thread it through every function
    signature by hand.

    Args:
        company_id: Tenant identifier to attach to all following log lines.

    Example:
        >>> bind_company("acme_corp")
        >>> log.info("ingestion_started")  # -> {"company_id": "acme_corp", "event": "ingestion_started", ...}
    """
    structlog.contextvars.bind_contextvars(company_id=company_id)


def clear_company_binding() -> None:
    """Remove any bound context vars (company_id and anything else bound this run). Call between tenants."""
    structlog.contextvars.clear_contextvars()
