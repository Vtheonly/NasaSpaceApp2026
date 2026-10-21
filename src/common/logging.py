"""Minimal logging helper with process-wide one-time configuration."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "[%(asctime)s %(levelname)s %(name)s] %(message)s"


def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Configure root logging once (unless *force*)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    handlers = [logging.StreamHandler(sys.stderr)]
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=force)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root logger on first use."""
    configure_logging()
    return logging.getLogger(name)
