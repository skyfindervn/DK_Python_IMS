"""
Shared database utilities.
Provides a dummy connection function for use across all apps in the monorepo.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def connect_db(url: str = "sqlite:///./dev.db") -> dict:
    """
    Dummy database connection function.

    In production, replace with a real connection (e.g., SQLAlchemy engine).

    Args:
        url: Database connection URL.

    Returns:
        A dict representing a mock connection handle.
    """
    logger.info("Connecting to database: %s", url)
    # TODO: Replace with real driver, e.g.:
    #   from sqlalchemy import create_engine
    #   return create_engine(url)
    return {"connected": True, "url": url}
