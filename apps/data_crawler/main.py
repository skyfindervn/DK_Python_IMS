"""
Data Crawler – entry point.

Run with:
    uv run python apps/data_crawler/main.py
"""

import logging

import requests

from shared.database import connect_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


TARGET_URL = "https://httpbin.org/get"


def crawl(url: str = TARGET_URL) -> dict:
    """
    Fetch data from *url* and persist it (stubbed).

    Args:
        url: The endpoint to crawl.

    Returns:
        Parsed JSON response as a dict.
    """
    logger.info("Crawling: %s", url)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    logger.info("Received %d bytes", len(response.content))

    # Stub: in a real crawler you would persist *data* via the shared DB
    db = connect_db()
    logger.info("Would persist data using connection: %s", db["url"])

    return data


if __name__ == "__main__":
    result = crawl()
    print(result)
