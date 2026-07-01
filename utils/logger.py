"""
Logging utilities for the reimbursement agent.
"""

import logging
from pathlib import Path

from config.settings import settings


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure application logging once.

    Args:
        level: Logging level used for root and file handlers.
    """

    Path(settings.LOGS_PATH).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                settings.LOGS_PATH / "travel_reimbursement_agent.log",
                encoding="utf-8",
            ),
        ],
        force=False,
    )
