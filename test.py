"""
monday_session_export.py
------------------------
Download Monday.com board groups and save them to ./sessions
"""

from pathlib import Path
import logging
import os
import sys
import traceback

import dotenv
from monday_extract_groups import fetch_data
from datetime import datetime
import time

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "sales_dash.log"


def configure_logger() -> logging.Logger:
    """Build a root logger with a rotating file handler + console echo."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Avoid duplicate logs if another module also imports this file
    logger.propagate = False
    return logger


logger = configure_logger()

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
dotenv.load_dotenv()
API_KEY = os.getenv("MONDAY_API_KEY")
if not API_KEY:
    logger.critical("Environment variable MONDAY_API_KEY not set - aborting.")
    sys.exit(1)

BOARD_ID = os.getenv("BOARD_ID")
GROUP_IDS = [
    "topics",
    "new_group34578__1",
    "new_group27351__1",
    "new_group54376__1",
    "new_group64021__1",
    "new_group65903__1",
    "new_group62617__1",
]
GROUP_NAMES = [
    "scheduled",
    "unqualified",
    "won",
    "cancelled",
    "noshow",
    "proposal",
    "lost",
]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    session_dir = Path("./sessions/sessions")
    session_dir.mkdir(exist_ok=True)

    try:
        logger.info("Requesting Monday.com data for board %s …", BOARD_ID)
        data = fetch_data()
        logger.info("Fetched %d items.", len(data))

        # Save each group to a csv file with timestamp in sessions folder
        logger.info("Saving data to csv files in sessions folder")
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        names = data.keys()
        for name in names:
            logger.info("Saving %s to csv file", name)
            data[name].to_csv(f"{session_dir}/{name}_{timestamp}.csv",
                              index=False)
            logger.info("Saved %s to csv file", name)
        logger.info("All files saved to session.")

    except Exception as exc:
        logger.error("Unhandled exception while fetching Monday data: %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        raise  # re-raise so CI / calling shell knows it failed


if __name__ == "__main__":
    main()
