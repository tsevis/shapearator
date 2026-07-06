"""
Shapearator desktop application entry point.
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path

from gui.main_window import MainWindow


def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"shapearator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info("Logging initialized: %s", log_file)


def handle_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def main() -> None:
    try:
        setup_logging()
        sys.excepthook = handle_exception
        root = tk.Tk()
        MainWindow(root)
        root.mainloop()
    except Exception as exc:
        logging.error("Fatal error: %s", exc)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
