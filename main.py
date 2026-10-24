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


ICON_PATH = Path(__file__).resolve().parent / "gui" / "assets" / "AppIcon-256.png"


def set_app_icon(root: tk.Tk) -> None:
    """Give the window its app icon; a missing or unreadable file costs only the icon."""
    try:
        icon = tk.PhotoImage(master=root, file=str(ICON_PATH))
        root.iconphoto(True, icon)
        root._app_icon = icon  # Tk drops the image if Python lets go of it
    except (tk.TclError, OSError) as exc:
        logging.getLogger(__name__).warning("Could not load app icon %s: %s", ICON_PATH, exc)


def main() -> None:
    try:
        setup_logging()
        sys.excepthook = handle_exception
        root = tk.Tk()
        set_app_icon(root)
        MainWindow(root)
        root.mainloop()
    except Exception as exc:
        logging.error("Fatal error: %s", exc)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
