"""
Theme helpers adapted from the Mozaix app shell.
"""

from __future__ import annotations

import tkinter as tk
from typing import Iterator, Optional


def _iter_theme_hosts(start: Optional[tk.Widget]) -> Iterator[object]:
    current = start
    visited = set()
    while current is not None:
        identity = id(current)
        if identity in visited:
            break
        visited.add(identity)
        yield current
        current = getattr(current, "master", None)


def sync_toplevel_theme(parent: tk.Widget, window: tk.Toplevel) -> None:
    if window is None:
        return
    for host in _iter_theme_hosts(parent):
        apply_appearance = getattr(host, "_apply_appearance_to_window", None)
        if callable(apply_appearance):
            try:
                apply_appearance(window)
                return
            except Exception:
                pass


def create_themed_toplevel(parent: tk.Widget) -> tk.Toplevel:
    window = tk.Toplevel(parent)
    sync_toplevel_theme(parent, window)
    try:
        window.after_idle(lambda w=window: sync_toplevel_theme(parent, w))
    except Exception:
        pass
    return window
