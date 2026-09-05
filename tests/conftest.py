"""Shared fixtures, and the guard that keeps window tests out of a plain run.

Tests run on a desktop, inside a live session. A test that builds a real Tk
window puts it on screen in front of whatever the person at the keyboard is
doing, and a suite doing that repeatedly is disruptive. So window tests are
opt-in, and opting in is not left to anyone's memory: asking for the `gui_root`
fixture is what marks a test `gui`, and `addopts` in pyproject.toml deselects
that marker. A new window test inherits the exclusion by construction.

Run them deliberately with `pytest -m gui`, and expect windows to appear.
"""
from __future__ import annotations

import tkinter as tk

import pytest

GUI_FIXTURE = "gui_root"


def pytest_collection_modifyitems(items):
    """Mark anything that asks for a real window, however it was written."""
    for item in items:
        if GUI_FIXTURE in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.gui)


@pytest.fixture
def gui_root():
    """A real Tk root window, destroyed afterwards.

    Requesting this is what makes a test `gui`. It is skipped rather than failed
    where no display exists, so the marked suite stays runnable on a headless
    machine.
    """
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display available for GUI tests: {exc}")
    try:
        yield root
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass  # a test may already have torn it down
