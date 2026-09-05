"""Theme-helper logic, exercised without constructing a single Tk widget.

`sync_toplevel_theme` only ever reads `.master` and calls
`_apply_appearance_to_window`, so duck-typed stand-ins cover it completely.
Nothing here opens a window, and a plain `pytest` run stays silent.
"""
from __future__ import annotations

from gui.theme_utils import _iter_theme_hosts, sync_toplevel_theme


class FakeHost:
    """A widget-shaped object: a link in a master chain, optionally themed."""

    def __init__(self, master=None, hook=None):
        self.master = master
        if hook is not None:
            self._apply_appearance_to_window = hook


class Recorder:
    """A themable hook that remembers the windows handed to it."""

    def __init__(self, raises: bool = False):
        self.calls = []
        self.raises = raises

    def __call__(self, window):
        self.calls.append(window)
        if self.raises:
            raise RuntimeError("theme host refused the window")


# --- _iter_theme_hosts ----------------------------------------------------

def test_walks_the_master_chain_from_the_widget_outward():
    root = FakeHost()
    middle = FakeHost(master=root)
    leaf = FakeHost(master=middle)
    assert list(_iter_theme_hosts(leaf)) == [leaf, middle, root]


def test_yields_nothing_for_a_detached_widget():
    assert list(_iter_theme_hosts(None)) == []


def test_a_master_cycle_terminates_instead_of_hanging():
    first = FakeHost()
    second = FakeHost(master=first)
    first.master = second
    assert list(_iter_theme_hosts(first)) == [first, second]


def test_a_widget_without_a_master_attribute_ends_the_chain():
    class Bare:
        pass

    bare = Bare()
    assert list(_iter_theme_hosts(bare)) == [bare]


# --- sync_toplevel_theme --------------------------------------------------

def test_the_nearest_themable_host_wins_and_the_walk_stops_there():
    near, far = Recorder(), Recorder()
    outer = FakeHost(hook=far)
    inner = FakeHost(master=outer, hook=near)
    window = object()

    sync_toplevel_theme(inner, window)

    assert near.calls == [window]
    assert far.calls == []


def test_hosts_that_cannot_theme_are_skipped():
    themable = Recorder()
    plain = FakeHost()                      # no hook at all
    not_callable = FakeHost(master=plain)
    not_callable._apply_appearance_to_window = "not a function"
    themable_host = FakeHost(master=not_callable, hook=themable)
    window = object()

    sync_toplevel_theme(themable_host, window)

    assert themable.calls == [window]


def test_a_failing_host_hands_off_to_the_next_one():
    broken, working = Recorder(raises=True), Recorder()
    outer = FakeHost(hook=working)
    inner = FakeHost(master=outer, hook=broken)
    window = object()

    sync_toplevel_theme(inner, window)

    assert broken.calls == [window]
    assert working.calls == [window]


def test_no_window_means_no_host_is_consulted():
    recorder = Recorder()
    sync_toplevel_theme(FakeHost(hook=recorder), None)
    assert recorder.calls == []


def test_an_entirely_unthemed_chain_is_not_an_error():
    sync_toplevel_theme(FakeHost(master=FakeHost()), object())
