"""
Mozaix-style main window shell for Shapearator.
"""

from __future__ import annotations

import platform
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from services.config_store import AppSettings, ConfigStore
from .settings_tab import SettingsTab
from .workspace_tab import WorkspaceTab


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root._mozaix_main_window_ref = self
        self.style = ttk.Style(self.root)
        self._appearance = "auto"
        self._theme_mode = "light"
        self.config_store = ConfigStore(Path("config") / "settings.json")
        self.settings = self.config_store.load()
        self._theme_mode = self.settings.appearance
        self._install_toplevel_theme_hook()
        self.setup_window()
        self._setup_theme()
        self.create_notebook()
        self.create_tabs()
        self.apply_theme(self._theme_mode)
        self.root.after(400, self._maybe_run_first_run)

    def _maybe_run_first_run(self) -> None:
        """Offer the model-download dialog the first time the app runs with no local model."""
        try:
            from services.first_run import needs_first_run

            if not needs_first_run(self.settings):
                return
            from .setup_dialog import SetupDialog

            SetupDialog(self.root, self.settings, on_complete=self._on_setup_complete)
        except Exception:
            pass

    def _on_setup_complete(self, settings: AppSettings) -> None:
        self.settings = settings
        self._save_settings(settings)
        try:
            self.settings_tab.refresh_models()
        except Exception:
            pass
        self.workspace_tab.update_settings(settings)

    def setup_window(self) -> None:
        self.root.title("Shapearator")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        if platform.system() == "Darwin":
            scaling = self.root.winfo_fpixels("1p")
            if scaling > 1.0:
                window_width = int(window_width / scaling)
                window_height = int(window_height / scaling)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(True, True)
        self.root.minsize(1100, 760)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def _setup_theme(self) -> None:
        system = platform.system().lower()
        if system == "darwin":
            self.style.theme_use("aqua")
        elif system == "windows":
            try:
                self.style.theme_use("vista")
            except tk.TclError:
                self.style.theme_use("default")
        else:
            try:
                self.style.theme_use("clam")
            except tk.TclError:
                self.style.theme_use("default")

    def _install_toplevel_theme_hook(self) -> None:
        if getattr(tk.Toplevel, "_shapearator_theme_hook_installed", False):
            return
        original_init = tk.Toplevel.__init__

        def themed_toplevel_init(toplevel_self, *args, **kwargs):
            original_init(toplevel_self, *args, **kwargs)
            try:
                app = None
                current = getattr(toplevel_self, "master", None)
                while current is not None and app is None:
                    app = getattr(current, "_mozaix_main_window_ref", None)
                    current = getattr(current, "master", None)
                if app is not None:
                    app._apply_appearance_to_window(toplevel_self)
            except Exception:
                pass

        tk.Toplevel.__init__ = themed_toplevel_init
        tk.Toplevel._shapearator_theme_hook_installed = True

    def create_notebook(self) -> None:
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.appearance_btn = ttk.Button(self.main_container, text="", command=self.toggle_appearance)
        self.appearance_btn.place(relx=1.0, x=-10, y=8, anchor="ne")

    def create_tabs(self) -> None:
        self.workspace_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.workspace_frame, text="Workspace")
        self.notebook.add(self.settings_frame, text="Settings")

        self.workspace_tab = WorkspaceTab(self.notebook, self.settings, self._save_settings)
        self.settings_tab = SettingsTab(self.notebook, self.settings, self._settings_changed)
        self.workspace_tab.grid(in_=self.workspace_frame, row=0, column=0, sticky="nsew")
        self.settings_tab.grid(in_=self.settings_frame, row=0, column=0, sticky="nsew")
        self.workspace_frame.rowconfigure(0, weight=1)
        self.workspace_frame.columnconfigure(0, weight=1)
        self.settings_frame.rowconfigure(0, weight=1)
        self.settings_frame.columnconfigure(0, weight=1)

    def _settings_changed(self, settings: AppSettings) -> None:
        self.settings = settings
        self.workspace_tab.update_settings(settings)
        self._save_settings(settings)

    def _save_settings(self, settings: AppSettings) -> None:
        settings.appearance = self._theme_mode
        self.settings = settings
        self.config_store.save(settings)

    def toggle_appearance(self) -> None:
        if platform.system() == "Darwin":
            try:
                if self._appearance in ("auto", "aqua") and self._theme_mode != "dark":
                    self.root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", ".", "darkaqua")
                    self._appearance = "darkaqua"
                    self.apply_theme("dark")
                    return
                self.root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", ".", "aqua")
                self._appearance = "aqua"
                self.apply_theme("light")
                return
            except tk.TclError:
                pass
        self.apply_theme("dark" if self._theme_mode == "light" else "light")

    def apply_theme(self, mode: str) -> None:
        self._theme_mode = mode
        dark = mode == "dark"
        colors = {
            "bg": "#17181c" if dark else "#f4f4f1",
            "fg": "#f2f2ef" if dark else "#1f1f1c",
            "muted": "#a8abb2" if dark else "#5a5a56",
            "surface": "#23252b" if dark else "#ffffff",
            "active": "#2f333b" if dark else "#e5e6df",
            "hero": "#6f8ef6" if dark else "#355fd6",
            "hero_text": "#ffffff",
        }
        self.root.configure(bg=colors["bg"])
        self.style.configure(".", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TFrame", background=colors["bg"])
        self.style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TButton", background=colors["surface"], foreground=colors["fg"], padding=6)
        self.style.map("TButton", background=[("active", colors["active"])])
        self.style.configure("TLabelframe", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("Muted.TLabel", background=colors["bg"], foreground=colors["muted"])
        self.style.configure("Strong.TLabel", background=colors["bg"], foreground=colors["fg"], font=("TkDefaultFont", 10, "bold"))
        self.style.configure(
            "Hero.TButton",
            background=colors["hero"],
            foreground=colors["hero_text"],
            padding=(10, 10),
            font=("TkDefaultFont", 11, "bold"),
        )
        self.style.map(
            "Hero.TButton",
            background=[("active", colors["active"]), ("pressed", colors["hero"])],
            foreground=[("active", colors["hero_text"])],
        )
        self.style.configure("TCheckbutton", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TRadiobutton", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TNotebook", background=colors["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=colors["surface"], foreground=colors["fg"], padding=(14, 8))
        self.style.map("TNotebook.Tab", background=[("selected", colors["active"])], foreground=[("selected", colors["fg"])])
        self.style.configure(
            "Treeview",
            background=colors["surface"],
            foreground=colors["fg"],
            fieldbackground=colors["surface"],
            rowheight=28,
        )
        self.style.configure("Treeview.Heading", background=colors["surface"], foreground=colors["fg"], padding=(8, 6))
        self.style.configure("TEntry", fieldbackground=colors["surface"])
        self.style.configure("TCombobox", fieldbackground=colors["surface"])
        self.style.configure("Horizontal.TProgressbar", background="#7aa2ff" if dark else "#4e72df")
        self.appearance_btn.configure(text="Light Mode" if dark else "Dark Mode")
        self.workspace_tab.apply_theme(mode)
        self.settings_tab.apply_theme(mode)
        self._apply_appearance_to_all_toplevels()
        self._save_settings(self.settings)

    def _apply_appearance_to_window(self, window: tk.Toplevel) -> None:
        if window is None or not window.winfo_exists():
            return
        if platform.system() == "Darwin":
            appearance = "darkaqua" if self._theme_mode == "dark" else "aqua"
            try:
                self.root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", window._w, appearance)
            except tk.TclError:
                pass
            return
        dark = self._theme_mode == "dark"
        colors = {
            "bg": "#17181c" if dark else "#f4f4f1",
            "fg": "#f2f2ef" if dark else "#1f1f1c",
            "surface": "#23252b" if dark else "#ffffff",
        }
        try:
            window.configure(bg=colors["bg"])
        except Exception:
            pass

    def _apply_appearance_to_all_toplevels(self) -> None:
        try:
            stack = self.root.tk.splitlist(self.root.tk.call("wm", "stackorder", "."))
        except tk.TclError:
            stack = ()
        for name in stack:
            if name == ".":
                continue
            try:
                widget = self.root.nametowidget(name)
                if isinstance(widget, tk.Toplevel):
                    self._apply_appearance_to_window(widget)
            except Exception:
                continue
