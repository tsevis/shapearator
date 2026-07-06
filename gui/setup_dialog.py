"""First-run model download dialog.

Shown once when no local vision model is available, so a freshly cloned copy of
the app can fetch everything it needs to caption icons — for Ollama and/or
llama.cpp — with a single click.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from services import first_run as fr
from services.config_store import AppSettings


class SetupDialog:
    def __init__(self, parent: tk.Misc, settings: AppSettings, on_complete=None):
        self.parent = parent
        self.settings = settings
        self.on_complete = on_complete
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.status = fr.detect_backends(settings)
        self.candidates = fr.build_candidates(settings, self.status)
        self.row_vars: list[tuple[fr.SetupCandidate, tk.BooleanVar]] = []
        self._busy = False

        self.win = tk.Toplevel(parent)
        self.win.title("Shapearator — First-Run Model Setup")
        self.win.transient(parent)
        self.win.geometry("640x560")
        self.win.protocol("WM_DELETE_WINDOW", self._skip)
        self._build()
        try:
            self.win.grab_set()
        except tk.TclError:
            pass

    def _build(self) -> None:
        outer = ttk.Frame(self.win, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Download local vision models", font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            outer,
            text=(
                "Shapearator can name icons using a local vision model. Pick which models to "
                "download for the backend(s) you use. Everything stays on your machine."
            ),
            wraplength=600,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))

        backend_bits = []
        backend_bits.append("Ollama: running" if self.status.ollama_reachable else "Ollama: not detected")
        backend_bits.append("llama.cpp: available" if self.status.llamacpp_binary else "llama.cpp: not installed")
        ttk.Label(outer, text="Detected backends → " + "   ·   ".join(backend_bits), style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 10)
        )

        list_frame = ttk.LabelFrame(outer, text="Available models", padding=12)
        list_frame.grid(row=3, column=0, sticky="nsew")
        outer.rowconfigure(3, weight=1)

        if not self.candidates:
            ttk.Label(
                list_frame,
                text="No supported backend detected. Install Ollama or llama.cpp, then reopen the app.",
                wraplength=560,
                justify="left",
            ).pack(anchor="w")
        for candidate in self.candidates:
            var = tk.BooleanVar(value=candidate.default_selected and not candidate.installed)
            self.row_vars.append((candidate, var))
            check = ttk.Checkbutton(list_frame, text=candidate.label, variable=var)
            if candidate.installed:
                check.state(["disabled"])
                var.set(False)
            check.pack(anchor="w", pady=2)

        self.progress = ttk.Progressbar(outer, mode="determinate", maximum=1000)
        self.progress.grid(row=4, column=0, sticky="ew", pady=(12, 4))
        self.status_var = tk.StringVar(value="Select models and choose Download, or skip for now.")
        ttk.Label(outer, textvariable=self.status_var, style="Muted.TLabel", wraplength=600, justify="left").grid(
            row=5, column=0, sticky="w"
        )

        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, sticky="e", pady=(12, 0))
        self.skip_btn = ttk.Button(buttons, text="Skip for now", command=self._skip)
        self.skip_btn.pack(side="right", padx=(8, 0))
        self.download_btn = ttk.Button(buttons, text="Download selected", style="Hero.TButton", command=self._start_download)
        self.download_btn.pack(side="right")
        if not self.candidates:
            self.download_btn.state(["disabled"])

    # --- actions ----------------------------------------------------------

    def _selected(self) -> list[fr.SetupCandidate]:
        return [c for c, var in self.row_vars if var.get() and not c.installed]

    def _start_download(self) -> None:
        selection = self._selected()
        if not selection:
            messagebox.showinfo("Nothing selected", "Tick at least one model to download, or choose Skip.")
            return
        self._busy = True
        self.download_btn.state(["disabled"])
        self.skip_btn.state(["disabled"])
        worker = threading.Thread(target=self._run_installs, args=(selection,), daemon=True)
        worker.start()
        self.win.after(100, self._drain_events)

    def _run_installs(self, selection: list[fr.SetupCandidate]) -> None:
        try:
            for candidate in selection:
                self.events.put(("model", candidate))
                fr.install_candidate(self.settings, candidate, lambda p: self.events.put(("progress", p)))
            self.events.put(("complete", selection[0]))
        except Exception as exc:  # surfaced to the UI thread
            self.events.put(("error", str(exc)))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "model":
                    self.status_var.set(f"Preparing {payload.spec.display_name} ({payload.backend})…")
                elif kind == "progress":
                    self._show_progress(payload)
                elif kind == "complete":
                    self._finish(payload)
                    return
                elif kind == "error":
                    self._fail(str(payload))
                    return
        except queue.Empty:
            pass
        if self._busy:
            self.win.after(120, self._drain_events)

    def _show_progress(self, progress) -> None:
        self.progress["value"] = int(progress.fraction * 1000)
        pct = f" {progress.fraction * 100:.0f}%" if progress.total else ""
        self.status_var.set(f"{progress.message}{pct}")

    def _finish(self, first: fr.SetupCandidate) -> None:
        self._busy = False
        self.progress["value"] = 1000
        fr.apply_active_model(self.settings, first)
        fr.mark_setup_complete({"installed": [c.spec.key for c, v in self.row_vars if v.get()]})
        self.status_var.set("Done. Models are ready.")
        if self.on_complete is not None:
            try:
                self.on_complete(self.settings)
            except Exception:
                pass
        self._close()

    def _fail(self, message: str) -> None:
        self._busy = False
        self.download_btn.state(["!disabled"])
        self.skip_btn.state(["!disabled"])
        self.status_var.set(f"Download failed: {message}")
        messagebox.showerror("Download failed", message)

    def _skip(self) -> None:
        if self._busy:
            return
        fr.mark_setup_complete({"skipped": True})
        self._close()

    def _close(self) -> None:
        try:
            self.win.grab_release()
        except tk.TclError:
            pass
        self.win.destroy()
