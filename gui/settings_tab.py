from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from services.config_store import AppSettings
from services.llamacpp_models import StartableLlamaModel, available_startable_models
from services.llamacpp_server import LlamaCppServerManager
from services.model_catalog import spec_for_model_name
from services.model_registry import ModelDescriptor, ModelRegistry


class SettingsTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, settings: AppSettings, on_change):
        super().__init__(parent)
        self.settings = settings
        self.on_change = on_change
        self.registry = ModelRegistry()
        self.provider_var = tk.StringVar(value=settings.provider)
        self.ollama_url_var = tk.StringVar(value=settings.ollama_url)
        self.ollama_model_var = tk.StringVar(value=settings.ollama_model)
        self.llamacpp_url_var = tk.StringVar(value=settings.llamacpp_url)
        self.llamacpp_model_var = tk.StringVar(value=settings.llamacpp_model)
        self.local_root_var = tk.StringVar(value=settings.local_model_root)
        self.local_model_var = tk.StringVar(value=settings.local_model_name)
        self.semantic_var = tk.BooleanVar(value=settings.semantic_naming)
        self.status_var = tk.StringVar(value="Local-only providers. Cloud models are ignored.")
        self.ollama_recommendation_var = tk.StringVar(value="")
        self.llamacpp_recommendation_var = tk.StringVar(value="")
        self.ollama_models: list[ModelDescriptor] = []
        self.llamacpp_models: list[ModelDescriptor] = []
        self.startable_llamacpp: list[StartableLlamaModel] = []
        self.server_manager = LlamaCppServerManager()
        self._server_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._build()
        self.refresh_models()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        card = ttk.Frame(self, padding=18)
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Model Provider", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))
        provider_row = ttk.Frame(card)
        provider_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Radiobutton(provider_row, text="Geometry Only", value="geometry", variable=self.provider_var, command=self._emit_change).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(provider_row, text="Ollama Local", value="ollama", variable=self.provider_var, command=self._emit_change).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(provider_row, text="llama.cpp Local", value="llamacpp", variable=self.provider_var, command=self._emit_change).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(provider_row, text="Model Directory", value="directory", variable=self.provider_var, command=self._emit_change).pack(side="left")

        ttk.Separator(card).grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)

        # Ollama provider settings
        ttk.Label(card, text="Ollama URL").grid(row=3, column=0, sticky="w")
        ttk.Entry(card, textvariable=self.ollama_url_var).grid(row=3, column=1, sticky="ew", padx=(12, 12))
        ttk.Button(card, text="Refresh Ollama Models", command=self.refresh_models).grid(row=3, column=2, sticky="e")

        ttk.Label(card, text="Ollama Model").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self.ollama_combo = ttk.Combobox(card, textvariable=self.ollama_model_var, state="readonly")
        self.ollama_combo.grid(row=4, column=1, sticky="ew", padx=(12, 12), pady=(10, 0))
        self.ollama_combo.bind("<<ComboboxSelected>>", lambda _event: self._emit_change())
        ttk.Button(card, text="Use Recommended", command=self._use_recommended_ollama).grid(row=4, column=2, sticky="e", pady=(10, 0))
        ttk.Label(card, textvariable=self.ollama_recommendation_var, wraplength=760, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        ttk.Separator(card).grid(row=6, column=0, columnspan=3, sticky="ew", pady=10)

        # llama.cpp provider settings
        ttk.Label(card, text="llama.cpp URL").grid(row=7, column=0, sticky="w")
        ttk.Entry(card, textvariable=self.llamacpp_url_var).grid(row=7, column=1, sticky="ew", padx=(12, 12))
        ttk.Button(card, text="Refresh llama.cpp Models", command=self.refresh_models).grid(row=7, column=2, sticky="e")

        ttk.Label(card, text="llama.cpp Model").grid(row=8, column=0, sticky="w", pady=(10, 0))
        self.llamacpp_combo = ttk.Combobox(card, textvariable=self.llamacpp_model_var, state="readonly")
        self.llamacpp_combo.grid(row=8, column=1, sticky="ew", padx=(12, 12), pady=(10, 0))
        self.llamacpp_combo.bind("<<ComboboxSelected>>", lambda _event: self._emit_change())
        llamacpp_buttons = ttk.Frame(card)
        llamacpp_buttons.grid(row=8, column=2, sticky="e", pady=(10, 0))
        ttk.Button(llamacpp_buttons, text="Use Recommended", command=self._use_recommended_llamacpp).pack(side="left", padx=(0, 6))
        self.start_server_btn = ttk.Button(llamacpp_buttons, text="Start Server", command=self._start_llamacpp_server)
        self.start_server_btn.pack(side="left")
        ttk.Label(card, textvariable=self.llamacpp_recommendation_var, wraplength=760, justify="left").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        ttk.Separator(card).grid(row=10, column=0, columnspan=3, sticky="ew", pady=10)

        # Directory catalog settings
        ttk.Label(card, text="Model Directory").grid(row=11, column=0, sticky="w")
        ttk.Entry(card, textvariable=self.local_root_var).grid(row=11, column=1, sticky="ew", padx=(12, 12))
        ttk.Button(card, text="Browse", command=self._browse_model_dir).grid(row=11, column=2, sticky="e")

        ttk.Label(card, text="Directory Model").grid(row=12, column=0, sticky="w", pady=(10, 0))
        self.local_combo = ttk.Combobox(card, textvariable=self.local_model_var, state="readonly")
        self.local_combo.grid(row=12, column=1, columnspan=2, sticky="ew", padx=(12, 0), pady=(10, 0))
        self.local_combo.bind("<<ComboboxSelected>>", lambda _event: self._emit_change())

        ttk.Separator(card).grid(row=13, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Checkbutton(
            card,
            text="Use selected model for semantic file naming and metadata when supported",
            variable=self.semantic_var,
            command=self._emit_change,
        ).grid(row=14, column=0, columnspan=3, sticky="w")

        ttk.Label(
            card,
            text="Recommended local vision stack for this app: Qwen3-VL first, then Qwen2.5-VL, MiniCPM-V, moondream for fastest passes, LLaVA as fallback. Works with either Ollama or a llama.cpp server.",
            wraplength=780,
            justify="left",
        ).grid(row=15, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(card, textvariable=self.status_var, foreground="#6a6a6a").grid(row=16, column=0, columnspan=3, sticky="w", pady=(12, 0))

    def _browse_model_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.local_root_var.get() or str(Path.home()))
        if chosen:
            self.local_root_var.set(chosen)
            self.refresh_models()
            self._emit_change()

    def refresh_models(self) -> None:
        ollama_models = self.registry.list_ollama_models()
        self.ollama_models = ollama_models
        llamacpp_models = self.registry.list_llamacpp_models(self.llamacpp_url_var.get().strip())
        self.llamacpp_models = llamacpp_models
        self.startable_llamacpp = available_startable_models(self.settings.models_root)
        directory_models = self.registry.list_directory_models(self.local_root_var.get())

        # The llama.cpp dropdown shows the running server's model plus every model
        # the app can launch on demand: ones it downloaded and ones already in
        # llama.cpp's cache. Recognized models use their clean catalog name rather
        # than the raw GGUF path llama-server reports for -m launches.
        llamacpp_names: list[str] = []
        for item in llamacpp_models:
            spec = spec_for_model_name(item.name)
            name = spec.display_name if spec is not None else item.name
            if name not in llamacpp_names:
                llamacpp_names.append(name)
        for model in self.startable_llamacpp:
            if model.display_name not in llamacpp_names:
                llamacpp_names.append(model.display_name)

        self.ollama_combo["values"] = [item.name for item in ollama_models]
        self.llamacpp_combo["values"] = llamacpp_names
        self.local_combo["values"] = [item.name for item in directory_models]

        if self.ollama_model_var.get() not in self.ollama_combo["values"] and self.ollama_combo["values"]:
            recommended = self.registry.recommended_ollama_model(ollama_models)
            self.ollama_model_var.set(recommended.name if recommended is not None else self.ollama_combo["values"][0])
        if self.llamacpp_model_var.get() not in llamacpp_names and llamacpp_names:
            self.llamacpp_model_var.set(llamacpp_names[0])
        if self.local_model_var.get() not in self.local_combo["values"] and self.local_combo["values"]:
            self.local_model_var.set(self.local_combo["values"][0])

        self._update_server_button_state()
        self._update_ollama_recommendation()
        self._update_llamacpp_recommendation()
        startable = len(self.startable_llamacpp)
        self.status_var.set(
            f"Discovered {len(ollama_models)} Ollama, {len(llamacpp_models)} llama.cpp running "
            f"({startable} available to start), and {len(directory_models)} directory models."
        )
        self._emit_change()

    def _update_server_button_state(self) -> None:
        # Offer Start Server only when a model can be launched and none is running here.
        can_start = bool(self.startable_llamacpp) and not self.llamacpp_models
        try:
            self.start_server_btn.state(["!disabled"] if can_start else ["disabled"])
        except tk.TclError:
            pass

    def _update_ollama_recommendation(self) -> None:
        self.ollama_recommendation_var.set(
            self._recommendation_text(
                self.ollama_models,
                self.ollama_model_var.get().strip(),
                self.registry.recommended_ollama_model(self.ollama_models),
            )
        )

    def _update_llamacpp_recommendation(self) -> None:
        text = self._recommendation_text(
            self.llamacpp_models,
            self.llamacpp_model_var.get().strip(),
            self.registry.recommended_llamacpp_model(self.llamacpp_models),
        )
        if not self.llamacpp_models:
            if self.startable_llamacpp:
                names = ", ".join(model.display_name for model in self.startable_llamacpp)
                text = (
                    f"No server running. Ready to start (downloaded or in your llama.cpp cache): {names}. "
                    "Pick one and click Start Server — the app launches llama-server for you."
                )
            else:
                text = (
                    "No llama.cpp server running and no model available yet. "
                    "Download one via first-run setup or `python shapearator.py --setup`, then Start Server."
                )
        self.llamacpp_recommendation_var.set(text)

    @staticmethod
    def _recommendation_text(
        models: list[ModelDescriptor],
        selected_name: str,
        recommended: ModelDescriptor | None,
    ) -> str:
        current = next((model for model in models if model.name == selected_name), None)
        lines = []
        if recommended is not None:
            lines.append(f"Recommended: {recommended.name}. {recommended.recommendation}")
        if current is not None and current.name != getattr(recommended, "name", None):
            lines.append(f"Selected: {current.name}. {current.recommendation}")
        elif current is not None:
            lines.append(f"Selected model note: {current.recommendation}")
        return " ".join(lines)

    def _use_recommended_ollama(self) -> None:
        recommended = self.registry.recommended_ollama_model(self.ollama_models)
        if recommended is None:
            return
        self.ollama_model_var.set(recommended.name)
        self.semantic_var.set(True)
        self.provider_var.set("ollama")
        self._update_ollama_recommendation()
        self._emit_change()

    def _use_recommended_llamacpp(self) -> None:
        recommended = self.registry.recommended_llamacpp_model(self.llamacpp_models)
        if recommended is not None:
            self.llamacpp_model_var.set(recommended.name)
        elif self.startable_llamacpp:
            # No server running; pre-select the best startable model to launch.
            self.llamacpp_model_var.set(self.startable_llamacpp[0].display_name)
        else:
            self.status_var.set(
                "No llama.cpp model available. Download one via first-run setup or `python shapearator.py --setup`."
            )
            return
        self.semantic_var.set(True)
        self.provider_var.set("llamacpp")
        self._update_llamacpp_recommendation()
        self._emit_change()

    def _selected_startable(self) -> StartableLlamaModel | None:
        """Return the StartableLlamaModel matching the dropdown, or the best one."""
        selected = self.llamacpp_model_var.get().strip()
        for model in self.startable_llamacpp:
            if model.display_name == selected:
                return model
        return self.startable_llamacpp[0] if self.startable_llamacpp else None

    def _start_llamacpp_server(self) -> None:
        model = self._selected_startable()
        if model is None:
            self.status_var.set(
                "No llama.cpp model to start. Use first-run setup or `python shapearator.py --setup`."
            )
            return
        url = self.llamacpp_url_var.get().strip()
        self.start_server_btn.state(["disabled"])
        source_hint = "from cache" if model.source == "cache" else "from download"
        self.status_var.set(f"Starting llama-server with {model.display_name} ({source_hint})… first load can take a moment.")

        def worker() -> None:
            try:
                if model.source == "cache" and model.hf_ref:
                    self.server_manager.start_hf(model.hf_ref, url)
                elif model.files is not None:
                    self.server_manager.start(model.files, url)
                else:
                    raise RuntimeError("Model has no launch source.")
                self._server_queue.put(("ok", model.display_name))
            except Exception as exc:  # surfaced to the UI thread
                self._server_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(250, self._poll_server_start)

    def _poll_server_start(self) -> None:
        try:
            kind, payload = self._server_queue.get_nowait()
        except queue.Empty:
            self.after(250, self._poll_server_start)
            return
        try:
            self.start_server_btn.state(["!disabled"])
        except tk.TclError:
            pass
        if kind == "ok":
            self.provider_var.set("llamacpp")
            self.semantic_var.set(True)
            self.llamacpp_model_var.set(payload)
            self.refresh_models()
            self.status_var.set(f"llama.cpp server running with {payload}.")
        else:
            self.status_var.set(f"Could not start llama-server: {payload}")

    def shutdown(self) -> None:
        """Stop any app-launched llama-server. Call on app close."""
        try:
            self.server_manager.stop()
        except Exception:
            pass

    def get_settings(self) -> AppSettings:
        self.settings.provider = self.provider_var.get()
        self.settings.ollama_url = self.ollama_url_var.get().strip()
        self.settings.ollama_model = self.ollama_model_var.get().strip()
        self.settings.llamacpp_url = self.llamacpp_url_var.get().strip()
        self.settings.llamacpp_model = self.llamacpp_model_var.get().strip()
        self.settings.local_model_root = self.local_root_var.get().strip()
        self.settings.local_model_name = self.local_model_var.get().strip()
        self.settings.semantic_naming = self.semantic_var.get()
        return self.settings

    def _emit_change(self) -> None:
        self._update_ollama_recommendation()
        self._update_llamacpp_recommendation()
        self.on_change(self.get_settings())

    def apply_theme(self, mode: str) -> None:
        try:
            self.configure(style="TFrame")
        except Exception:
            pass
