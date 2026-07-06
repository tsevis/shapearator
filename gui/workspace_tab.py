from __future__ import annotations

import threading
import tkinter as tk
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from services.config_store import AppSettings
from services.extractor import ExtractionProgress, ExtractionResult, IconExtractor, export_svg_to_png
from services.vision import is_local_url


CANVAS_MODE_LABELS = {
    "original": "A. Keep every icon at its original isolated size on the common canvas.",
    "uniform_to_largest": "B. Scale the largest icon to fit the canvas, then apply that exact same scale to all icons.",
    "individual_fit": "C. Scale each icon individually to fit the canvas while keeping its proportions.",
}

BITMAP_EXPORT_MODE_LABELS = {
    "keep_background": "A. Keep the original background color and fill the full bitmap canvas with it.",
    "transparent_preserve_interior": "B. Export transparent bitmaps while preserving enclosed white or light interior details.",
}

DETECTION_PRESETS = {
    "Balanced": {"padding": 12, "min_area": 200, "merge_gap": 13},
    "Tiny Details": {"padding": 8, "min_area": 70, "merge_gap": 9},
    "Loose Sketches": {"padding": 16, "min_area": 140, "merge_gap": 19},
    "Bold Shapes": {"padding": 14, "min_area": 320, "merge_gap": 15},
}


class WorkspaceTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, settings: AppSettings, on_settings_commit):
        super().__init__(parent)
        self.settings = settings
        self.on_settings_commit = on_settings_commit
        self.input_var = tk.StringVar(value=settings.last_input_path)
        self.output_var = tk.StringVar(value=settings.last_output_dir or str(Path.cwd() / "exports"))
        self.padding_var = tk.IntVar(value=settings.padding)
        self.min_area_var = tk.IntVar(value=settings.min_area)
        self.merge_gap_var = tk.IntVar(value=settings.merge_gap)
        self.detection_preset_var = tk.StringVar(value=self._preset_name_for_values())
        self.output_width_var = tk.IntVar(value=settings.output_width)
        self.output_height_var = tk.IntVar(value=settings.output_height)
        self.canvas_mode_var = tk.StringVar(value=settings.canvas_mode)
        self.bitmap_export_mode_var = tk.StringVar(value=settings.bitmap_export_mode)
        self.provider_summary_var = tk.StringVar(value=self._provider_summary())
        self.status_var = tk.StringVar(value="Choose a sheet, confirm your output canvas, then extract.")
        self.canvas_hint_var = tk.StringVar(value=CANVAS_MODE_LABELS.get(settings.canvas_mode, ""))
        self.progress_label_var = tk.StringVar(value="Idle")
        self.progress_value_var = tk.DoubleVar(value=0.0)
        self.export_png_var = tk.BooleanVar(value="png" in settings.default_formats)
        self.export_jpg_var = tk.BooleanVar(value="jpg" in settings.default_formats)
        self.export_tiff_var = tk.BooleanVar(value="tiff" in settings.default_formats)
        self.export_svg_var = tk.BooleanVar(value="svg" in settings.default_formats)
        self.preview_photo = None
        self.preview_cache_dir = Path(tempfile.mkdtemp(prefix="shapearator_preview_"))
        self.current_result: ExtractionResult | None = None
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        shell = ttk.Frame(self, padding=(18, 18, 18, 18))
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=5)
        shell.columnconfigure(1, weight=4)
        shell.rowconfigure(0, weight=1)

        left_column = ttk.Frame(shell)
        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_column.columnconfigure(0, weight=1)
        left_column.rowconfigure(2, weight=1)

        right_column = ttk.Frame(shell)
        right_column.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_column.columnconfigure(0, weight=1)
        right_column.rowconfigure(0, weight=2)
        right_column.rowconfigure(1, weight=7)

        source_card = ttk.LabelFrame(left_column, text="Source", padding=12)
        source_card.grid(row=0, column=0, sticky="ew")
        source_card.columnconfigure(0, weight=0)
        source_card.columnconfigure(1, weight=1)
        source_card.columnconfigure(2, weight=0)
        ttk.Label(source_card, text="Input Sheet").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_card, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=(10, 10))
        ttk.Button(source_card, text="Browse", command=self._browse_input).grid(row=0, column=2, sticky="e")
        ttk.Label(source_card, text="Output Folder").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(source_card, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=(10, 10), pady=(6, 0))
        ttk.Button(source_card, text="Browse", command=self._browse_output).grid(row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Label(source_card, textvariable=self.provider_summary_var, style="Strong.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(
            source_card,
            text="Tip: SVG input preserves vector cleanliness best. PNG input works beautifully when the shapes are clearly separated.",
            wraplength=620,
            justify="left",
            style="Muted.TLabel",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(3, 0))

        detection_card = ttk.LabelFrame(left_column, text="Detection", padding=12)
        detection_card.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for col in range(8):
            detection_card.columnconfigure(col, weight=1 if col in (1, 3, 5, 7) else 0)
        ttk.Label(detection_card, text="Padding").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(detection_card, from_=0, to=100, textvariable=self.padding_var, width=6).grid(row=0, column=1, sticky="w", padx=(6, 10))
        ttk.Label(detection_card, text="Min Area").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(detection_card, from_=10, to=5000, textvariable=self.min_area_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 10))
        ttk.Label(detection_card, text="Merge").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(detection_card, from_=3, to=99, textvariable=self.merge_gap_var, width=6).grid(row=0, column=5, sticky="w", padx=(6, 12))
        ttk.Label(detection_card, text="Preset").grid(row=0, column=6, sticky="w")
        preset_combo = ttk.Combobox(
            detection_card,
            textvariable=self.detection_preset_var,
            values=list(DETECTION_PRESETS.keys()),
            state="readonly",
            width=16,
        )
        preset_combo.grid(row=0, column=7, sticky="w")
        preset_combo.bind("<<ComboboxSelected>>", self._apply_detection_preset)
        ttk.Label(
            detection_card,
            text="Use Padding to breathe around each icon. Use Min Area to suppress dust. Use Merge to reconnect multi-stroke marks.",
            wraplength=700,
            justify="left",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))

        export_card = ttk.LabelFrame(left_column, text="Output Studio", padding=14)
        export_card.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        export_card.columnconfigure(0, weight=1)
        export_card.columnconfigure(1, weight=1)
        export_card.rowconfigure(1, weight=1)

        output_left = ttk.Frame(export_card)
        output_left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        output_left.columnconfigure(0, weight=1)

        output_right = ttk.Frame(export_card)
        output_right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        output_right.columnconfigure(0, weight=1)

        formats_card = ttk.LabelFrame(output_left, text="Formats", padding=10)
        formats_card.grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(formats_card, text="PNG", variable=self.export_png_var).pack(side="left")
        ttk.Checkbutton(formats_card, text="JPG", variable=self.export_jpg_var).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(formats_card, text="TIFF", variable=self.export_tiff_var).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(formats_card, text="SVG", variable=self.export_svg_var).pack(side="left", padx=(12, 0))

        canvas_card = ttk.LabelFrame(output_left, text="Common Output Canvas", padding=10)
        canvas_card.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        canvas_card.columnconfigure(1, weight=1)
        ttk.Label(canvas_card, text="Width (px)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(canvas_card, from_=32, to=8192, textvariable=self.output_width_var, width=8).grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Label(canvas_card, text="Height (px)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(canvas_card, from_=32, to=8192, textvariable=self.output_height_var, width=8).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(6, 0))
        ttk.Label(
            canvas_card,
            text="These dimensions define the final canvas for every bitmap export and the width and height of SVG exports.",
            wraplength=280,
            justify="left",
            style="Muted.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        bitmap_card = ttk.LabelFrame(output_left, text="Bitmap Export", padding=10)
        bitmap_card.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        bitmap_card.columnconfigure(0, weight=1)
        for row, (mode_key, label) in enumerate(BITMAP_EXPORT_MODE_LABELS.items()):
            ttk.Radiobutton(
                bitmap_card,
                text=label,
                value=mode_key,
                variable=self.bitmap_export_mode_var,
            ).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 8, 0))

        mode_card = ttk.LabelFrame(output_right, text="Canvas Behavior", padding=10)
        mode_card.grid(row=0, column=0, sticky="nsew")
        mode_card.columnconfigure(0, weight=1)
        for row, (mode_key, label) in enumerate(CANVAS_MODE_LABELS.items()):
            ttk.Radiobutton(
                mode_card,
                text=label,
                value=mode_key,
                variable=self.canvas_mode_var,
                command=self._update_canvas_hint,
            ).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 6, 0))
        ttk.Label(mode_card, textvariable=self.canvas_hint_var, wraplength=360, justify="left").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )

        action_card = ttk.LabelFrame(left_column, text="Run", padding=12)
        action_card.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        action_card.columnconfigure(0, weight=1)
        self.extract_button = tk.Button(action_card, text="Extract Shapes", command=self._start_extraction, relief="flat", bd=0)
        self.extract_button.grid(row=0, column=0, sticky="ew")
        ttk.Label(action_card, textvariable=self.status_var, wraplength=760, justify="left").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(action_card, textvariable=self.progress_label_var, style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.progress_bar = ttk.Progressbar(
            action_card,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_value_var,
        )
        self.progress_bar.grid(row=3, column=0, sticky="ew", pady=(5, 0))

        preview_card = ttk.LabelFrame(right_column, text="Preview", padding=10)
        preview_card.grid(row=0, column=0, sticky="nsew")
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(1, weight=1)
        self.preview_meta_var = tk.StringVar(value="No preview yet.")
        ttk.Label(preview_card, textvariable=self.preview_meta_var, wraplength=520, justify="left", style="Strong.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(preview_card, text="No preview yet.", anchor="center")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        results_card = ttk.LabelFrame(right_column, text="Extracted Items", padding=10)
        results_card.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        results_card.columnconfigure(0, weight=1)
        results_card.rowconfigure(0, weight=1)
        self.results_tree = ttk.Treeview(results_card, columns=("formats", "source", "canvas"), show="tree headings", selectmode="browse")
        self.results_tree.heading("#0", text="Name")
        self.results_tree.heading("formats", text="Formats")
        self.results_tree.heading("source", text="Source Size")
        self.results_tree.heading("canvas", text="Canvas")
        self.results_tree.column("#0", width=190, stretch=True)
        self.results_tree.column("formats", width=120, stretch=False)
        self.results_tree.column("source", width=110, stretch=False)
        self.results_tree.column("canvas", width=110, stretch=False)
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        results_scroll = ttk.Scrollbar(results_card, orient="vertical", command=self.results_tree.yview)
        results_scroll.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=results_scroll.set)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_select_result)

    def update_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.provider_summary_var.set(self._provider_summary())

    def _provider_summary(self) -> str:
        if self.settings.provider == "ollama":
            return f"Active provider: Ollama local ({self.settings.ollama_model})"
        if self.settings.provider == "llamacpp":
            return f"Active provider: llama.cpp local ({self.settings.llamacpp_model or 'loaded model'})"
        if self.settings.provider == "directory":
            model_name = self.settings.local_model_name or "directory catalog"
            return f"Active provider: Local directory ({model_name})"
        return "Active provider: Geometry-only local extraction"

    def _update_canvas_hint(self) -> None:
        self.canvas_hint_var.set(CANVAS_MODE_LABELS.get(self.canvas_mode_var.get(), ""))

    def _preset_name_for_values(self) -> str:
        for name, preset in DETECTION_PRESETS.items():
            if (
                preset["padding"] == self.padding_var.get()
                and preset["min_area"] == self.min_area_var.get()
                and preset["merge_gap"] == self.merge_gap_var.get()
            ):
                return name
        return "Balanced"

    def _apply_detection_preset(self, _event=None) -> None:
        preset = DETECTION_PRESETS.get(self.detection_preset_var.get())
        if preset is None:
            return
        self.padding_var.set(preset["padding"])
        self.min_area_var.set(preset["min_area"])
        self.merge_gap_var.set(preset["merge_gap"])

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Supported", "*.png *.svg"), ("PNG", "*.png"), ("SVG", "*.svg")])
        if path:
            self.input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(Path.cwd()))
        if path:
            self.output_var.set(path)

    def _selected_formats(self) -> set[str]:
        formats = set()
        if self.export_png_var.get():
            formats.add("png")
        if self.export_jpg_var.get():
            formats.add("jpg")
        if self.export_tiff_var.get():
            formats.add("tiff")
        if self.export_svg_var.get():
            formats.add("svg")
        return formats

    def _start_extraction(self) -> None:
        formats = self._selected_formats()
        input_path = Path(self.input_var.get().strip())
        output_dir = Path(self.output_var.get().strip())
        if not input_path.exists():
            messagebox.showerror("Missing Input", "Choose a valid PNG or SVG sheet first.")
            return
        if not formats:
            messagebox.showerror("No Export Format", "Choose at least one export format.")
            return
        if self.output_width_var.get() <= 0 or self.output_height_var.get() <= 0:
            messagebox.showerror("Canvas Size", "Canvas width and height must be positive pixel values.")
            return
        if self.settings.provider == "ollama" and not is_local_url(self.settings.ollama_url):
            messagebox.showerror("Local Only", "Ollama must point to a local endpoint such as http://127.0.0.1:11434.")
            return
        if self.settings.provider == "llamacpp" and not is_local_url(self.settings.llamacpp_url):
            messagebox.showerror("Local Only", "llama.cpp must point to a local endpoint such as http://127.0.0.1:8080.")
            return

        self.settings.last_input_path = str(input_path)
        self.settings.last_output_dir = str(output_dir)
        self.settings.default_formats = sorted(formats)
        self.settings.padding = self.padding_var.get()
        self.settings.min_area = self.min_area_var.get()
        self.settings.merge_gap = self.merge_gap_var.get()
        self.settings.output_width = self.output_width_var.get()
        self.settings.output_height = self.output_height_var.get()
        self.settings.canvas_mode = self.canvas_mode_var.get()
        self.settings.bitmap_export_mode = self.bitmap_export_mode_var.get()
        self.on_settings_commit(self.settings)

        self.progress_value_var.set(0.0)
        self.progress_label_var.set("Preparing extraction...")
        self.status_var.set("Extraction running...")

        def worker() -> None:
            try:
                result = IconExtractor(self.settings).extract(
                    input_path,
                    output_dir,
                    formats,
                    progress_callback=self._queue_progress_update,
                )
            except Exception as exc:
                self.after(0, lambda: self._handle_error(exc))
                return
            self.after(0, lambda: self._handle_result(result))

        threading.Thread(target=worker, daemon=True).start()

    def _queue_progress_update(self, progress: ExtractionProgress) -> None:
        self.after(0, lambda p=progress: self._handle_progress(p))

    def _handle_progress(self, progress: ExtractionProgress) -> None:
        self.progress_value_var.set(progress.fraction * 100.0)
        self.progress_label_var.set(progress.message)

    def _handle_error(self, exc: Exception) -> None:
        self.status_var.set("Extraction failed.")
        self.progress_label_var.set("Extraction failed.")
        messagebox.showerror("Extraction Failed", str(exc))

    def _handle_result(self, result: ExtractionResult) -> None:
        self.current_result = result
        self.results_tree.delete(*self.results_tree.get_children())
        for icon in result.icons:
            source_text = f"{icon.source_size[0]} x {icon.source_size[1]}"
            canvas_text = f"{icon.canvas_size[0]} x {icon.canvas_size[1]}"
            self.results_tree.insert(
                "",
                "end",
                iid=str(icon.index),
                text=icon.stem,
                values=(", ".join(sorted(icon.outputs.keys())), source_text, canvas_text),
            )
        self.progress_value_var.set(100.0)
        self.progress_label_var.set(f"Done. Exported {len(result.icons)} icons.")
        self.status_var.set(f"Extracted {len(result.icons)} icons to {result.output_dir}")
        if result.icons:
            self.results_tree.selection_set(str(result.icons[0].index))
            self._show_preview(result.icons[0])

    def _on_select_result(self, _event=None) -> None:
        if self.current_result is None:
            return
        selection = self.results_tree.selection()
        if not selection:
            return
        index = int(selection[0]) - 1
        if 0 <= index < len(self.current_result.icons):
            self._show_preview(self.current_result.icons[index])

    def _show_preview(self, icon) -> None:
        path = self._resolve_preview_path(icon)
        self.preview_meta_var.set(
            f"{icon.stem}  |  source {icon.source_size[0]} x {icon.source_size[1]}  |  canvas {icon.canvas_size[0]} x {icon.canvas_size[1]}"
        )
        if path is None or not path.exists():
            self.preview_photo = None
            self.preview_label.configure(text="No preview available for this item.", image="")
            return
        with Image.open(path).convert("RGBA") as image:
            thumb = ImageOps.contain(image, (420, 420))
        self.preview_photo = ImageTk.PhotoImage(thumb)
        self.preview_label.configure(text="", image=self.preview_photo)

    def _resolve_preview_path(self, icon) -> Path | None:
        if icon.preview_path is not None and icon.preview_path.exists():
            return icon.preview_path
        svg_path = icon.outputs.get("svg")
        if svg_path is None or not svg_path.exists():
            return None
        preview_path = self.preview_cache_dir / f"{icon.stem}_preview.png"
        if not preview_path.exists():
            try:
                export_svg_to_png(svg_path, preview_path)
            except Exception:
                return None
        return preview_path

    def apply_theme(self, mode: str) -> None:
        try:
            self.configure(style="TFrame")
        except Exception:
            pass
        dark = mode == "dark"
        hero_bg = "#6f8ef6" if dark else "#355fd6"
        hero_active = "#5c7be7" if dark else "#284fbf"
        hero_fg = "#ffffff" if dark else "#1f1f1c"
        card_bg = "#23252b" if dark else "#ffffff"
        try:
            self.extract_button.configure(
                bg=hero_bg,
                fg=hero_fg,
                activebackground=hero_active,
                activeforeground=hero_fg,
                disabledforeground=hero_fg,
                font=("TkDefaultFont", 11, "bold"),
                padx=12,
                pady=10,
                highlightthickness=0,
                borderwidth=0,
            )
        except Exception:
            pass
        try:
            self.preview_label.configure(background=card_bg, foreground=hero_fg if dark else "#1f1f1c")
        except Exception:
            pass
