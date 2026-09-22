from __future__ import annotations

import threading
import tkinter as tk
import tempfile
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from services.config_store import AppSettings
from services.detection_presets import (
    preset_names,
    preset_name_for_values,
    values_for_preset,
)
from services.extractor import ExtractionProgress, ExtractionResult, IconExtractor
from services.preview_cache import resolve_preview_path
from services.request_validation import validate_extraction_request
from services.run_summary import (
    describe_input,
    describe_icon,
    describe_progress_completion,
    describe_result,
    format_size,
    provider_summary,
)
from services.batch import BatchOutcome, extract_folder
from services.semantic_naming import naming_requested
from services.vision import PreflightResult, preflight


CANVAS_MODE_LABELS = {
    "original": "A. Keep every icon at its original isolated size on the common canvas.",
    "uniform_to_largest": "B. Scale the largest icon to fit the canvas, then apply that exact same scale to all icons.",
    "individual_fit": "C. Scale each icon individually to fit the canvas while keeping its proportions.",
}

#: One layered Photoshop file per sheet. Two questions, two pickers: what a
#: layer is made of, and where it sits.
PSD_LAYER_LABELS = {
    "bitmap": "Bitmap layers",
    "bitmap_paths": "Bitmap + editable paths",
    "vector": "Vector shape layers",
}

PSD_LAYOUT_LABELS = {
    "sheet": "Rebuild the sheet",
    "canvas": "Each shape on the export canvas",
}

#: Short enough for the Detection row; the schema value is what gets saved.
SVG_SPLIT_LABELS = {
    "auto": "Auto - follow the artwork",
    "shape": "Every shape its own file",
    "cluster": "Group shapes that touch",
}

BITMAP_EXPORT_MODE_LABELS = {
    "keep_background": "A. Keep the original background color and fill the full bitmap canvas with it.",
    "transparent_preserve_interior": "B. Export transparent bitmaps while preserving enclosed white or light interior details.",
}

def _key_for(labels: dict[str, str], label: str, fallback: str) -> str:
    """Map a dropdown label back to the schema value it stands for.

    An unrecognised label means a settings file written by a newer version;
    falling back is how every other field degrades.
    """
    for key, text in labels.items():
        if text == label:
            return key
    return fallback


def psd_layers_key(label: str) -> str:
    return _key_for(PSD_LAYER_LABELS, label, "bitmap")


def psd_layout_key(label: str) -> str:
    return _key_for(PSD_LAYOUT_LABELS, label, "sheet")


def svg_split_key(label: str) -> str:
    """Map a dropdown label back to the schema value it stands for.

    An unrecognised label means a settings file written by a newer version;
    falling back to ``auto`` is how every other field degrades.
    """
    for key, text in SVG_SPLIT_LABELS.items():
        if text == label:
            return key
    return "auto"


class WorkspaceTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, settings: AppSettings, on_settings_commit):
        super().__init__(parent)
        self.settings = settings
        self.on_settings_commit = on_settings_commit
        self.input_var = tk.StringVar(value=settings.last_input_path)
        self.input_hint_var = tk.StringVar(value=describe_input(Path(settings.last_input_path)))
        # The count must follow a path typed in by hand, not only one picked
        # from the dialog, or the hint quietly describes a previous choice.
        self.input_var.trace_add("write", self._refresh_input_hint)
        self.output_var = tk.StringVar(value=settings.last_output_dir or str(Path.cwd() / "exports"))
        self.padding_var = tk.IntVar(value=settings.padding)
        self.min_area_var = tk.IntVar(value=settings.min_area)
        self.merge_gap_var = tk.IntVar(value=settings.merge_gap)
        self.svg_split_var = tk.StringVar(
            value=SVG_SPLIT_LABELS.get(settings.svg_split, SVG_SPLIT_LABELS["auto"])
        )
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
        self.export_psd_var = tk.BooleanVar(value="psd" in settings.default_formats)
        self.psd_layers_var = tk.StringVar(
            value=PSD_LAYER_LABELS.get(settings.psd_layers, PSD_LAYER_LABELS["bitmap"]))
        self.psd_layout_var = tk.StringVar(
            value=PSD_LAYOUT_LABELS.get(settings.psd_layout, PSD_LAYOUT_LABELS["sheet"]))
        self.preview_photo = None
        self.preview_cache_dir = Path(tempfile.mkdtemp(prefix="shapearator_preview_"))
        self.current_result: ExtractionResult | None = None
        self.current_icons: list = []
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
        ttk.Label(source_card, text="Input").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_card, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=(10, 10))
        input_buttons = ttk.Frame(source_card)
        input_buttons.grid(row=0, column=2, sticky="e")
        ttk.Button(input_buttons, text="Sheet", command=self._browse_input).grid(row=0, column=0)
        ttk.Button(input_buttons, text="Folder", command=self._browse_input_folder).grid(row=0, column=1, padx=(6, 0))
        ttk.Label(source_card, text="Output Folder").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(source_card, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=(10, 10), pady=(6, 0))
        ttk.Button(source_card, text="Browse", command=self._browse_output).grid(row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Label(source_card, textvariable=self.input_hint_var, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(source_card, textvariable=self.provider_summary_var, style="Strong.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
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
        ttk.Label(detection_card, text="Split").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            detection_card,
            textvariable=self.svg_split_var,
            values=list(SVG_SPLIT_LABELS.values()),
            state="readonly",
            width=24,
        ).grid(row=1, column=1, columnspan=3, sticky="w", padx=(6, 10), pady=(8, 0))
        ttk.Label(
            detection_card,
            text="SVG sheets only. Min Area and Merge apply when shapes are grouped.",
            style="Muted.TLabel",
        ).grid(row=1, column=4, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(detection_card, text="Preset").grid(row=0, column=6, sticky="w")
        preset_combo = ttk.Combobox(
            detection_card,
            textvariable=self.detection_preset_var,
            values=list(preset_names()),
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
        ttk.Checkbutton(formats_card, text="PSD", variable=self.export_psd_var).pack(side="left", padx=(12, 0))

        psd_card = ttk.LabelFrame(output_left, text="Photoshop File", padding=10)
        psd_card.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        psd_card.columnconfigure(1, weight=1)
        ttk.Label(psd_card, text="Layers").grid(row=0, column=0, sticky="w")
        ttk.Combobox(psd_card, textvariable=self.psd_layers_var, state="readonly",
                     values=list(PSD_LAYER_LABELS.values()), width=26).grid(
            row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(psd_card, text="Layout").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(psd_card, textvariable=self.psd_layout_var, state="readonly",
                     values=list(PSD_LAYOUT_LABELS.values()), width=26).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        ttk.Label(
            psd_card,
            text="One file per sheet, one layer per shape. Rebuilding the sheet ignores "
                 "the canvas above, because the document takes the artwork's size.",
            style="Muted.TLabel", wraplength=360, justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

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
        return provider_summary(self.settings)

    def _update_canvas_hint(self) -> None:
        self.canvas_hint_var.set(CANVAS_MODE_LABELS.get(self.canvas_mode_var.get(), ""))

    def _preset_name_for_values(self) -> str:
        return preset_name_for_values(
            self.padding_var.get(),
            self.min_area_var.get(),
            self.merge_gap_var.get(),
        )

    def _apply_detection_preset(self, _event=None) -> None:
        preset = values_for_preset(self.detection_preset_var.get())
        if preset is None:
            return
        self.padding_var.set(preset.padding)
        self.min_area_var.set(preset.min_area)
        self.merge_gap_var.set(preset.merge_gap)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Supported", "*.png *.svg"), ("PNG", "*.png"), ("SVG", "*.svg")])
        if path:
            self.input_var.set(path)

    def _browse_input_folder(self) -> None:
        """Pick a folder of sheets; every one inside is extracted in one run."""
        path = filedialog.askdirectory(
            title="Choose a folder of sheets",
            initialdir=self.input_var.get() or str(Path.cwd()),
        )
        if path:
            self.input_var.set(path)

    def _refresh_input_hint(self, *_args) -> None:
        self.input_hint_var.set(describe_input(Path(self.input_var.get().strip())))

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
        if self.export_psd_var.get():
            formats.add("psd")
        return formats

    def _start_extraction(self) -> None:
        formats = self._selected_formats()
        input_path = Path(self.input_var.get().strip())
        output_dir = Path(self.output_var.get().strip())
        # Validate a candidate carrying the spinbox values rather than writing
        # them in first, so a rejected click leaves the live settings untouched.
        candidate = replace(
            self.settings,
            output_width=self.output_width_var.get(),
            output_height=self.output_height_var.get(),
        )
        issue = validate_extraction_request(candidate, input_path, formats)
        if issue is not None:
            messagebox.showerror(issue.title, issue.message)
            return

        self.settings.last_input_path = str(input_path)
        self.settings.last_output_dir = str(output_dir)
        self.settings.default_formats = sorted(formats)
        self.settings.padding = self.padding_var.get()
        self.settings.min_area = self.min_area_var.get()
        self.settings.merge_gap = self.merge_gap_var.get()
        self.settings.svg_split = svg_split_key(self.svg_split_var.get())
        self.settings.psd_layers = psd_layers_key(self.psd_layers_var.get())
        self.settings.psd_layout = psd_layout_key(self.psd_layout_var.get())
        self.settings.output_width = self.output_width_var.get()
        self.settings.output_height = self.output_height_var.get()
        self.settings.canvas_mode = self.canvas_mode_var.get()
        self.settings.bitmap_export_mode = self.bitmap_export_mode_var.get()
        self.on_settings_commit(self.settings)

        allow_unnamed = self._confirm_semantic_backend()
        if allow_unnamed is None:
            return

        self.progress_value_var.set(0.0)
        self.progress_label_var.set("Preparing extraction...")
        self.status_var.set("Extraction running...")

        def worker() -> None:
            try:
                if input_path.is_dir():
                    outcome = extract_folder(
                        self.settings, input_path, output_dir, formats,
                        progress_callback=self._queue_progress_update,
                        allow_unnamed=allow_unnamed,
                    )
                    self.after(0, lambda value=outcome: self._handle_batch_result(value))
                    return
                result = IconExtractor(self.settings).extract(
                    input_path,
                    output_dir,
                    formats,
                    progress_callback=self._queue_progress_update,
                    allow_unnamed=allow_unnamed,
                )
            except Exception as exc:
                # Bind now: `exc` is unbound once the except block exits, so a
                # bare closure would raise NameError instead of showing the error.
                self.after(0, lambda error=exc: self._handle_error(error))
                return
            self.after(0, lambda value=result: self._handle_result(value))

        threading.Thread(target=worker, daemon=True).start()

    def _confirm_semantic_backend(self) -> bool | None:
        """Check the vision backend before exporting anything.

        Returns the ``allow_unnamed`` flag to run with, or ``None`` if the user
        cancelled. Checking here rather than mid-run means an unreachable model
        costs a few seconds instead of a full export that silently produced
        generic filenames.
        """
        if not naming_requested(self.settings):
            return False

        self.status_var.set("Checking local model backend...")
        self.update_idletasks()
        try:
            result = preflight(self.settings)
        except Exception as exc:  # a broken endpoint must not kill the click
            result = PreflightResult(False, self.settings.provider, str(exc))

        if result.ok:
            self.status_var.set("Model ready.")
            return False

        self.status_var.set("Model backend not ready.")
        proceed = messagebox.askyesno(
            "Model Not Ready",
            f"{result.message}\n\n"
            "Export anyway with generic filenames (icon_001, icon_002, ...)?\n"
            "Metadata will record that no model named these icons.",
            default=messagebox.NO,
        )
        return True if proceed else None

    def _queue_progress_update(self, progress: ExtractionProgress) -> None:
        self.after(0, lambda p=progress: self._handle_progress(p))

    def _handle_progress(self, progress: ExtractionProgress) -> None:
        self.progress_value_var.set(progress.fraction * 100.0)
        self.progress_label_var.set(progress.message)

    def _handle_error(self, exc: Exception) -> None:
        self.status_var.set("Extraction failed.")
        self.progress_label_var.set("Extraction failed.")
        messagebox.showerror("Extraction Failed", str(exc))

    def _fill_results(self, icons, status: str, labels: list[str] | None = None) -> None:
        """Populate the table from any run.

        Rows are keyed by position, not by ``icon.index``: a folder run numbers
        every sheet's icons from one, so five sheets would all claim the id
        "1" and Tk would refuse the second.
        """
        self.current_icons = list(icons)
        self.results_tree.delete(*self.results_tree.get_children())
        for position, icon in enumerate(self.current_icons):
            self.results_tree.insert(
                "",
                "end",
                iid=str(position),
                text=labels[position] if labels else icon.stem,
                values=(
                    ", ".join(sorted(icon.outputs.keys())),
                    format_size(icon.source_size),
                    format_size(icon.canvas_size),
                ),
            )
        self.progress_value_var.set(100.0)
        self.progress_label_var.set(describe_progress_completion(len(self.current_icons)))
        self.status_var.set(status)
        if self.current_icons:
            self.results_tree.selection_set("0")
            self._show_preview(self.current_icons[0])

    def _handle_result(self, result: ExtractionResult) -> None:
        self.current_result = result
        self._fill_results(result.icons, describe_result(result))
        if result.warnings:
            messagebox.showwarning("Completed With Warnings", "\n\n".join(result.warnings))

    def _handle_batch_result(self, outcome: BatchOutcome) -> None:
        self.current_result = None
        # Every sheet numbers its icons from one, so naming the sheet is the
        # only thing that tells five rows called icon_001 apart. A run that
        # covered one sheet needs no prefix.
        extracted = [sheet for sheet in outcome.sheets if not sheet.failed]
        labels = None
        if len(extracted) > 1:
            labels = [
                f"{sheet.output_dir.name}/{icon.stem}"
                for sheet in extracted
                for icon in sheet.icons
            ]
        self._fill_results(outcome.icons, outcome.summary(), labels)
        if outcome.warnings:
            messagebox.showwarning("Completed With Warnings", "\n\n".join(outcome.warnings))

    def _on_select_result(self, _event=None) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        position = int(selection[0])
        if 0 <= position < len(self.current_icons):
            self._show_preview(self.current_icons[position])

    def _show_preview(self, icon) -> None:
        path = self._resolve_preview_path(icon)
        self.preview_meta_var.set(describe_icon(icon))
        if path is None or not path.exists():
            self.preview_photo = None
            self.preview_label.configure(text="No preview available for this item.", image="")
            return
        with Image.open(path).convert("RGBA") as image:
            thumb = ImageOps.contain(image, (420, 420))
        self.preview_photo = ImageTk.PhotoImage(thumb)
        self.preview_label.configure(text="", image=self.preview_photo)

    def _resolve_preview_path(self, icon) -> Path | None:
        return resolve_preview_path(icon, self.preview_cache_dir)

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
