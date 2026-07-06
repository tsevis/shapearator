# File Structure

This document explains the main repository layout for Shapearator and distinguishes source code from generated or sample assets.

## Top Level

```text
shapearator/
  config/            # persisted settings + first-run marker (runtime state)
  docs/              # sample sheets, exports, screenshots (mostly not source)
  documents/         # reserved working area
  gui/               # desktop interface (tkinter/ttk)
  logs/              # generated run logs
  models/            # downloaded llama.cpp GGUF weights (gitignored)
  services/          # shared engine, model backends, settings, setup logic
  tests/             # pytest suite
  main.py            # GUI entrypoint
  shapearator.py     # CLI entrypoint
  run.sh             # GUI launcher (installs deps on first run)
  requirements.txt   # Python dependencies
  README.md
  MANUAL.md
  FILE_STRUCTURE.md
  LICENSE
```

## Core Application Code

### `main.py`

Primary GUI entrypoint.

Responsibilities:

- initializes logging
- creates the Tk root window
- starts the main desktop app shell (which triggers first-run model setup when needed)

### `shapearator.py`

Primary CLI entrypoint.

Responsibilities:

- parses CLI arguments
- runs the headless model installer for `--setup` / `--setup-all`
- optionally loads `config/settings.json`
- applies detection presets and explicit overrides
- validates provider and export settings (local-only endpoints)
- prints a provider preflight readout before a naming run
- runs extraction through the shared engine
- optionally saves resolved config

### `run.sh`

Convenience launcher for the GUI.

Responsibilities:

- resolves the repository root
- selects a Python interpreter (`PYTHON_BIN` override, project `.venv`, known pyenv, or `python3`)
- installs `requirements.txt` on first run if dependencies are missing
- starts `main.py`

### `requirements.txt`

Python dependency manifest: `Pillow`, `opencv-python`, `numpy`, `requests`, `huggingface_hub`.
External tools (`inkscape`, `potrace`) and model backends (Ollama, llama.cpp) are installed separately.

## GUI Layer

### `gui/`

Desktop application interface built with `tkinter` and `ttk`.

Important files:

- `main_window.py`: top-level app shell, notebook tabs, theme management, first-run trigger
- `workspace_tab.py`: extraction workflow, previews, results list, run controls
- `settings_tab.py`: provider selection, Ollama settings, llama.cpp settings, directory model settings
- `setup_dialog.py`: first-run model-download dialog (backend detection, per-model checkboxes, threaded progress)
- `theme_utils.py`: helper utilities for themed toplevel windows

This folder is source code.

## Shared Services

### `services/`

Shared backend logic used by both the GUI and CLI.

Configuration and paths:

- `config_store.py`: `AppSettings` schema plus JSON load/save support
- `paths.py`: repo-relative locations (config dir, models dir, setup marker) — no absolute paths

Extraction engine:

- `extractor.py`: core extraction pipeline, export pipeline, metadata generation, raster and vector handling

Model discovery, catalog, and downloads:

- `model_catalog.py`: single source of truth for recommended vision models (Ollama tag + Hugging Face GGUF specs)
- `model_registry.py`: local Ollama, llama.cpp, and directory model discovery
- `model_bootstrap.py`: on-demand model downloads (Ollama `/api/pull` streaming + Hugging Face GGUF/mmproj with resume)
- `first_run.py`: first-run detection and install orchestration shared by GUI and CLI

Vision providers (semantic naming):

- `vision.py`: shared helpers (prompt, response parsing, local-URL guard), retry/backoff, preflight capability check, and the provider client factory
- `ollama_client.py`: local-only Ollama vision client (native `/api/generate`)
- `llamacpp_client.py`: local-only llama.cpp vision client (OpenAI-compatible `/v1/chat/completions`)
- `llamacpp_server.py`: optional launcher for a local `llama-server` against downloaded GGUF weights

This folder is source code and acts as the application core.

## Tests

### `tests/`

`pytest` suite that runs fully offline (network and servers are mocked).

- `test_provider_migration.py`: shared vision helpers, factory selection, both clients, CLI validation
- `test_bootstrap_and_setup.py`: model catalog, retry/backoff, preflight, downloader, first-run flow, and cross-provider parity

Run with `pytest -q` from the repository root.

## Configuration and Runtime State

### `config/`

Stores persisted app state.

- `settings.json`: saved defaults (provider, model URLs/names, formats, canvas settings, detection values, last-used paths)
- `setup_state.json`: first-run completion marker (gitignored)

This folder contains runtime configuration, not source logic.

### `models/`

Downloaded llama.cpp GGUF weights and `mmproj` projectors, organized as
`models/llamacpp/<model-key>/`. Created by the first-run installer and **gitignored**.
Ollama models are managed by Ollama itself and do not live here.

### `logs/`

Runtime log output created by the GUI entrypoint (`shapearator_YYYYMMDD_HHMMSS.log`). Generated output.

## Documentation and Asset Folders

### `docs/`

Mixed working directory containing screenshots, source art, test sheets, and many generated example outputs:

- sample raster and vector icon sheets
- exported `png/`, `jpg/`, `tiff/`, `svg/` asset folders and per-icon `metadata/`
- screenshots such as `GUI.png` and the `docs/readme/` images used by `README.md`
- working/test collections (e.g. `SVG TEST`, `COLORS`, `TestMac`, `color test`)

Much of `docs/` is sample content, experiments, or generated extraction output — not the main source-code area.

### `documents/`

Reserved documentation working area; not part of the current runtime flow.

## Export Output Structure

A chosen output directory contains only the selected export formats plus metadata:

```text
output-dir/
  png/       # bitmap PNG exports
  jpg/       # bitmap JPG exports
  tiff/      # bitmap TIFF exports
  svg/       # vector-native or wrapped/traced SVG exports
  metadata/  # per-icon JSON metadata files
```

During extraction the engine may create intermediate `_work_png/` and `_work_svg/`
folders inside the output directory; it cleans them up when no longer needed.

## Source vs Generated Content

Source code or maintained docs:

- `main.py`, `shapearator.py`, `run.sh`, `requirements.txt`
- `gui/`, `services/`, `tests/`
- `README.md`, `MANUAL.md`, `FILE_STRUCTURE.md`, `LICENSE`

Runtime state, generated content, or working assets:

- `config/settings.json`, `config/setup_state.json`
- `logs/`, `models/`
- most of `docs/`
- export folders produced by extraction runs

## Maintenance Guidance

If you are changing application behavior, the most likely files to update are:

- `shapearator.py` — CLI behavior and `--setup` flow
- `gui/workspace_tab.py` — extraction UI behavior
- `gui/settings_tab.py` — provider/settings UI behavior
- `gui/setup_dialog.py` — first-run download UI
- `services/extractor.py` — core extraction/export logic
- `services/vision.py` — provider contract, preflight, retries
- `services/model_catalog.py` — recommended models and download specs
- `services/config_store.py` — persisted settings shape

If you are updating public project docs, keep `README.md`, `MANUAL.md`, and `FILE_STRUCTURE.md` aligned.
