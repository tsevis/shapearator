# File Structure

What lives where, and which files are source rather than generated output.
`README.md` covers behaviour; this document covers layout.

## Top level

```text
shapearator/
  shapearator.py     # CLI entry point
  main.py            # GUI entry point
  run.sh             # GUI launcher; installs dependencies on first run
  services/          # the engine
  gui/               # tkinter/ttk desktop interface
  tests/             # pytest suite
  config/            # persisted settings and the first-run marker (runtime state)
  models/            # downloaded llama.cpp GGUF weights (gitignored)
  docs/              # committed sample sheets and screenshots (see below)
  logs/              # generated run logs
  requirements.txt   # runtime dependencies
  pyproject.toml     # pytest configuration
  README.md  MANUAL.md  FILE_STRUCTURE.md  LICENSE
```

## Entry points

### `shapearator.py`

The CLI. Parses arguments, resolves settings (defaults, then config with
`--use-config`, then a detection preset, then explicit flags), validates the
result, runs the extractor, and reports progress and completion. Also hosts
`--setup` / `--setup-all` for headless model downloads.

### `main.py`

The GUI. Initializes logging, creates the Tk root, and starts the desktop
shell, which triggers first-run model setup when needed.

### `run.sh`

Launches the GUI, installing the Python dependencies the first time they cannot
be imported.

## The engine — `services/`

Shared by the GUI, the CLI, and [MacShapearator](https://github.com/tsevis/MacShapearator).
No module here imports from `gui/`.

### Settings

| Module | Role |
| --- | --- |
| `settings_schema.py` | the `AppSettings` dataclass, the allowed-value catalog, and per-field coercion — the single source of truth the CLI's argparse choices are built from |
| `config_store.py` | JSON load/save on top of that schema; unknown keys and invalid values degrade per field instead of raising |
| `paths.py` | repo-relative locations (config dir, models dir, setup marker) |

### Extraction

| Module | Role |
| --- | --- |
| `extractor.py` | orchestration: `IconExtractor`, the run pipeline, metadata generation |
| `extraction_types.py` | `ExtractedIcon`, `NamingSummary`, `ExtractionResult`, `ExtractionProgress` — the data handed to callers |
| `geometry.py` | `Box`, foreground masks, blob detection, reading-order sorting, uniform scale |
| `raster_ops.py` | canvas composition, transparency (enclosed holes preserved), palette and monochrome analysis |
| `svg_ops.py` | SVG parsing, icon-level discovery, transitive definition resolution, fragment building, canvas normalization, and the Inkscape/potrace calls |
| `export_commit.py` | staged exports and the run manifest: publishes a run only once every step succeeds, and replaces only the files the manifest lists |
| `metadata_paths.py` | reduces paths and error text bound for exported files to a portable, non-identifying form |

### Naming and models

| Module | Role |
| --- | --- |
| `semantic_naming.py` | the pre-export backend gate, per-icon labeling and renaming with rollback, and the naming summary |
| `vision.py` | shared prompt and response parsing, the local-URL guard, retry/backoff, `preflight`, and the provider client factory |
| `ollama_client.py` | local-only Ollama client (native `/api/generate`) |
| `llamacpp_client.py` | local-only llama.cpp client (OpenAI-compatible `/v1/chat/completions`) |
| `llamacpp_server.py` | optional launcher for a local `llama-server` |
| `llamacpp_models.py` | discovers startable llama.cpp vision models, downloaded or already in llama.cpp's cache |
| `model_catalog.py` | the recommended vision models: Ollama tags and Hugging Face GGUF specs |
| `model_registry.py` | discovery across Ollama, llama.cpp and a plain directory |
| `model_bootstrap.py` | downloads models with progress and resume |
| `first_run.py` | first-run detection and install orchestration, shared by GUI and CLI |

## GUI — `gui/`

| Module | Role |
| --- | --- |
| `main_window.py` | app shell, tabs, appearance toggle, settings persistence |
| `workspace_tab.py` | source and output pickers, detection and export controls, run, preview, results |
| `settings_tab.py` | provider, endpoints, models, and runtime status |
| `setup_dialog.py` | first-run model download UI |
| `theme_utils.py` | light/dark helpers |

## Tests — `tests/`

Fully offline: network calls and subprocesses are mocked, so neither a model
backend nor Inkscape is required. Run with `pytest -q`.

The suite also opens no windows. The `gui/` tests cover module-level tables and
static methods, and stand in for widgets with duck-typed objects, so nothing
constructs a `Tk` root. A test that ever does need a real window must carry the
`gui` marker registered in `pyproject.toml`; `addopts` excludes that marker, so
a plain run stays silent.

| File | Covers |
| --- | --- |
| `test_settings_schema.py` | per-field coercion, unknown/malformed config recovery, CLI/schema parity |
| `test_geometry.py` | box arithmetic, masks, blob detection, reading order |
| `test_raster_ops.py` | canvas composition and clipping, interior-hole transparency, palette analysis |
| `test_svg_ops.py` | viewBox parsing, id assignment, fragment building, normalization, metadata injection |
| `test_svg_grouping.py` | icon-level discovery, grouped vs loose artwork, ancestor transforms |
| `test_svg_definitions.py` | reference scanning, transitive definition resolution, stylesheet retention |
| `test_semantic_naming.py` | preflight enforcement and downgrade, per-icon status, rename rollback, truthful metadata |
| `test_svg_only_naming.py` | naming SVG-only exports via a temporary preview |
| `test_export_commit.py` | stale-output replacement, preservation of untracked files, manifest contents, rollback |
| `test_metadata_privacy.py` | exported metadata discloses no local filesystem paths |
| `test_provider_migration.py` | shared vision helpers, factory selection, both clients, CLI validation |
| `test_bootstrap_and_setup.py` | catalog, retry/backoff, preflight, downloader, first-run flow |
| `test_gui_labels.py` | GUI label maps against the schema, and its preset table against the CLI's |
| `test_gui_settings_tab.py` | the model-recommendation text: selected vs recommended, and missing either |
| `test_gui_theme.py` | master-chain walking, nearest themable host, cycles, and failing hosts |

## Runtime state

| Path | Contents |
| --- | --- |
| `config/settings.json` | persisted settings (gitignored) |
| `config/setup_state.json` | first-run completion marker (gitignored) |
| `models/` | downloaded GGUF weights and `mmproj` projectors (gitignored) |
| `logs/` | run logs (gitignored) |

## Export output

A chosen output directory holds only the selected formats plus metadata:

```text
output-dir/
  png/  jpg/  tiff/  svg/
  metadata/                       # per-icon JSON
  .shapearator-manifest.json      # what the last run wrote; the only files a later run replaces
```

A run is built inside `.shapearator-staging/` in the output folder and moved
into place only once every step succeeds, so a failure leaves the previous
export intact. Intermediate `_work_png/` and `_work_svg/` folders live in
staging and are discarded with it — they never reach the output directory.

Only the directories above are ever written or removed. Files you place in the
output folder are not tracked by the manifest and are never deleted.

## Source vs generated

**Source, maintained by hand:** `shapearator.py`, `main.py`, `run.sh`,
`services/`, `gui/`, `tests/`, `requirements.txt`, `pyproject.toml`, and the
markdown documents.

**Sample assets:** `docs/base.png`, `docs/base.svg`, `docs/base.ai`, and
`docs/readme/` screenshots. These four entries are the whole of the committed
`docs/`.

**Generated, safe to delete:** `config/settings.json`, `config/setup_state.json`,
`models/`, `logs/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
`.coverage`, and any export folder.

`docs/` doubles as a scratch area for local test runs, and those runs are large
— an export of a full sheet is tens of megabytes. `.gitignore` therefore ignores
`docs/` as a whole and re-admits only the four sample assets above, so a stray
`git add .` cannot commit a run. A genuinely new doc asset needs its own
negation line in `.gitignore`, or `git add -f`.

## Where to make a change

| To change | Edit |
| --- | --- |
| CLI behaviour or flags | `shapearator.py` |
| extraction pipeline | `services/extractor.py` |
| how icons are found in an SVG | `services/svg_ops.py` |
| detection on raster sheets | `services/geometry.py` |
| how a run replaces a previous export | `services/export_commit.py` |
| naming behaviour or preflight | `services/semantic_naming.py`, `services/vision.py` |
| recommended models or downloads | `services/model_catalog.py`, `services/model_bootstrap.py` |
| the settings shape or its validation | `services/settings_schema.py` |
| extraction UI | `gui/workspace_tab.py` |
| provider/settings UI | `gui/settings_tab.py` |

Keep `README.md`, `MANUAL.md` and this file aligned when public behaviour
changes.
