# Shapearator Manual

This manual explains how to use Shapearator through both the desktop GUI and the CLI.

## 1. What Shapearator Is For

Shapearator extracts individual icons from a larger icon sheet and exports each icon as one or more files on a common canvas. It is designed for hand-drawn UI icon sheets, symbol collections, and mixed icon boards where icons are spatially separated enough to detect cleanly.

Typical use cases:

- splitting a sketch sheet into per-icon assets
- generating `PNG`, `JPG`, `TIFF`, and `SVG` exports in one run
- normalizing icon sizes to a shared canvas
- creating metadata for asset catalogs
- assigning semantic filenames with a local Ollama or llama.cpp vision model

## 2. Before You Start

Make sure the following are available:

- Python environment with the app dependencies installed
- `inkscape` on `PATH`
- `potrace` on `PATH`

If you want semantic naming, one of:

- a running local Ollama instance with a vision model selected in the app or CLI, or
- a running local llama.cpp `llama-server` (with a vision model and its `--mmproj`) and its URL/model selected

You do not have to set the model up by hand. On first launch with no local vision
model, the GUI shows a one-click setup dialog that downloads the recommended model
for whichever backend you have. From the CLI, run `python shapearator.py --setup`
(or `--setup-all`). Ollama models are pulled with `ollama pull`; llama.cpp models
are downloaded as GGUF weights plus their `mmproj` projector into `models/`, and
the app can start `llama-server` for you against them.

Supported input formats:

- `PNG`
- `SVG`

## 3. Quick Start

### GUI quick start

1. Launch the app with `./run.sh` or `python main.py`.
2. Open the `Workspace` tab.
3. Choose an input sheet.
4. Choose an output folder.
5. Select your export formats.
6. Set the output canvas size.
7. Choose a canvas behavior.
8. Click `Extract Icons`.

### CLI quick start

```bash
python shapearator.py sheet.png --output-dir exports
```

This runs extraction using default settings and writes results into `exports/`.

## 4. GUI Workflow

The GUI is split into two tabs:

- `Workspace`
- `Settings`

### Workspace tab

The `Workspace` tab is where extraction runs happen.

Source section:

- `Input Sheet`: path to the source `PNG` or `SVG`
- `Output Folder`: destination for generated assets
- provider summary line: shows the currently active provider mode

Detection section:

- `Padding`: adds space around each extracted icon crop
- `Min Area`: removes tiny detections such as dust or accidental marks
- `Merge`: reconnects nearby strokes into one icon group
- `Preset`: loads one of the built-in detection presets

Output Studio section:

- `Formats`: choose any combination of `PNG`, `JPG`, `TIFF`, and `SVG`
- `Width` and `Height`: define the final export canvas size
- `Bitmap Export`: choose how bitmap backgrounds should behave
- `Canvas Behavior`: choose how icons are scaled onto the common canvas

Run section:

- `Extract Icons`: starts the extraction pipeline
- status line: shows the current run state
- progress bar: shows extraction progress across phases

Preview section:

- shows a preview of the selected extracted icon
- reflects either a bitmap export or a preview generated from `SVG`

Extracted Items section:

- lists every exported icon
- shows formats, source size, and canvas size
- selecting an item updates the preview

### Settings tab

The `Settings` tab configures local provider behavior.

Provider options:

- `Geometry Only`: extraction only, no semantic naming
- `Ollama Local`: local Ollama provider for semantic naming
- `llama.cpp Local`: local llama.cpp (`llama-server`) provider for semantic naming
- `Model Directory`: local model catalog selection for future adapters

Ollama settings:

- `Ollama URL`: must be local-only
- `Ollama Model`: selected local model
- `Refresh Ollama Models`: rescans local models
- `Use Recommended`: picks the app’s preferred local vision model when available

llama.cpp settings:

- `llama.cpp URL`: must be local-only (default `http://127.0.0.1:8080`)
- `llama.cpp Model`: model reported by the running server (`/v1/models`)
- `Refresh llama.cpp Models`: queries the server for its loaded model
- `Use Recommended`: picks the app’s preferred local vision model when available

Directory settings:

- `Model Directory`: local folder used to discover models
- `Directory Model`: selected item from that folder

Semantic naming:

- `Use selected model for semantic file naming and metadata when supported`

## 5. CLI Workflow

The CLI uses the same shared extraction engine as the GUI.

Basic shape:

```bash
python shapearator.py INPUT --output-dir OUTPUT [options]
```

### Core options

- `--output-dir`
- `--formats`
- `--output-width`
- `--output-height`
- `--canvas-mode`
- `--bitmap-export-mode`
- `--padding`
- `--min-area`
- `--merge-gap`

### Provider options

- `--provider`
- `--ollama-url`
- `--ollama-model`
- `--llamacpp-url`
- `--llamacpp-model`
- `--local-model-root`
- `--setup` (download the default vision model for available backends, then exit)
- `--setup-all` (download every recommended vision model, then exit)
- `--local-model-name`
- `--semantic-naming`
- `--no-semantic-naming`

### Config options

- `--use-config`
- `--save-config`

### Detection preset option

- `--detection-preset`

Supported preset names:

- `Balanced`
- `Tiny Details`
- `Loose Sketches`
- `Bold Shapes`

Preset precedence:

1. If `--use-config` is set, load config values first.
2. If `--detection-preset` is set, apply the preset values next.
3. If explicit `--padding`, `--min-area`, or `--merge-gap` values are provided, they override both config and preset values.

### CLI examples

Geometry-only run:

```bash
python shapearator.py sheet.png \
  --output-dir exports \
  --provider geometry \
  --formats png svg
```

Run with config defaults:

```bash
python shapearator.py sheet.svg \
  --use-config \
  --output-dir exports
```

Run with preset and explicit override:

```bash
python shapearator.py sheet.png \
  --output-dir exports \
  --detection-preset "Loose Sketches" \
  --min-area 180
```

Run with local Ollama semantic naming:

```bash
python shapearator.py sheet.png \
  --output-dir exports \
  --provider ollama \
  --ollama-url http://127.0.0.1:11434 \
  --ollama-model qwen2.5vl:3b \
  --semantic-naming
```

Run with local llama.cpp semantic naming:

```bash
python shapearator.py sheet.png \
  --output-dir exports \
  --provider llamacpp \
  --llamacpp-url http://127.0.0.1:8080 \
  --llamacpp-model qwen2.5-vl \
  --semantic-naming
```

## 6. Understanding the Main Controls

### Canvas modes

`original`

- keeps the isolated icon at its extracted size
- best when you want raw relative differences preserved

`uniform_to_largest`

- scales the largest icon to fit the canvas
- applies the same scale to all icons
- best when the set should feel visually consistent

`individual_fit`

- scales each icon independently to fit the canvas
- best when every icon should use maximum available space

### Bitmap export modes

`keep_background`

- fills the bitmap canvas with the detected source background
- useful when the original sheet background matters

`transparent_preserve_interior`

- exports with transparency
- tries to preserve internal white or light details
- usually the best choice for reusable icon assets

### Detection parameters

`Padding`

- adds breathing room around the extracted symbol

`Min Area`

- filters out small detections
- lower values catch finer details but may also keep noise

`Merge Gap`

- reconnects nearby strokes into one symbol group
- higher values help with sketchy, broken marks

## 7. PNG vs SVG Input

### When to use PNG

Use `PNG` input when:

- the source is a scanned or exported bitmap sheet
- icons are clearly separated
- you want geometry-based local extraction

### When to use SVG

Use `SVG` input when:

- the sheet is already vector
- you want the cleanest possible `SVG` outputs
- you want grouped vector extraction rather than raster tracing

In general, `SVG` input gives the highest-quality downstream vector exports.

## 8. Semantic Naming

Semantic naming is optional and local-only.

How it works:

1. extraction runs normally
2. preview images are generated per icon
3. the selected local vision model (Ollama or llama.cpp) is asked to identify each icon
4. labels are converted into filesystem-safe slugs
5. exported files are renamed to the semantic label
6. metadata is updated with label, tags, and confidence

Important behavior:

- semantic naming runs when `provider=ollama` or `provider=llamacpp`
- the provider URL (Ollama or llama.cpp) must be local
- duplicate names are disambiguated with numeric suffixes
- if semantic identification fails, extraction still completes

## 9. Output Structure

A typical run may produce:

```text
exports/
  png/
  jpg/
  tiff/
  svg/
  metadata/
```

Only selected formats are created.

Each icon’s metadata JSON may include:

- label and tags
- confidence
- source bounds
- source size
- canvas size
- pipeline/provider information
- model used
- dominant color and palette
- vector mode
- exported file paths

## 10. Config File Behavior

The app stores settings in:

```text
config/settings.json
```

GUI behavior:

- loads automatically on startup
- saves automatically when settings change

CLI behavior:

- ignores config unless `--use-config` is passed
- saves config only when `--save-config` is passed

This makes CLI runs safe for automation without unexpectedly changing your defaults.

## 11. Troubleshooting

`Input file not found`

- confirm the path exists
- confirm the file extension is `.png` or `.svg`

`At least one export format must be selected`

- pass one or more values to `--formats`
- or use config values with `--use-config`

`Canvas width and height must be positive integers`

- set both dimensions above zero

`Ollama provider requires a local endpoint` / `llama.cpp provider requires a local endpoint`

- use `localhost`, `127.0.0.1`, or `::1`
- do not point to remote or cloud endpoints

llama.cpp model dropdown is empty:

- start `llama-server` with a vision model and its `--mmproj` projector
- confirm the URL matches the server port (default `8080`)
- click `Refresh llama.cpp Models`

Very small icons are not being detected:

- lower `Min Area`
- try `Tiny Details`

Icons are being merged incorrectly:

- lower `Merge Gap`
- increase `Min Area` if noise is being grouped

Single icons are being split into multiple pieces:

- increase `Merge Gap`
- try `Loose Sketches`

Vector output looks rasterized:

- start from `SVG` input when possible
- note that colored raster sources may be embedded in `SVG` to preserve fidelity

## 12. Recommended Starting Points

For clean monochrome icon sheets:

- preset: `Balanced`
- canvas mode: `uniform_to_largest`
- bitmap export mode: `transparent_preserve_interior`

For tiny or delicate symbols:

- preset: `Tiny Details`
- lower `min-area` if needed

For broken sketch lines:

- preset: `Loose Sketches`
- increase `merge-gap` if needed

For bold geometric shapes:

- preset: `Bold Shapes`
- keep `uniform_to_largest` unless every icon should fill the frame independently
