"""What the vision model is actually shown.

Found by running a real SVG sheet through the packaged Mac app: all 47 icons
came back named "heart", with naming reporting 47 named and 0 failed. The
exports were correct and all different; the labels were identical.

The cause is in the bitmap. An icon exported from an SVG source carries its
artwork entirely in the alpha channel -- every RGB pixel is (0, 0, 0), and the
shape exists only as opacity. Shown that file, anything that flattens or
ignores alpha sees one black square, so the model answers the same word for
every icon and reports high confidence doing it.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from services.config_store import AppSettings
from services.extraction_types import ExtractedIcon
from services.semantic_naming import apply_semantic_names


def _settings() -> AppSettings:
    return AppSettings(provider="ollama", semantic_naming=True, ollama_model="qwen2.5vl:3b")


def _alpha_only_png(path: Path) -> Path:
    """A bitmap exactly as the SVG pipeline writes it: black everywhere, with
    the drawing present only as opacity."""
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for x in range(8, 24):
        for y in range(8, 24):
            image.putpixel((x, y), (0, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _icon(tmp_path: Path) -> ExtractedIcon:
    preview = _alpha_only_png(tmp_path / "png" / "icon_001.png")
    return ExtractedIcon(
        index=1,
        stem="icon_001",
        outputs={"png": preview},
        preview_path=preview,
        canvas_size=(512, 512),
        source_size=(32, 32),
        source_bounds=(0, 0, 32, 32),
    )


def test_the_model_is_shown_a_picture_and_not_a_black_square(tmp_path):
    seen: dict[str, object] = {}

    def capture(_model, image_path: Path):
        # Read it here: a throwaway preview may not outlive the call.
        with Image.open(image_path) as shown:
            seen["extrema"] = shown.convert("RGB").getextrema()
        return {"label": "square"}

    client = MagicMock()
    client.identify_icon.side_effect = capture
    with patch("services.semantic_naming.build_vision_client", return_value=client):
        apply_semantic_names(_settings(), [_icon(tmp_path)])

    channel_ranges = seen["extrema"]
    assert channel_ranges is not None
    flat = all(low == high for low, high in channel_ranges)
    assert not flat, (
        "the model was handed a single flat colour "
        f"(RGB extrema {channel_ranges}); every icon looks identical to it"
    )


def test_the_original_export_is_left_alone(tmp_path):
    """Compositing is for the model's benefit. The user asked for transparent
    bitmaps and must still get them."""
    icon = _icon(tmp_path)
    exported = icon.outputs["png"]

    client = MagicMock()
    client.identify_icon.return_value = {"label": "square"}
    with patch("services.semantic_naming.build_vision_client", return_value=client):
        icons, _summary, _warnings = apply_semantic_names(_settings(), [icon])

    with Image.open(icons[0].outputs["png"]) as after:
        assert after.mode == "RGBA"
        assert after.getchannel("A").getextrema() == (0, 255), "transparency was flattened"
