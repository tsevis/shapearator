from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppSettings:
    appearance: str = "light"
    provider: str = "geometry"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5vl:3b"
    llamacpp_url: str = "http://127.0.0.1:8080"
    llamacpp_model: str = ""
    models_root: str = ""  # where downloaded llama.cpp GGUF weights live; blank = <repo>/models
    local_model_root: str = ""  # optional directory-provider catalog root
    local_model_name: str = ""
    semantic_naming: bool = False
    default_formats: list[str] = field(default_factory=lambda: ["png", "svg"])
    output_width: int = 512
    output_height: int = 512
    canvas_mode: str = "uniform_to_largest"
    bitmap_export_mode: str = "transparent_preserve_interior"
    padding: int = 12
    min_area: int = 200
    merge_gap: int = 13
    last_input_path: str = ""
    last_output_dir: str = ""


class ConfigStore:
    def __init__(self, config_path: Path):
        self.config_path = config_path

    def load(self) -> AppSettings:
        if not self.config_path.exists():
            return AppSettings()
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return AppSettings()
        base = asdict(AppSettings())
        base.update(data)
        return AppSettings(**base)

    def save(self, settings: AppSettings) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(asdict(settings), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
