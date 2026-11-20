"""Loads .env (+ config/settings.yaml later). All paths resolve inside the project."""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _str(key: str, default: str = "") -> str:
    # strip inline "# comment" that env.example uses
    return os.environ.get(key, default).split("#", 1)[0].strip()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_data_gb: float
    min_free_disk_gb: float
    keep_temp_files: bool
    ffmpeg: str
    ffprobe: str
    scorer_mode: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    transcribe_language: str
    chunk_seconds: int
    auto_render_top_n: int
    clip_min: int
    clip_max: int
    width: int
    height: int
    chat_activity: bool
    analytics: bool
    ollama_model: str
    ollama_host: str
    max_llm_candidates: int
    excerpt_max_words: int
    music_check: bool
    yaml: dict

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def _load_yaml() -> dict:
    p = ROOT / "config" / "settings.yaml"
    return yaml.safe_load(p.read_text()) or {} if p.exists() else {}


def load(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env", override=False)
    data_dir = Path(_str("DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()
    # Keep model caches inside the project unless the user overrides.
    hf = Path(_str("HF_HOME", str(data_dir / "models")))
    if not hf.is_absolute():
        hf = (ROOT / hf).resolve()
    os.environ["HF_HOME"] = str(hf)
    # yt-dlp's partial-download check ignores --ffmpeg-location (it only looks on PATH), so put the
    # configured ffmpeg's directory first on PATH for this process.
    fb = _str("FFMPEG_BIN", "ffmpeg")
    if os.path.isabs(fb):
        d = str(Path(fb).parent)
        if d not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    s = Settings(
        data_dir=data_dir,
        max_data_gb=float(_str("MAX_DATA_GB", "50")),
        min_free_disk_gb=float(_str("MIN_FREE_DISK_GB", "10")),
        keep_temp_files=_bool(_str("KEEP_TEMP_FILES") or None, False),
        ffmpeg=_str("FFMPEG_BIN", "ffmpeg"),
        ffprobe=_str("FFPROBE_BIN", "ffprobe"),
        scorer_mode=_str("SCORER_MODE", "heuristic"),
        whisper_model=_str("WHISPER_MODEL", "small"),
        whisper_device=_str("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", "int8"),
        transcribe_language=_str("TRANSCRIBE_LANGUAGE", "auto"),
        chunk_seconds=int(_str("TRANSCRIBE_CHUNK_SECONDS", "600")),
        auto_render_top_n=int(_str("AUTO_RENDER_TOP_N", "3")),
        clip_min=int(_str("CLIP_MIN_SECONDS", "15")),
        clip_max=int(_str("CLIP_MAX_SECONDS", "59")),
        width=int(_str("OUTPUT_WIDTH", "1080")),
        height=int(_str("OUTPUT_HEIGHT", "1920")),
        chat_activity=_bool(_str("CHAT_ACTIVITY_ENABLED") or None, False),
        analytics=_bool(_str("ANALYTICS_ENABLED") or None, False),
        ollama_model=_str("OLLAMA_MODEL"),
        ollama_host=_str("OLLAMA_HOST", "http://localhost:11434"),
        max_llm_candidates=int(_str("MAX_LLM_CANDIDATES", "12")),
        excerpt_max_words=int(_str("EXCERPT_MAX_WORDS", "120")),
        music_check=_bool(_str("MUSIC_CHECK_ENABLED") or None, True),
        yaml=_load_yaml(),
    )
    if s.scorer_mode not in ("heuristic", "local", "hybrid"):
        raise ValueError(f"SCORER_MODE must be heuristic|local|hybrid, got {s.scorer_mode!r}")
    if s.max_data_gb <= 0:
        raise ValueError("MAX_DATA_GB must be > 0")
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
