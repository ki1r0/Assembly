from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = Path(os.environ.get("ASSEMBLY_ASSET_ROOT", str(PROJECT_ROOT / "assets"))).expanduser()
TMP_DIR = PROJECT_ROOT / "tmp"


def _parse_env_value(raw: str) -> str:
    value = str(raw or "").strip().strip('"\'').strip()
    if value.startswith("${") and value.endswith("}") and ":-" in value:
        body = value[2:-1]
        _, default = body.split(":-", 1)
        return default.strip().strip('"\'').strip()
    return value


def _parse_env_file(path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    if not path.is_file():
        return parsed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("export "):
                text = text[len("export ") :].strip()
            if "=" not in text:
                continue
            key, raw_value = text.split("=", 1)
            key = key.strip()
            if not key:
                continue
            parsed[key] = _parse_env_value(raw_value)
    return parsed


def runtime_env_candidates() -> tuple[Path, ...]:
    return (
        CONFIG_DIR / "runtime_env.env",
        CONFIG_DIR / "runtime_env.sh",
        CONFIG_DIR / "runtime_env.local.env",
        CONFIG_DIR / "runtime_env.local.sh",
    )


def load_runtime_env_defaults() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for candidate in runtime_env_candidates():
        try:
            merged.update(_parse_env_file(candidate))
        except Exception:
            continue
    for key, value in merged.items():
        os.environ.setdefault(key, value)
    return merged


def find_isaacsim_root() -> Path | None:
    candidates = [
        os.getenv("ISAACSIM_ROOT", "").strip(),
        os.getenv("ISAAC_SIM_ROOT", "").strip(),
        os.getenv("ISAACLAB_ROOT", "").strip(),
        "/isaac-sim",
        str(PROJECT_ROOT.parent / "isaacsim"),
        str(PROJECT_ROOT.parent.parent / "isaacsim"),
        "/opt/nvidia/isaacsim",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def asset_browser_cache_dir() -> Path:
    configured = os.getenv("ISAACSIM_ASSET_BROWSER_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache).expanduser() if xdg_cache else (Path.home() / ".cache")
    return base / "isaacsim.asset.browser"


def _expand_globs(patterns: Iterable[str]) -> Iterator[Path]:
    import glob

    for pattern in patterns:
        if not pattern:
            continue
        for match in glob.glob(pattern):
            path = Path(match)
            if path.exists():
                yield path


def usd_python_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = [
        os.getenv("USD_PYTHONPATH", "").strip(),
        os.getenv("ISAACSIM_USD_PYTHONPATH", "").strip(),
    ]
    for item in explicit:
        if item:
            path = Path(item).expanduser()
            if path.exists():
                paths.append(path)

    isaacsim_root = find_isaacsim_root()
    if isaacsim_root is not None:
        for rel in (
            "kit/python/lib/python3.11/site-packages",
            "_build/linux-x86_64/release/exts/omni.usd.libs/bin",
        ):
            path = isaacsim_root / rel
            if path.exists():
                paths.append(path)

    home = Path.home()
    packman_globs = [
        str(home / ".cache/packman/chk/usd.py*/**/lib/python"),
        str(home / ".cache/packman/chk/python/*/lib/python3.11/site-packages"),
    ]
    for path in _expand_globs(packman_globs):
        if path not in paths:
            paths.append(path)
    return paths
