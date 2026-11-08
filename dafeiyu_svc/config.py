"""路径与推理参数的统一入口（配置文件 → 环境变量覆盖）。"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


@dataclass
class Paths:
    ddsp_repo: Path = ROOT / "third_party" / "DDSP-SVC"
    assets: Path = ROOT / "assets"
    weights: Path = ROOT / "weights" / "v6"


@dataclass
class Infer:
    steps: int = 20
    t_start: float = 0.7
    method: str = "euler"
    key: float = 0.0
    smooth: int = 0
    threhold: float = -60.0
    f0_min: float = 60.0
    f0_max: float = 800.0
    device: str = ""


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    infer: Infer = field(default_factory=Infer)


def load(config_path: str | os.PathLike | None = None) -> Config:
    p = Path(config_path) if config_path else ROOT / "config.toml"
    raw = _load_toml(p)
    cfg = Config()

    for key in ("ddsp_repo", "assets", "weights"):
        v = raw.get("paths", {}).get(key)
        if v:
            setattr(cfg.paths, key, Path(v).expanduser())

    env_map = {
        "DAFEIYU_DDSP_REPO": "ddsp_repo",
        "DAFEIYU_ASSETS": "assets",
        "DAFEIYU_WEIGHTS": "weights",
    }
    for env, key in env_map.items():
        if os.environ.get(env):
            setattr(cfg.paths, key, Path(os.environ[env]).expanduser())

    for key, val in raw.get("infer", {}).items():
        if hasattr(cfg.infer, key):
            setattr(cfg.infer, key, val)

    cfg.paths.ddsp_repo = cfg.paths.ddsp_repo.resolve()
    cfg.paths.assets = cfg.paths.assets.resolve()
    cfg.paths.weights = cfg.paths.weights.resolve()
    return cfg
