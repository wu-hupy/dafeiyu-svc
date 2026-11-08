"""把训练输出打成一个可直接发布的权重目录。

用法:
    python tools/export_weights.py \
        --ckpt  D:/path/exp/dafeiyu_v6/model_24000.pt \
        --config D:/path/exp/dafeiyu_v6/config.yaml \
        --out release/weights/v6 \
        --step 24000

产出:
    model_24000.pt        （原样复制，附 sha256）
    config.yaml           （把 expdir/train_path 去掉、路径改成 pretrain/ 相对布局）
    SHA256SUMS
    LICENSE-WEIGHTS.md    （从仓库根复制）
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml

DROP_KEYS = ("expdir", "train_path", "valid_path", "workdir")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--name", default=None, help="输出文件名，默认按 step 命名")
    a = ap.parse_args(argv)

    ckpt, cfgp = Path(a.ckpt), Path(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    step = a.step
    if step is None:
        digits = "".join(ch for ch in ckpt.stem if ch.isdigit())
        step = int(digits) if digits else 0
    name = a.name or f"model_{step}.pt"
    dst = out / name
    shutil.copyfile(ckpt, dst)

    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
    for key in ("expdir", "gpu_id"):
        if isinstance(cfg.get("env"), dict) and key in cfg["env"]:
            cfg["env"][key] = 0 if key == "gpu_id" else ""
    for key in ("train_path", "valid_path"):
        if isinstance(cfg.get("data"), dict) and key in cfg["data"]:
            cfg["data"][key] = ""
    for k in DROP_KEYS:
        if k in cfg:
            cfg[k] = ""
    cfg.setdefault("data", {})["encoder"] = "contentvec768l12_hf"
    cfg["data"]["encoder_ckpt"] = "pretrain/contentvec_hf"
    cfg.setdefault("vocoder", {})["type"] = "nsf-hifigan"
    cfg["vocoder"]["ckpt"] = "pretrain/nsf_hifigan/model.ckpt"
    (out / "config.yaml").write_text(
        "# 由 tools/export_weights.py 生成；相对路径以 assets/ 为根\n"
        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    lic = Path(__file__).resolve().parent.parent / "LICENSE-WEIGHTS.md"
    if lic.exists():
        shutil.copyfile(lic, out / "LICENSE-WEIGHTS.md")

    digest = sha256(dst)
    (out / "SHA256SUMS").write_text(f"{digest}  {name}\n", encoding="utf-8")
    print(f"✓ {out}")
    print(f"  {name}  {dst.stat().st_size / 1e6:.1f} MB  sha256={digest}")


if __name__ == "__main__":
    main()
