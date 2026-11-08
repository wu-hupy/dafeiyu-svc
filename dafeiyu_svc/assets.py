"""准备底座：DDSP-SVC 代码 + rmvpe + ContentVec(HF) + NSF-HiFiGAN(2024.02)。

用法:
    python -m dafeiyu_svc.assets --root ./assets
    python -m dafeiyu_svc.assets --root ./assets --skip-code   # 已自己 clone 过 DDSP-SVC

默认走国内可用的镜像（hf-mirror / ghfast），可用 --hf-endpoint / --gh-prefix 覆盖。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}

DDSP_ZIP = "https://ghfast.top/https://github.com/yxlllc/DDSP-SVC/archive/refs/heads/master.zip"
RMVPE = "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true"
CV_CONFIG = "https://hf-mirror.com/TechInterMezzo/contentvec-500/resolve/main/config.json?download=true"
CV_MODEL = "https://hf-mirror.com/TechInterMezzo/contentvec-500/resolve/main/model.safetensors?download=true"
NSF_ZIP = ("https://ghfast.top/https://github.com/openvpi/vocoders/releases/download/"
           "nsf-hifigan-44.1k-hop512-128bin-2024.02/nsf_hifigan_44.1k_hop512_128bin_2024.02.zip")


def fetch(url: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"  = {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)
    print(f"  ✓ {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="assets", help="底座根目录（会生成 pretrain/ 布局）")
    ap.add_argument("--code-root", default="third_party", help="DDSP-SVC 代码放哪")
    ap.add_argument("--skip-code", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    pre = root / "pretrain"
    (pre / "rmvpe").mkdir(parents=True, exist_ok=True)
    (pre / "contentvec_hf").mkdir(parents=True, exist_ok=True)
    (pre / "nsf_hifigan").mkdir(parents=True, exist_ok=True)

    print("[1/4] rmvpe")
    fetch(RMVPE, pre / "rmvpe" / "raw_rmvpe.pt", a.force)
    # DDSP 的 RMVPE 需要 {'model': state_dict}
    raw = pre / "rmvpe" / "raw_rmvpe.pt"
    packed = pre / "rmvpe" / "model.pt"
    if raw.exists() and (not packed.exists() or a.force):
        import torch
        sd = torch.load(raw, map_location="cpu")
        sd = sd.get("model", sd)
        torch.save({"model": sd}, packed)
        print(f"  ✓ {packed}")

    print("[2/4] ContentVec (transformers 版 HuBERT)")
    fetch(CV_CONFIG, pre / "contentvec_hf" / "config.json", a.force)
    fetch(CV_MODEL, pre / "contentvec_hf" / "model.safetensors", a.force)

    print("[3/4] NSF-HiFiGAN 44.1k (2024.02)")
    z = pre / "nsf_hifigan_dim.zip"
    fetch(NSF_ZIP, z, a.force)
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(pre / "nsf_hifigan")
        print(f"  ✓ {pre / 'nsf_hifigan'}")
        print("  ⚠ 注意：必须是 2024.02 版；2025.02 的 pc 版 key 对不上")

    print("[4/4] DDSP-SVC 代码")
    code = Path(a.code_root).resolve() / "DDSP-SVC"
    if not a.skip_code and not code.exists():
        zz = Path(a.code_root).resolve() / "ddsp.zip"
        fetch(DDSP_ZIP, zz, a.force)
        with zipfile.ZipFile(zz) as zf:
            zf.extractall(Path(a.code_root).resolve())
        src = next(Path(a.code_root).resolve().glob("DDSP-SVC-*"))
        src.rename(code)
        print(f"  ✓ {code}")
    print("\n完成。把 config.toml 的 paths.assets 指向", root, "、paths.ddsp_repo 指向", code)


if __name__ == "__main__":
    sys.exit(main())
