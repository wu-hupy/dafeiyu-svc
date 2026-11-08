"""一条命令把 v6 跑起来所需的一切备齐并校验。

它做的事：
  1. 底座（DDSP-SVC 代码 + rmvpe + ContentVec-HF + NSF-HiFiGAN 2024.02）
  2. v6 权重（Releases / ModelScope，67.8 MB）
  3. 逐个校验字节数与 sha256，最后打印一张"还差什么"的表

用法：
    python -m dafeiyu_svc.bundle --root .            # 全部装到当前目录
    python -m dafeiyu_svc.bundle --root D:/svc --skip-code
    python -m dafeiyu_svc.bundle --root . --check-only   # 只体检，不下载

产物布局（与 config.example.toml 的默认值一致）：
    <root>/assets/pretrain/{rmvpe,contentvec_hf,nsf_hifigan}/...
    <root>/third_party/DDSP-SVC/...
    <root>/weights/v6/{model_24000.pt,config.yaml}
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import assets as base_assets

UA = {"User-Agent": "Mozilla/5.0"}

WEIGHTS_REPO = "wuhupy/dafeiyu-svc-v6"
MS = "https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={name}"
GH = "https://github.com/wu-hupy/dafeiyu-svc/releases/download/v0.1.0/dafeiyu-svc-v6-weights.zip"

# name -> (size, sha256 or None)  —— rmvpe 是下载包的大小，落盘后会被重打包成 {'model': sd}
EXPECT = {
    "rmvpe": (181_184_272, None),
    "contentvec": (188_767_088, None),
    "nsf_hifigan": (52_814_405, None),
    "weights": (67_828_627, "f672ed1fcf19021ff5afc213d35d0a3ec31fd5eb2b413efac48756558ba0bea5"),
    "weights_config": (1_045, None),
}

WEIGHT_FILES = ("model_24000.pt", "config.yaml")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path, expect_size: int | None, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"  = 已有 {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = tmp.stat().st_size
    if expect_size and got != expect_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"大小不符：{dst.name} 期望 {expect_size} 实得 {got}")
    tmp.replace(dst)
    print(f"  ✓ {dst} ({got / 1e6:.1f} MB)")


def install_weights(root: Path, force: bool) -> Path:
    out = root / "weights" / "v6"
    out.mkdir(parents=True, exist_ok=True)
    for name in WEIGHT_FILES:
        exp = EXPECT["weights"] if name == "model_24000.pt" else EXPECT["weights_config"]
        fetch(MS.format(repo=WEIGHTS_REPO, name=name), out / name, exp[0], force)
    ckpt = out / "model_24000.pt"
    if ckpt.exists():
        got = sha256_of(ckpt)
        want = EXPECT["weights"][1]
        if want and got != want:
            raise SystemExit(f"权重 sha256 不符：期望 {want} 实得 {got}")
        print(f"  ✓ sha256 {got[:16]}…")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="备齐 dafeiyu-svc v6 所需底座与权重")
    ap.add_argument("--root", default=".", help="安装根目录（默认当前目录）")
    ap.add_argument("--skip-code", action="store_true", help="已有 DDSP-SVC 代码时跳过")
    ap.add_argument("--force", action="store_true", help="重下已存在的文件")
    ap.add_argument("--check-only", action="store_true", help="只体检，不下载")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[root] {root}")

    if a.check_only:
        return report(root)

    print("\n[1/2] 底座（DDSP-SVC 代码 + rmvpe + ContentVec-HF + NSF-HiFiGAN）")
    argv2 = ["--root", str(root / "assets"), "--code-root", str(root / "third_party")]
    if a.skip_code:
        argv2.append("--skip-code")
    if a.force:
        argv2.append("--force")
    base_assets.main(argv2)

    print("\n[2/2] v6 权重")
    install_weights(root, a.force)

    print()
    return report(root)


def report(root: Path) -> int:
    pre = root / "assets" / "pretrain"
    checks = [
        # rmvpe 落盘后是重打包过的 {'model': sd}，字节数与下载包不同，只查存在与量级
        ("rmvpe", pre / "rmvpe" / "model.pt", None),
        ("contentvec", pre / "contentvec_hf" / "model.safetensors", EXPECT["contentvec"][0]),
        ("nsf_hifigan", pre / "nsf_hifigan" / "model.ckpt", None),
        ("DDSP-SVC 代码", root / "third_party" / "DDSP-SVC" / "ddsp" / "vocoder.py", None),
        ("v6 权重", root / "weights" / "v6" / "model_24000.pt", EXPECT["weights"][0]),
        ("v6 config", root / "weights" / "v6" / "config.yaml", None),
    ]
    print("体检：")
    missing = 0
    for name, path, size in checks:
        if path.exists() and path.stat().st_size > 0:
            if size and path.stat().st_size != size:
                print(f"  ✗ {name}: 大小 {path.stat().st_size} 与期望 {size} 不符")
                missing += 1
            else:
                print(f"  ✓ {name}: {path.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"  ✗ {name}: 缺失 {path}")
            missing += 1

    if missing:
        print(f"\n还差 {missing} 项。去掉 --check-only 重跑本命令即可补齐。")
        return 1

    print("\n全部就绪。下一步：")
    print("  cp config.example.toml config.toml")
    print("  python -m dafeiyu_svc.convert --input dry.wav --output out.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
