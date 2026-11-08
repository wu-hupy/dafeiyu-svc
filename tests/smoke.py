"""最小冒烟测试：不依赖任何受版权保护的音频。

生成 3 秒"类人声"信号（谐波 + 缓慢 f0 滑动 + 轻微噪声），跑一遍换声，
检查：时长一致、无 NaN、峰值不爆、电平在合理区间、结果可复现（同 seed）。

用法:
    python tests/smoke.py --assets ./assets --weights ./weights/v6
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_voice_like(sr: int = 44100, dur: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0 = 220 * 2 ** (np.sin(2 * np.pi * 0.4 * t) * 0.15)  # 缓慢滑动
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(n, dtype=np.float64)
    for k in range(1, 12):
        y += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    env = np.minimum(1, np.minimum(t / 0.3, (dur - t) / 0.3)).clip(0)
    y = y * env
    y += 0.005 * rng.standard_normal(n) * env
    return (0.2 * y / np.abs(y).max()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--weights", default="weights/v6")
    ap.add_argument("--ddsp-repo", default="third_party/DDSP-SVC")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()

    from dafeiyu_svc import config as cfgmod
    from dafeiyu_svc.convert import convert, load_engine

    cfg = cfgmod.load()
    cfg.paths.assets = os.path.abspath(a.assets)
    cfg.paths.weights = os.path.abspath(a.weights)
    cfg.paths.ddsp_repo = os.path.abspath(a.ddsp_repo)
    cfg.infer.steps = a.steps

    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    engine = load_engine(cfg)

    sr = 44100
    y = make_voice_like(sr)
    outs = []
    for _ in range(2):
        torch.manual_seed(0)
        outs.append(convert(engine, y.copy(), sr, cfg, key=0.0))

    ok = True
    for i, o in enumerate(outs):
        d = len(o) - len(y)
        rms = 20 * np.log10(np.sqrt((o ** 2).mean()) + 1e-12)
        peak = float(np.abs(o).max())
        print(f"run{i}: len {len(o)} (src {len(y)}, diff {d})  rms {rms:.1f} dBFS  peak {peak:.2f}")
        ok &= np.isfinite(o).all() and abs(d) <= sr * 0.05 and peak <= 1.0 and -45 < rms < 0

    same = np.allclose(outs[0], outs[1], atol=1e-6)
    print("deterministic:", same)
    if not same:
        print("  ! 同 seed 结果不一致：检查 collect_rng / dropout / 采样实现")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, outs[0], sr)
        print("wrote", f.name)

    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
