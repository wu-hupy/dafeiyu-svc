"""换声推理：wav → wav。

用法:
    python -m dafeiyu_svc.convert --input dry.wav --output out.wav [--weights DIR] [--assets DIR] [--steps 20]

要点（都是踩过的坑，别改成"看起来更规范"的写法）:
  * DDSP 的 rmvpe / contentvec / 声码器都用相对 `pretrain/...` 路径，因此必须在
    `assets/` 目录里执行；调用前把用户路径全部绝对化。
  * checkpoint 必须是 NSF-HiFiGAN **2024.02**（2025.02 的 pc 版 key 对不上）。
  * 切片起点落在字头上时，声码器冷启动会"砰"一下 → 段首 10 ms 升余弦。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from pathlib import Path


def _yaml_utf8_guard() -> None:
    """DDSP 用 `open(config.yaml)`（locale 编码）读配置，中文注释在 GBK 机器上会炸；这里强制 UTF-8。"""
    import io

    import yaml

    orig = yaml.safe_load

    def patched(stream, *args, **kwargs):
        name = getattr(stream, "name", None)
        if name and hasattr(stream, "read"):
            try:
                with io.open(name, "r", encoding="utf-8") as f:
                    return orig(f, *args, **kwargs)
            except (OSError, UnicodeDecodeError):
                pass
        return orig(stream, *args, **kwargs)

    yaml.safe_load = patched


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def load_engine(cfg):
    """把 DDSP 仓库塞进 sys.path，切到 assets 目录，加载模型与声码器。"""
    cfg.paths.weights = Path(cfg.paths.weights).expanduser().resolve()
    cfg.paths.assets = Path(cfg.paths.assets).expanduser().resolve()
    cfg.paths.ddsp_repo = Path(cfg.paths.ddsp_repo).expanduser().resolve()
    _yaml_utf8_guard()
    repo = str(cfg.paths.ddsp_repo)
    if not os.path.isdir(repo):
        raise SystemExit(f"DDSP-SVC 代码目录不存在: {repo}\n先跑 `python -m dafeiyu_svc.assets`")
    sys.path.insert(0, repo)
    os.chdir(str(cfg.paths.assets))

    from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder  # noqa: E402
    from ddsp.core import upsample  # noqa: E402
    from reflow.vocoder import load_model_vocoder  # noqa: E402
    from main_reflow import split, cross_fade  # noqa: E402

    ckpt = str(cfg.paths.weights / "model_24000.pt")
    if not os.path.exists(ckpt):
        cands = sorted(cfg.paths.weights.glob("model_*.pt"))
        if not cands:
            raise SystemExit(f"没找到权重: {cfg.paths.weights}")
        ckpt = str(cands[-1])

    device = cfg.infer.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocoder, args = load_model_vocoder(ckpt, device=device)
    return dict(model=model, vocoder=vocoder, args=args, device=device,
                F0_Extractor=F0_Extractor, Volume_Extractor=Volume_Extractor,
                Units_Encoder=Units_Encoder, upsample=upsample,
                split=split, cross_fade=cross_fade)


def convert(engine, wav: np.ndarray, sr: int, cfg, key: float = 0.0) -> np.ndarray:
    import librosa
    from scipy.signal import resample as _resample

    device = engine["device"]
    args = engine["args"]
    model, vocoder = engine["model"], engine["vocoder"]

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)

    hop_size = args.data.block_size * sr / args.data.sampling_rate
    f0 = engine["F0_Extractor"]("rmvpe", sr, hop_size, cfg.infer.f0_min, cfg.infer.f0_max).extract(
        wav, uv_interp=True, device=device)
    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)
    f0 = f0 * 2 ** (float(key) / 12)

    volume = engine["Volume_Extractor"](hop_size).extract(wav)
    mask = (volume > 10 ** (cfg.infer.threhold / 20)).astype("float")
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n:n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = engine["upsample"](mask, args.data.block_size).squeeze(-1)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

    encoder = engine["Units_Encoder"](
        args.data.encoder, args.data.encoder_ckpt,
        args.data.encoder_sample_rate, args.data.encoder_hop_size,
        cnhubertsoft_gate=10, device=device)

    spk_id = torch.LongTensor(np.array([[0]])).to(device)
    aug_shift = torch.from_numpy(np.array([[0.0]])).float().to(device)

    total = len(wav)
    out = np.zeros(total, dtype=np.float32)
    segments = engine["split"](wav, sr, hop_size)

    with torch.no_grad():
        for start_frame, seg_audio in segments:
            seg = torch.from_numpy(seg_audio).float().unsqueeze(0).to(device)
            units = encoder.encode(seg, sr, hop_size)
            seg_f0 = f0[:, start_frame:start_frame + units.size(1), :]
            seg_vol = volume[:, start_frame:start_frame + units.size(1), :]

            ddsp_wav, _, _ = model.ddsp_model(units, seg_f0, seg_vol,
                                              spk_id=spk_id, aug_shift=aug_shift, infer=True)
            mel = vocoder.extract(ddsp_wav)
            mel = model.reflow_model(mel, gt_spec=mel, infer=True,
                                     infer_step=cfg.infer.steps, method=cfg.infer.method,
                                     t_start=cfg.infer.t_start, use_tqdm=False)
            seg_wav = vocoder.infer(mel, seg_f0[:, -mel.shape[1]:])
            seg_wav = np.asarray(seg_wav.squeeze().cpu().numpy(), dtype=np.float32).reshape(-1)

            m = np.asarray(mask[:, start_frame * args.data.block_size:
                                (start_frame + units.size(1)) * args.data.block_size]
                           .squeeze().cpu().numpy(), dtype=np.float32).reshape(-1)
            n = min(len(seg_wav), len(m))
            seg_wav = seg_wav[:n] * m[:n]

            s0 = int(round(start_frame * hop_size))
            target = min(len(seg_audio), total - s0)
            if target <= 0:
                continue
            if len(seg_wav) != target:
                seg_wav = _resample(seg_wav, target)
            seg_wav = np.asarray(seg_wav, dtype=np.float32)[:target]

            fade = min(int(0.010 * sr), target // 2)
            if fade > 1:
                k = np.linspace(0, 1, fade, dtype=np.float32)
                seg_wav[:fade] *= 0.5 - 0.5 * np.cos(np.pi * k)
            tail = min(256, target // 2)
            if tail > 1:
                seg_wav[-tail:] *= np.linspace(1, 0, tail, dtype=np.float32)

            out[s0:s0 + target] += seg_wav
    return out


def main(argv=None):
    from . import config as cfgmod

    ap = argparse.ArgumentParser(description="DDSP-SVC 换声（dafeiyu-svc）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=None, help="权重目录（含 model_*.pt 与 config.yaml）")
    ap.add_argument("--assets", default=None, help="底座目录（pretrain/ 布局）")
    ap.add_argument("--ddsp-repo", default=None)
    ap.add_argument("--config", default=None, help="config.toml 路径")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--t-start", type=float, default=None)
    ap.add_argument("--key", type=float, default=None, help="移调半音")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0, help="reflow 从噪声起步，固定种子才可复现")
    a = ap.parse_args(argv)

    cfg = cfgmod.load(a.config)
    for name, val in (("weights", a.weights), ("assets", a.assets), ("ddsp_repo", a.ddsp_repo)):
        if val:
            setattr(cfg.paths, name, Path(val).expanduser())
    if a.steps is not None:
        cfg.infer.steps = a.steps
    if a.t_start is not None:
        cfg.infer.t_start = a.t_start
    if a.device is not None:
        cfg.infer.device = a.device
    key = cfg.infer.key if a.key is None else a.key

    inp, outp = _abs(a.input), _abs(a.output)
    cfg.paths.weights = cfg.paths.weights.resolve()
    cfg.paths.assets = cfg.paths.assets.resolve()
    cfg.paths.ddsp_repo = cfg.paths.ddsp_repo.resolve()

    import random

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    engine = load_engine(cfg)
    wav, sr = sf.read(inp, dtype="float32")
    print(f"input {inp}  {len(wav) / sr:.2f}s  sr={sr}  target_sr={engine['args'].data.sampling_rate}")
    y = convert(engine, wav, sr, cfg, key=key)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    sf.write(outp, y, engine["args"].data.sampling_rate)
    print(f"written {outp}  {len(y) / engine['args'].data.sampling_rate:.2f}s")


if __name__ == "__main__":
    main()
