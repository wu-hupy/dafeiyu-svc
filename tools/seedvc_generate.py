#!/usr/bin/env python
"""用 Seed-VC 批量生成「教师换声」数据（外部进程调用，不 import 其代码）。

为什么是子进程而不是 import：
    Seed-VC 是 GPL-3.0。本仓库是 MIT。把 GPL 代码 import 进来会让整仓变成衍生作品，
    直接传染成 GPL-3.0。子进程调用是「两个独立程序通过命令行通信」，不构成链接，
    所以本脚本可以留在 MIT 里；你自己装的 Seed-VC 仍然是它自己的 GPL。

用法：
    python tools/seedvc_generate.py --seedvc-dir D:/path/to/seed-vc --list clips.txt --ref reference.wav --out data/teacher --steps 50 --cfg 0.7

    （clips.txt 每行一个源音频的绝对路径）

产物：
    <out>/<clip id>.wav         44.1k 单声道（Seed-VC 44k 模型原样输出）
    <out>/manifest.json         id / src / out / steps / cfg / 耗时

注意：
    - 首次运行会下载 whisper-small 与 bigvgan 到 Seed-VC 的 checkpoints/hf_cache，
      国内请设 HF_ENDPOINT=https://hf-mirror.com（本脚本默认就设了）。
    - --resume 会跳过 manifest 里已经成功的条目。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def find_output(outdir: Path, since: float) -> Path | None:
    """Seed-VC 的输出文件名由它自己拼，找这一轮新出现的 wav 即可。"""
    best, best_m = None, since
    for p in outdir.rglob("*.wav"):
        m = p.stat().st_mtime
        if m >= best_m:
            best, best_m = p, m
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedvc-dir", required=True, help="seed-vc 仓库根（含 inference.py）")
    ap.add_argument("--list", required=True, help="每行一个源音频绝对路径")
    ap.add_argument("--ref", required=True, help="音色参考音 wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=0.7)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=int, default=900, help="单条超时（秒）")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    root = Path(a.seedvc_dir).expanduser().resolve()
    if not (root / "inference.py").is_file():
        print(f"[x] 找不到 {root / 'inference.py'}", file=sys.stderr)
        return 2

    items = [Path(s.strip()) for s in Path(a.list).read_text(encoding="utf-8").splitlines() if s.strip()]
    items = [p for p in items if p.is_file()]
    if not items:
        print("[x] 清单里没有可用的文件", file=sys.stderr)
        return 2

    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    local_out = root / "_svc_work"          # Seed-VC 只写自己目录下，避免相对路径混乱
    local_out.mkdir(exist_ok=True)

    mf_path = out / "manifest.json"
    done = {}
    if a.resume and mf_path.is_file():
        done = {r["id"]: r for r in json.loads(mf_path.read_text(encoding="utf-8"))}

    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("PYTHONUTF8", "1")

    records, t0, ok, fail = [], time.time(), 0, 0
    for i, src in enumerate(items, 1):
        cid = src.stem
        if cid in done and (out / f"{cid}.wav").is_file():
            records.append(done[cid])
            continue

        t = time.time()
        since = t - 1.0
        cmd = [
            a.python, "inference.py",
            "--source", str(src),
            "--target", str(Path(a.ref).expanduser().resolve()),
            "--output", str(local_out),
            "--f0-condition", "True",
            "--auto-f0-adjust", "False",
            "--semi-tone-shift", "0",
            "--inference-cfg-rate", str(a.cfg),
            "--diffusion-steps", str(a.steps),
            "--fp16", "True",
        ]
        try:
            subprocess.run(cmd, cwd=str(root), env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           timeout=a.timeout)
            produced = find_output(local_out, since)
            if produced is None:
                raise RuntimeError("没有产出 wav")
            dst = out / f"{cid}.wav"
            produced.replace(dst)
            rec = {"id": cid, "src": str(src), "out": str(dst),
                   "steps": a.steps, "cfg": a.cfg, "seconds": round(time.time() - t, 1)}
            ok += 1
        except subprocess.TimeoutExpired:
            rec = {"id": cid, "src": str(src), "error": f"timeout {a.timeout}s"}
            fail += 1
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-1:]
            rec = {"id": cid, "src": str(src), "error": " ".join(tail) or f"exit {e.returncode}"}
            fail += 1
        except Exception as e:  # noqa: BLE001
            rec = {"id": cid, "src": str(src), "error": f"{type(e).__name__}: {e}"}
            fail += 1

        records.append(rec)
        if i % 10 == 0 or i == len(items):
            el = time.time() - t0
            eta = el / i * (len(items) - i)
            print(f"[{i}/{len(items)}] ok={ok} fail={fail}  {el/60:.1f}min  eta {eta/60:.1f}min", flush=True)
            mf_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    mf_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] ok={ok} fail={fail}  -> {out}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
