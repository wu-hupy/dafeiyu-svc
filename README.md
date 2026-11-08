# dafeiyu-svc

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE-WEIGHTS.md)
[![python](https://img.shields.io/badge/python-3.10%2B-informational.svg)](requirements.txt)

仓库：<https://github.com/wu-hupy/dafeiyu-svc> · 训练与造数据：<https://github.com/wu-hupy/dafeiyu-svc-train>（GPL-3.0）

用 DDSP-SVC（Reflow）做**歌声换声（many-to-one singing voice conversion）**的一套可复现流水线：
数据清洗 → 教师换声 → 配对频响校正 → 空气带补齐 → 一致性剪枝 → 训练 → 推理 → 评测。

> ⚠️ 本仓库**不含任何音色参考音、不含训练音频、不含成品歌曲**。
> 参考音请自备（`--prompt`），并确保你拥有它的使用权。

## 1. 这是什么

- 一个 16.9M 参数的 Reflow 换声模型（checkpoint `model_24000.pt`，权重发布在
  [ModelScope: `wuhupy/dafeiyu-svc-v6`](https://modelscope.cn/models/wuhupy/dafeiyu-svc-v6)）
- 一套训练配方：如何从公开歌声数据造出**干净、音色互相一致**的训练集
- 一套评测工具：`tilt / hi10k / 12–16k / 非谐波细节 / f0 误差 / CAM++ 身份相似度`
- 一套踩坑记录：`docs/TROUBLESHOOTING.md`（这是本项目最值钱的部分之一）

## 2. 快速开始（推理）

**先看清要下什么** —— 这是本仓最容易误解的地方：

| 要下什么 | 大小 | 从哪来 |
|---|---|---|
| v6 权重 `model_24000.pt` | **67.8 MB** | 本仓 Releases / ModelScope |
| 底座：DDSP-SVC 代码 + rmvpe + ContentVec-HF + NSF-HiFiGAN(2024.02) | **约 480 MB** | 上游，脚本自动拉 |
| 合计 | **约 0.55 GB** | |

所以「下个 67 MB 就能跑」是错的：**67 MB 只是 Reflow 那部分权重**，回放还要底座四件。

### 一条命令拉齐并校验

```bash
pip install -r requirements.txt
python -m dafeiyu_svc.bundle --root .        # 拉齐底座 + 权重，逐项校验字节数与 sha256
```

它会打印一张体检表，缺什么、多大、路径在哪，一目了然；再跑一次就是增量补齐：

```bash
python -m dafeiyu_svc.bundle --root . --check-only    # 只体检，不下载
```

产物布局与 `config.example.toml` 的默认值一致：

```
./assets/pretrain/{rmvpe,contentvec_hf,nsf_hifigan}/...
./third_party/DDSP-SVC/...
./weights/v6/{model_24000.pt,config.yaml}
```

国内网络默认走 hf-mirror / ghfast；要换镜像看 `dafeiyu_svc/assets.py` 顶部的 URL 常量。

### 换声

```bash
cp config.example.toml config.toml
python -m dafeiyu_svc.convert --input dry_vocal.wav --output out.wav
```

`config.toml` 里写死的路径可用 `--assets / --weights / --ddsp-repo` 或环境变量
`DAFEIYU_ASSETS / DAFEIYU_WEIGHTS / DAFEIYU_DDSP_REPO` 覆盖。

自检（合成 3 秒类人声，不依赖任何受版权保护的音频）：

```bash
python tests/smoke.py --assets ./assets --weights ./weights/v6
# SMOKE: PASS   len diff 0 / rms -20.5 dBFS / deterministic: True
```

> 若你自己换过 `config.yaml` 并在里面写了中文注释：DDSP 用 locale 编码读 YAML，
> 在中文 Windows 上会报 `UnicodeDecodeError: 'gbk' codec ...`。本包的加载器已强制 UTF-8，
> 直接调 DDSP 原始脚本时请设 `PYTHONUTF8=1`。

## 3. 训练（配方已发布）

完整的训练配方在 **[`docs/TRAINING.md`](docs/TRAINING.md)**：选片 → 教师换声（Seed-VC 44.1k）
→ 配对频响校正 → 一致性剪枝 → Reflow 训练 → 四个对象横比验收。每一步都写了阈值、
实测数字，以及"为什么不能用统一曲线 / 不能用到质心距离"这类反例。

造数据阶段用到 Seed-VC，它是 **GPL-3.0**，所以本仓库**不 import 它的代码**，只用
[`tools/seedvc_generate.py`](tools/seedvc_generate.py) 以**子进程**方式调用你自己装的 Seed-VC
（两个独立程序通过命令行通信，不构成链接，因此本仓库可以保持 MIT）。

```bash
pip install -r requirements.txt
python tools/seedvc_generate.py --seedvc-dir /path/to/seed-vc --list clips.txt \
    --ref my_reference.wav --out data/teacher --steps 50 --cfg 0.7
```

> 想要"一个仓里跑完全部流程"的话，唯一诚实的做法是把整仓改成 GPL-3.0；
> 我们不这么做，是为了让推理部分能被商业项目直接依赖。

对应的训练仓版本：**[`dafeiyu-svc-train v0.1.0`](https://github.com/wu-hupy/dafeiyu-svc-train/tree/v0.1.0)**。
训练仓更新后，本文档与 v6 权重的对应关系以这个 tag 为准。

## 4. 已知限制

- 参考音如果是**说话**而不是**唱歌**，模型会继承它"没有气声、没有 12k 以上"的上限（见 MODEL_CARD）。
- 24 kHz 的教师（SVS 类）会把高频封顶，学生模型学不回来；要空气感请用 44.1k 教师。
- 颗粒感（"砂纸"）主要来自声码器（NSF-HiFiGAN 的噪声支路），换预测器/加步数都压不下去。
- 只支持单声部：输入必须先去和声（见 `docs/PIPELINE.md` 的 karaoke 前置步骤）。

## 5. 许可

- **代码**：MIT（见 `LICENSE`）
- **模型权重**：CC BY-NC 4.0（非商用），因为训练数据源自 Opencpop（其条款限定 non-commercial research/education）
- 第三方组件与数据的许可见 `NOTICE.md`

## 6. 引用

如果你在论文/项目里用到这套流程：

```bibtex
@misc{dafeiyu_svc,
  title  = {dafeiyu-svc: a reproducible pipeline for many-to-one singing voice conversion},
  year   = {2026},
  note   = {https://github.com/wu-hupy/dafeiyu-svc}
}
```
