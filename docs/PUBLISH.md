# PUBLISH — 发布到哪、怎么发

## 1. 三处分工

| 内容 | 平台 | 备注 |
|---|---|---|
| 代码 + 文档 | **GitHub**（主仓） | MIT |
| 代码镜像 | **Gitee** | 国内 clone 不用代理（GitHub 直连常超时） |
| 权重 | **HuggingFace Hub** + **ModelScope** | 双发；HF 门面，MS 给国内速度 |
| Demo（可选） | HF Space / ModelScope 创空间 | 只有要做网页试用才需要 |

## 2. 首次发布

```bash
# ---- 代码 ----
cd release/dafeiyu-svc
git init -b main
git add -A
git commit -m "feat: DDSP-SVC v6 singing voice conversion pipeline"
gh repo create <your-name>/dafeiyu-svc --public --source=. --push
# 或者手动：git remote add origin git@github.com:<your-name>/dafeiyu-svc.git && git push -u origin main

# ---- Gitee 镜像 ----
# 网页新建空仓，然后：
git remote add gitee git@gitee.com:<your-name>/dafeiyu-svc.git
git push gitee main

# ---- 权重（HuggingFace）----
pip install -U huggingface_hub
huggingface-cli login
python tools/export_weights.py \
  --ckpt  <path>/exp/dafeiyu_v6/model_24000.pt \
  --config <path>/exp/dafeiyu_v6/config.yaml \
  --out    ./_weights_release
huggingface-cli upload <your-name>/dafeiyu-svc-v6 ./_weights_release . --repo-type model

# ---- 权重（ModelScope 镜像）----
pip install modelscope
modelscope login --token <YOUR_MS_TOKEN>
modelscope upload <your-name>/dafeiyu-svc-v6 ./_weights_release
```

## 3. 权重仓要有什么

```
model_24000.pt
config.yaml            # encoder=contentvec768l12_hf，路径以 assets/ 为根
SHA256SUMS
LICENSE-WEIGHTS.md     # CC BY-NC 4.0
MODEL_CARD.md          # 直接复用仓库里的那份
README.md              # 3 行：是什么、怎么用、限制
```

HF 的 repo 元信息（写进 `README.md` 顶部的 YAML）：

```yaml
---
license: cc-by-nc-4.0
tags: [audio, voice-conversion, singing-voice-conversion, ddsp-svc, reflow]
datasets: [wenet/opencpop]
language: [zh]
---
```

## 4. 发布前自检

- [ ] `grep -r "D:\\\\" .` 和 `grep -r "/home/" .` 没有绝对路径泄漏
- [ ] 仓库里没有任何 `.wav / .mp3 / .flac`（`git ls-files | grep -E '\.(wav|mp3|flac)$'` 应为空）
- [ ] 没有 `config.toml`（只有 `config.example.toml`）
- [ ] 在**干净目录** clone 下来能跑通 smoke test
- [ ] README 第一屏就写了"不含音色参考音、非商用"
- [ ] `NOTICE.md` 覆盖了 DDSP-SVC / ContentVec / rmvpe / NSF-HiFiGAN / Opencpop

## 5. 公告文案（可直接用）

> dafeiyu-svc：一套把「教师换声 → 配对校正 → 空气带补齐 → 一致性剪枝 → Reflow 训练」串起来的歌声换声流水线，
> 附 16.9M 参数模型与全部踩坑记录。
> **仓库不含任何音色参考音与训练音频**，权重限非商用（训练数据为 Opencpop，NC 条款）。
> 请勿用于未经同意的声音模仿。

## 6. 之后（v0.2）

- 训练脚本整理发布（当前只有配方文档 `docs/PIPELINE.md`）
- 评测脚本（`tilt / hi10k / 12–16k / 细节 / CAM++`）
- 一个"无音频依赖"的合成样例，用于 CI
