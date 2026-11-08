# NOTICE — 第三方组件、数据与许可

本项目代码以 MIT 发布（见 `LICENSE`）；**模型权重**以 CC BY-NC 4.0 发布（见 `LICENSE-WEIGHTS.md`）。
下列组件/数据不随本仓库分发，需使用者自行获取，并遵守各自的许可。

## 训练数据

| 名称 | 许可 / 条款 | 说明 |
|---|---|---|
| Opencpop（`wenet/opencpop`） | `TERMS_OF_ACCESS`：**non-commercial research and education only** | 本项目权重由它训练而来，因此权重限非商用 |

## 代码与模型

| 名称 | 许可 | 用途 |
|---|---|---|
| DDSP-SVC 6.2 (`yxlllc/DDSP-SVC`) | MIT | 训练/推理框架（本仓库含补丁） |
| ContentVec (`lengyue233/content-vec-best`, HF 版 `TechInterMezzo/contentvec-500`) | 见上游模型卡 | 内容特征（768 维，第 12 层） |
| rmvpe (`lj1995/VoiceConversionWebUI`) | 见上游 | F0 提取 |
| NSF-HiFiGAN 44.1k (openvpi/vocoders, **2024.02**) | MIT（随包 NOTICE/STATEMENTS） | 声码器（`model.ckpt` 与 `NOTICE.txt` 需一并分发） |
| SoulX-Singer | Apache-2.0 | 教师之一（SVS/SVC） |
| Seed-VC (`Plachtaa/seed-vc`) | **GPL-3.0** | 仅作为离线教师工具调用；本仓库**不包含**其代码 |
| FlashSR (`YatharthS/FlashSR`) | 见上游 | 带宽扩展（造数据用） |
| Mel-Band-RoFormer karaoke (`aufr33/viperx`, `Frazer/Becruily`, `Anvuew`, `GaboxR67`) | 见各模型卡 | 主唱/和声分离 |
| `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956` | 见上游 | 初版分离模型 |
| CAM++ (`iic/speech_campplus_sv_zh-cn_16k-common`, ModelScope) | Apache-2.0 | 身份相似度评测 |

## 音色参考音

本项目**不包含**任何参考音。历史上用过的参考音来自第三方公开仓库
`zexuanw958-svg/dafeiyu-config-pack`（MIT）。该仓库 README 声明"插件与表情包来自原作者"，
因此我们不认为可以对该插件内附带的音频做转授权，故不随仓库分发。

## 版权声明

「大肥鱼」为其各自权利人的名称/形象。本项目与之**没有任何关联**，也未获得其授权或背书。
