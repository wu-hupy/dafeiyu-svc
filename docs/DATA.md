# DATA — 数据来源与合规说明

## 训练数据

| 项 | 值 |
|---|---|
| 数据集 | Opencpop（`wenet/opencpop`） |
| 规模 | 100 首中文女声 / 5.226 h / 3756 段 3–8 s 切片 |
| 授权 | `TERMS_OF_ACCESS`：**non-commercial research and education only** |
| 我们实际用的 | 3540 段（3–8 s）/ 304.2 min，经教师换声 + 校正 + 剪枝后取 1092 段 / 91.8 min |
| 留出 | 整首 2081、2095 两首做验证 |

**结论**：由它训练出的权重只能非商用。这条限制不因再加工而消失。

## 我们不分发的东西

- 任何原始音频（Opencpop、成品歌曲、分轨）
- 任何音色参考音（含其 EQ 版、BWE 版变体）
- 训练中间产物（换声后的 3540 条）

## 音色参考音的历史

参考音取自第三方公开仓库 `zexuanw958-svg/dafeiyu-config-pack`（MIT）。
该仓库 README 声明"插件与表情包来自原作者"，而参考音在该插件目录内，
因此我们**不认为可以对其转授权**，决定不随仓库分发。

记录（供审计）：

- 文件：`plugins/astrbot_plugin_dafeiyu_sing/sing_server/voice/prompt.wav`
- 32 kHz / 5.20 s / 332,844 B / sha256 `75bf23de…`
- 与另一份 5.84 s 版本（`dafeiyu-prompt.wav`，sha256 `0c7d1ef5…`）为同一段录音的不同裁剪

## 评测素材

- 评测用的《雨爱》片段来自 CD 音源经 BS-RoFormer 分离的主唱轨，**仅在本地使用**，不随仓库分发。
- 公开可复现的评测请改用 Opencpop 的 test 切片（同样受 NC 限制）。

## 复现你需要准备

1. Opencpop（自行申请/下载，遵守其条款）
2. 一个你**有权使用**的音色参考音（10 s 左右、干净、无伴奏）
3. 一条待换声的干声（去伴奏、去和声）
