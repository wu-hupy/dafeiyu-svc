# TROUBLESHOOTING — 我们真踩过的坑

按"你会先遇到"的顺序排。每条都写清**症状 / 原因 / 解法**。

## 环境

### `ModuleNotFoundError: No module named 'fairseq'`
DDSP-SVC 默认的 ContentVec 是 fairseq 的 pickle，没有 fairseq 连 `torch.load` 都进不去。
**解法**：用 transformers 格式的 ContentVec（见 `patches/contentvec_hf.md`），或装 fairseq（Python 3.11 上编译很痛苦）。

### 声码器加载报 `Unexpected key(s): "source_conv.weight"` / 缺 `m_source.*`
`pretrain/nsf_hifigan/model.ckpt` 拿成了 **2025.02 的 pc 版**。
**解法**：换 2024.02（`nsf_hifigan_44.1k_hop512_128bin_2024.02.zip`，52,814,405 B），
`openvpi/vocoders` 的 release。HF 上 `Nevertree/PC-NSF-HIFIGAN` 与 2025.02 字节相同，也是错的那份。

### `FileNotFoundError: 'pretrain/nsf_hifigan/model'`
DDSP 的 vocoder 懒加载且用相对路径；chdir 回来后就找不到。
**解法**：把 `vocoder.ckpt` 写成相对 `assets/` 的完整文件名（带 `.ckpt`），并在 `assets/` 目录里运行。

### Windows：两个进程同时 bind 同一个端口
netstat 显示两个 PID 都 LISTENING，请求被分给旧进程，改代码不生效。
**解法**：按 PID 杀干净再起；重启服务前 `Get-Process python | Stop-Process`。

### `ModuleNotFoundError: No module named 'audiotools'`（Seed-VC）
Descript 的 dac 依赖 `audiotools`，PyPI 上没有。
**解法**：写一个最小桩包满足 import（我们只用了 `AudioSignal`、`STFTParams`、`ml.Accelerator`）。

## 数据

### 训练出来的模型"哑"
不是数据脏，而是**参考音本身没有高频**：说话参考 12–16k 只有 0.03%，学生模型把这个上限学进去了。
**解法**：换 44.1k 歌唱参考，或在训练目标里补空气带（`fill` 步骤）。

### 换完"一人一句"，两种音色交替
数据里混了两种音色（真人 + 换声），模型没有音色条件，只能靠句子级线索分区。
**解法**：两两余弦剪枝到 ≥0.82；不要用"到质心的距离"，多峰分布会失效。

### 用 SoulX-SVC 当老师，细节全没了
它是 SVS 模型被当 SVC 用：**重新合成**而不是转换，非谐波细节从 7.2% 掉到 0.13%，12k 以上归零。
**解法**：换 44.1k 的 Seed-VC；或接受"干净但闷"。

### 和声没去掉 → 听起来糊
分离器给的人声轨常混着二声部，两条 f0 喂进去会被糊成一条。
**解法**：先做 karaoke 主唱/和声分离；再加"只在有和声的段落用分离结果"的门控拼接（否则模型留洞）。

### 主唱被"分残"（整句消失）
karaoke 模型会把主唱整块判给和声轨（我们的旧模型在跳楼机上 20.1% 的有声帧被判静音）。
**解法**：换 Frazer 版 BS-RoFormer（1.0%），或三模型平均（1.1%）。

## 训练

### 前 1k 步 `test()` 崩溃、`FileNotFoundError`
config 里 `vocoder.ckpt` 路径没写扩展名，验证时懒加载才炸。
**解法**：写全 `model.ckpt`；崩了也能从中断的 `model_*.pt` 续训。

### val 指标一直不涨
多半是 val 与 train 同源（同句泄漏），或数据量已经过配（90 分钟 → 20k 步之后再训没收益）。

### 想做 2:1 的数据配比，但采样器不支持权重
DDSP 的采样是均匀随机。**解法**：把想加权的文件在目录里放两份（文件名不同即可）。

## 推理

### 波形每跑一次都不一样
reflow 从随机噪声起步。**解法**：固定 `--seed`。

### 长静音后的第一个字"砰"一声
门控增益按 20 ms 帧算 + `np.repeat` 铺到样本 → 帧边界台阶；切片起点又正好在字头。
**解法**：帧域 100 ms 平滑 + 逐样本插值 + 段首 10 ms 升余弦（本项目已内置）。

### 有"砂纸/颗粒"感
声码器噪声支路的帧率级随机相位，1–4k 包络抖动是源 1.5–2.5 倍。
已验证**无效**的手段：加步数（50→100）、换 2025.02 声码器、mel 时间平滑、多遍平均（相位不同，会互相抵消）。
剩下的方向：换声码器家族、换模型架构。

## 网络

### GitHub / HuggingFace 下载失败
- GitHub：用 `https://ghfast.top/https://github.com/...`
- HF：`https://hf-mirror.com/...`，**resolve 链接必须带 `?download=true`**，否则大量 429
- 大文件用分块 + `Content-Range` 校验 + sha256，别用"下载失败就重头来"的写法
- 几 KB 的 yaml 不要走 hf-mirror（会卡在 chunk0 超时），直连 ghfast
