# 补丁：让 DDSP-SVC 用 transformers 格式的 ContentVec

**为什么**：原版的 `pretrain/contentvec/checkpoint_best_legacy_500.pt` 是 fairseq 的 pickle，
没有 fairseq 连 `torch.load` 都会抛 `ModuleNotFoundError: No module named 'fairseq'`。
装 fairseq 在 Python 3.11 上要编译，且会拖进一堆旧依赖。

**为什么不用桩模块**：我们试过用 `meta_path` 造假 fairseq 模块把权重读出来（可行，能拿到 214 键 state dict），
但既然 HF 上有等价的 transformers 版（`TechInterMezzo/contentvec-500`），直接用更干净。

## 1. 准备权重

```bash
# 落到 assets/pretrain/contentvec_hf/
config.json          (1375 B)
model.safetensors    (188,767,088 B)
```

对应的 encoder 名字在本项目里叫 `contentvec768l12_hf`。

## 2. 改 `ddsp/vocoder.py`（两处）

### (a) 顶部 import 加兜底

```python
try:
    from fairseq import checkpoint_utils
except Exception:
    checkpoint_utils = None
```

### (b) 新增一个编码器类 + 一个分支

```python
class Audio2ContentVec768L12HF(nn.Module):
    """用 transformers 的 HubertModel 取第 12 层隐状态，替代 fairseq 版 ContentVec。"""

    def __init__(self, path, device="cpu"):
        super().__init__()
        from transformers import HubertModel
        self.model = HubertModel.from_pretrained(path).to(device).eval()
        self.device = device

    def forward(self, wav, sample_rate, hop_size):
        # wav: [B, T] float32，sample_rate 需为 16000
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        wav = torch.nn.functional.pad(wav, (160, 160))
        with torch.no_grad():
            out = self.model(wav.to(self.device), output_hidden_states=True)
        return out.hidden_states[12]
```

然后在 `Units_Encoder.__init__` 的分派处加：

```python
elif encoder == 'contentvec768l12_hf':
    self.encoder = Audio2ContentVec768L12HF(encoder_ckpt, device)
    self.encoder_sample_rate = 16000
    self.encoder_hop_size = 320
```

## 3. 验证

```python
from ddsp.vocoder import Units_Encoder
enc = Units_Encoder('contentvec768l12_hf', 'pretrain/contentvec_hf', 16000, 320, device='cuda')
y = enc.encode(torch.randn(1, 44100).cuda(), 44100, 441)   # 期望 [1, ~T/320, 768]
```

正常输出的特征：`std ≈ 0.32`，相邻帧余弦 ≈ 0.85，远距帧余弦 ≈ 0.04。
