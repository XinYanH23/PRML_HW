"""
与 hw_4.ipynb 对齐的 Transformer 实现，并为大作业第二部分预留配置开关：
- use_residual：是否残差（2.3；False 时为 ``LayerNorm(Dropout(Sublayer(x)))``，无 ``x+``）
- pos_encoding_type：sin / none（2.1；none 时 ``PositionalEncoding`` 直接返回输入，不加 PE）
- qkv_mode：standard / shared（2.2；仅在同源 Q=K=V 的自注意力上使用 shared）
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                scores = scores + mask
        attn_weights = F.softmax(scores, dim=-1)
        # Dropout 仅在 training=True 时随机置零；eval() 下为恒等映射，返回的 attn_weights 与 softmax 输出一致，
        # 便于推理期热力图 / BLEU 与训练分布对齐（见仓库根目录《说明.md》）。
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_mode: str = "standard",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")
        if qkv_mode not in ("standard", "shared"):
            raise ValueError('qkv_mode 须为 "standard" 或 "shared"')
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv_mode = qkv_mode

        if qkv_mode == "standard":
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
        else:
            self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key.size(1) != value.size(1):
            raise ValueError("key 与 value 的序列长度必须一致")

        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.size(1)

        if self.qkv_mode == "shared" and key is query and value is query:
            qkv = self.W_qkv(query)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            if self.qkv_mode == "shared":
                raise RuntimeError(
                    "qkv_mode='shared' 仅允许在自注意力中传入同一 tensor 作为 Q,K,V；"
                    "交叉注意力请使用 qkv_mode='standard'。"
                )
            q = self.W_q(query)
            k = self.W_k(key)
            v = self.W_v(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        attn_out, attn_weights = self.attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        return self.W_o(attn_out), attn_weights


class PositionalEncoding(nn.Module):
    """
    正弦/余弦绝对位置编码（论文 Section 3.5），或 2.1 消融 ``encoding_type="none"``。

    **预测（无 PE 时注意力退化，与「排列不变性」相关）**：
    缩放点积自注意力对 **key 维的置换** 在结构上具有等变性：若仅由词嵌入提供内容、无位置区分，
    则各位置在表示空间上缺少可区分的「序号坐标」，编码器对德语词序（框式结构、动词第二位等）
    的显式建模能力会显著变弱；解码端除因果 mask 外亦无绝对位置，跨位置依赖更难对齐到正确语序，
    往往表现为 BLEU 下降、长句乱序或主谓宾粘连错误。因果 mask 只禁止「看未来」，并不编码「谁在第几位」。
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        encoding_type: str = "sin",
    ) -> None:
        super().__init__()
        if encoding_type not in ("sin", "none"):
            raise ValueError('encoding_type 须为 "sin" 或 "none"')
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout) if encoding_type == "sin" else None

        if encoding_type == "sin":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        else:
            self.register_buffer("pe", torch.empty(0))  # 占位，forward 不使用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding_type == "none":
            # 2.1 消融：不加任何位置信号，也不在此处做 dropout（与「仅去掉 PE」一致）
            return x
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        assert self.dropout is not None
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.w_1(x))
        h = self.dropout(h)
        return self.w_2(h)


class EncoderLayer(nn.Module):
    """
    单层 Encoder（Post-LN）。
    use_residual=True（默认）：LayerNorm(x + Dropout(Sublayer(x)))。
    use_residual=False（2.3 消融）：LayerNorm(Dropout(Sublayer(x)))，无恒等捷径；深层下梯度更难回传，
    常表现为 loss 下降慢、震荡或不收敛，可与 Baseline 前若干 epoch 对比论证残差作用。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        qkv_mode: str = "standard",
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, qkv_mode=qkv_mode)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        if self.use_residual:
            x = self.norm1(x + self.dropout1(attn_out))
        else:
            x = self.norm1(self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        if self.use_residual:
            x = self.norm2(x + self.dropout2(ffn_out))
        else:
            x = self.norm2(self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """
    单层 Decoder（Post-LN）；use_residual 语义同 EncoderLayer（2.3 消融时三处子层均无 x+）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_residual: bool = True,
        qkv_mode: str = "standard",
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, qkv_mode=qkv_mode)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, qkv_mode="standard")
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sa_out, _ = self.self_attn(x, x, x, tgt_mask)
        if self.use_residual:
            x = self.norm1(x + self.dropout1(sa_out))
        else:
            x = self.norm1(self.dropout1(sa_out))
        ca_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        if self.use_residual:
            x = self.norm2(x + self.dropout2(ca_out))
        else:
            x = self.norm2(self.dropout2(ca_out))
        ff_out = self.ffn(x)
        if self.use_residual:
            x = self.norm3(x + self.dropout3(ff_out))
        else:
            x = self.norm3(self.dropout3(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        N: int = 6,
        dropout: float = 0.1,
        use_residual: bool = True,
        qkv_mode: str = "standard",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout, use_residual=use_residual, qkv_mode=qkv_mode)
                for _ in range(N)
            ]
        )

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        N: int = 6,
        dropout: float = 0.1,
        use_residual: bool = True,
        qkv_mode: str = "standard",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout, use_residual=use_residual, qkv_mode=qkv_mode)
                for _ in range(N)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


def make_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(size: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    m = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return m.view(1, 1, size, size)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        N: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
        pos_encoding_type: str = "sin",
        use_residual: bool = True,
        qkv_mode: str = "standard",
    ) -> None:
        super().__init__()
        if pos_encoding_type not in ("sin", "none"):
            raise ValueError('pos_encoding_type 须为 "sin" 或 "none"')
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(
            d_model,
            max_len=max_len,
            dropout=dropout,
            encoding_type=pos_encoding_type,
        )
        self.encoder = Encoder(
            d_model, num_heads, d_ff, N=N, dropout=dropout, use_residual=use_residual, qkv_mode=qkv_mode
        )
        self.decoder = Decoder(
            d_model, num_heads, d_ff, N=N, dropout=dropout, use_residual=use_residual, qkv_mode=qkv_mode
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        trg: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        trg_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        y = self.tgt_embed(trg) * math.sqrt(self.d_model)
        y = self.pos_enc(y)
        return self.decoder(y, memory, trg_mask, memory_mask=src_mask)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        trg_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        dec_out = self.decode(trg, memory, src_mask, trg_mask)
        return self.generator(dec_out)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        mask = targets.ne(self.padding_idx)
        with torch.no_grad():
            dist = torch.empty_like(log_probs).fill_(self.smoothing / (self.vocab_size - 1))
            dist.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
        loss = F.kl_div(log_probs, dist, reduction="none").sum(-1)
        return (loss * mask.float()).sum() / mask.sum().clamp_min(1.0)


class NoamOpt:
    def __init__(
        self,
        model_size: int,
        factor: float,
        warmup: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def step(self) -> None:
        self._step += 1
        lr = self.rate()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.step()

    def rate(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self._step
        step = max(1, step)
        return self.factor * (self.model_size ** (-0.5)) * min(
            step ** (-0.5), step * (self.warmup ** (-1.5))
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


def build_noam_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
    factor: float = 1.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        s = step + 1
        return factor * (d_model ** (-0.5)) * min(s ** (-0.5), s * (warmup_steps ** (-1.5)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_transformer_from_config(
    config: dict,
    src_vocab_size: int,
    tgt_vocab_size: int,
) -> Transformer:
    """由 train.py 中的 config 字典构造模型。"""
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        N=config["num_layers"],
        dropout=config["dropout"],
        max_len=config.get("max_pe_len", 5000),
        pos_encoding_type=config.get("pos_encoding_type", "sin"),
        use_residual=config.get("use_residual", True),
        qkv_mode=config.get("qkv_mode", "standard"),
    )
