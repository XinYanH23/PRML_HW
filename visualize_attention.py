#!/usr/bin/env python3
"""
加载 ``best_model.pt``，对给定德语句（或测试集某一行）提取 Encoder **自注意力**权重并绘制热力图。

用法示例::

    cd hw_4
    python visualize_attention.py \\
        --checkpoint runs/run_xxx/best_model.pt \\
        --test-index 0 \\
        --layer 5 \\
        --out attn_compare.png

跨图对比热力图明暗时，可加 ``--attn-colorbar fixed01 --attn-vmin 0 --attn-vmax 1`` 固定色条（见《说明.md》）。

在 **sin PE 训练** 的 checkpoint 上：先按训练设置画图，再仅改 ``pos_enc.encoding_type`` 为 ``none``
做第二次前向（不重建模型，避免 ``pe`` buffer 形状不一致）；用于观察热力图与对角线统计量变化。
无 PE 训练的权重无法安全注入 sin 表，脚本会只输出 ``none`` 一张图并给出提示。

依赖: ``matplotlib``（``pip install matplotlib``）

---------------------------------------------------------------------------
报告可引用结论（注意力熵，对比「正常推理」与「推断期关闭 PE」）：

在**同一组已训练权重**下，仅将 ``pos_enc.encoding_type`` 置为 ``none`` 时，
各行注意力分布往往更分散或更不规则，**平均香农熵（Attention Entropy）通常升高**；
而启用正弦位置编码时，不同绝对位置可被区分，查询–键匹配更**尖锐**，
从而**降低**注意力在键维上的平均不确定性（熵更低）。
这与「位置编码减轻位置歧义、有利于稳定、可解释的局部/对角相关结构」的叙述一致。
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 与 train.py 同目录
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from transformer_model import build_transformer_from_config, make_pad_mask

import train as train_mod


def _resolve_device(s: str) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _merge_config(base: Dict[str, Any], ckpt_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base)
    if ckpt_cfg:
        for k, v in ckpt_cfg.items():
            out[k] = v
    return out


@torch.no_grad()
def encoder_self_attention_weights(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    返回指定 Encoder 层的自注意力权重 ``[B, num_heads, L, L]``（与训练时 mask 一致）。
    """
    model.eval()
    enc = model.encoder
    if layer_idx < 0 or layer_idx >= len(enc.layers):
        raise ValueError(f"layer_idx 须在 [0, {len(enc.layers) - 1}]，当前为 {layer_idx}")

    x = model.src_embed(src) * math.sqrt(model.d_model)
    x = model.pos_enc(x)
    for i in range(layer_idx):
        x = enc.layers[i](x, src_mask)
    _, attn = enc.layers[layer_idx].self_attn(x, x, x, src_mask)
    return attn


def _src_valid_length(src_row: torch.Tensor, pad_idx: int) -> int:
    for i in range(src_row.numel() - 1, -1, -1):
        if int(src_row[i].item()) != pad_idx:
            return i + 1
    return 0


def _tokens_from_ids(ids: List[int], itos: List[str]) -> List[str]:
    return [itos[i] if 0 <= i < len(itos) else "<oob>" for i in ids]


def attention_entropy_nats(attn: torch.Tensor, valid_len: int, eps: float = 1e-8) -> float:
    """
    多头 Encoder 自注意力 ``attn: [H, L, L]``（行 softmax：对键维求和为 1）。
    对每个 head、每个 query 位置计算香农熵 ``-sum_k p_k log p_k``，再在 head 与 query 上取平均。
    单位：nats（自然对数）；用于对比 PE 开/关时注意力分布的「尖锐 vs 弥散」程度。
    """
    if valid_len <= 0:
        return 0.0
    p = attn[:, :valid_len, :valid_len].float().clamp_min(eps)
    ent_rows = -(p * p.log()).sum(dim=-1)  # [H, L]
    return float(ent_rows.mean().item())


def trace_mass_ratio(attn: torch.Tensor, valid_len: int) -> float:
    """
    统计量 ``Tr(A) / sum(A)``：对角质量占全矩阵（有效子块、所有 head）总质量的比例。
    对角占优时该值更大，便于与熵一同报告。
    """
    if valid_len <= 0:
        return 0.0
    a = attn[:, :valid_len, :valid_len].float()
    num = torch.diagonal(a, dim1=-2, dim2=-1).sum()
    den = a.sum().clamp_min(1e-12)
    return float((num / den).item())


def diagonal_concentration(attn: torch.Tensor, valid_len: int) -> Tuple[float, float, float]:
    """
    attn: [H, L, L]；仅在非 pad 的 valid_len×valid_len 子块上统计。

    返回 (mean_diag, mean_off, diag/off 比值)。
    """
    if valid_len <= 0:
        return 0.0, 0.0, 0.0
    a = attn[:, :valid_len, :valid_len].float()
    H, L, _ = a.shape
    diag = torch.diagonal(a, dim1=-2, dim2=-1)  # [H, L]
    mean_diag = diag.mean().item()
    # 全体均值与对角均值
    mean_all = a.mean().item()
    # off-diagonal 均值：用 (sum_all - sum_diag) / (L*L - L) / H
    sum_all = a.sum()
    sum_diag = diag.sum()
    denom = H * (L * L - L) if L > 1 else 1.0
    mean_off = ((sum_all - sum_diag) / denom).item() if L > 1 else mean_all
    ratio = mean_diag / mean_off if mean_off > 1e-8 else float("inf")
    return mean_diag, mean_off, ratio


def plot_multihead_heatmap(
    attn: torch.Tensor,
    tokens: List[str],
    title: str,
    out_path: Path,
    *,
    valid_len: int,
    attention_entropy_nats: float,
    max_heads_per_row: int = 4,
    colorbar_mode: str = "auto",
    attn_vmin: float = 0.0,
    attn_vmax: float = 1.0,
) -> None:
    """
    colorbar_mode:
      - ``auto``：有对比度时用分位数 + turbo；近似均匀时用相对均值的 RdBu_r（见《说明.md》）。
      - ``fixed01``：``vmin=attn_vmin, vmax=attn_vmax``（默认 0~1）+ viridis，便于跨图对比「稀释」程度。
      - ``deviation``：强制相对全局均值的 RdBu_r。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # [H, L, L] -> 只画 valid 部分
    a = attn[:, :valid_len, :valid_len].cpu().numpy()
    H = a.shape[0]
    ncols = min(H, max_heads_per_row)
    nrows = math.ceil(H / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), squeeze=False)
    tok_labels = tokens[:valid_len]
    for h in range(H):
        r, c = divmod(h, ncols)
        ax = axes[r][c]
        mat = a[h].astype(np.float64)
        if colorbar_mode == "fixed01":
            im = ax.imshow(mat, cmap="viridis", vmin=attn_vmin, vmax=attn_vmax, aspect="auto")
        elif colorbar_mode == "deviation":
            m = float(mat.mean())
            d = mat - m
            lim = max(float(np.abs(d).max()), 1e-8)
            im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        else:
            spread = float(mat.max() - mat.min())
            if spread > 1e-5:
                lo, hi = np.percentile(mat, [1.0, 99.0])
                lo = min(lo, float(mat.min()))
                hi = max(hi, float(mat.max()))
                if hi - lo < 1e-8:
                    lo, hi = float(mat.min()), float(mat.max()) + 1e-8
                im = ax.imshow(mat, cmap="turbo", vmin=lo, vmax=hi, aspect="auto")
            else:
                m = float(mat.mean())
                d = mat - m
                lim = max(float(np.abs(d).max()), 1e-8)
                im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"head {h}")
        ax.set_xticks(range(valid_len))
        ax.set_yticks(range(valid_len))
        ax.set_xticklabels(tok_labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(tok_labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for h in range(H, nrows * ncols):
        r, c = divmod(h, ncols)
        axes[r][c].set_visible(False)
    ent_line = f"Attention Entropy = {attention_entropy_nats:.4f} nats"
    fig.suptitle(f"{title}\n{ent_line}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder 多头自注意力热力图（PE 开/关对比）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="best_model.pt 路径（内含 model_state_dict 与可选 config）",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="若提供，则从该目录读取 config.json 补全超参（与训练时一致）",
    )
    parser.add_argument(
        "--de",
        type=str,
        default="",
        help="直接输入一句德语（与 --test-index 二选一）",
    )
    parser.add_argument("--test-index", type=int, default=-1, help="测试集句子下标（默认 0，若同时给 --de 则忽略）")
    parser.add_argument("--layer", type=int, default=-1, help="Encoder 层索引（默认最后一层）")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--out", type=str, default="attention_vis.png", help="输出图像路径（sin/none 会各写一张带后缀）")
    parser.add_argument(
        "--local-repo",
        type=str,
        default="",
        help="本地 Multi30k tok 仓库根目录（留空则与 train.py 相同自动探测）",
    )
    parser.add_argument(
        "--attn-colorbar",
        type=str,
        choices=("auto", "fixed01", "deviation"),
        default="auto",
        help="热力图：auto=自适应；fixed01=固定 vmin/vmax（默认 0~1，viridis，跨图可比）；deviation=相对均值 RdBu_r",
    )
    parser.add_argument(
        "--attn-vmin",
        type=float,
        default=0.0,
        help="与 --attn-colorbar fixed01 配合（默认 0）",
    )
    parser.add_argument(
        "--attn-vmax",
        type=float,
        default=1.0,
        help="与 --attn-colorbar fixed01 配合（默认 1；注意力行 softmax 后权重∈[0,1]）",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"找不到 checkpoint: {ckpt_path}")

    device = _resolve_device(args.device)
    ckpt = _load_checkpoint(ckpt_path, device)

    # 与训练对齐的 config（缺省与 train.py config 一致字段）
    base_cfg: Dict[str, Any] = {
        "num_layers": 6,
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_pe_len": 5000,
        "use_residual": True,
        "pos_encoding_type": "sin",
        "qkv_mode": "standard",
        "min_freq": 2,
        "max_src_len": 128,
        "max_tgt_len": 128,
        "data_root": ".data",
        "local_dataset_repo": "",
        "seed": 42,
    }
    if args.run_dir:
        cfg_path = Path(args.run_dir).resolve() / "config.json"
        if cfg_path.is_file():
            base_cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    merged = _merge_config(base_cfg, ckpt.get("config"))
    # 必须与保存权重时的 PE 结构一致，否则 ``pos_enc.pe`` 形状对不上
    train_pe = str(merged.get("pos_encoding_type", "sin"))

    auto_local = _SCRIPT_DIR / "7c37f-main" / "dataset-master"
    local_repo = (merged.get("local_dataset_repo") or "").strip() or args.local_repo.strip()
    if not local_repo and auto_local.is_dir():
        local_repo = str(auto_local)
    train_mod.set_seed(int(merged["seed"]))
    train_pairs, _, test_pairs = train_mod.load_multi30k_pairs(
        merged["data_root"],
        local_repo=local_repo or None,
    )
    tok_de, tok_en = train_mod.build_tokenizers()
    vocab_src, vocab_tgt = train_mod.build_vocabs(
        train_pairs, tok_de, tok_en, int(merged["min_freq"])
    )
    pad_idx = vocab_src[train_mod.PAD_TOK]
    itos = train_mod._vocab_itos(vocab_src)

    if args.de.strip():
        de_text = args.de.strip()
    else:
        idx = 0 if args.test_index < 0 else args.test_index
        if idx >= len(test_pairs):
            raise SystemExit(f"test_index={idx} 超出测试集大小 {len(test_pairs)}")
        de_text = test_pairs[idx][0]

    src_ids = train_mod.numericalize_src(de_text, tok_de, vocab_src, int(merged["max_src_len"]))
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = make_pad_mask(src, pad_idx).to(device)

    state = ckpt["model_state_dict"]
    src_vs = state["src_embed.weight"].shape[0]
    tgt_vs = state["tgt_embed.weight"].shape[0]

    layer_idx = args.layer
    if layer_idx < 0:
        layer_idx = int(merged["num_layers"]) - 1

    out_base = Path(args.out).resolve()
    stem, suf = out_base.stem, out_base.suffix
    if not suf:
        suf = ".png"
    parent = out_base.parent

    cfg_load = dict(merged)
    cfg_load["pos_encoding_type"] = train_pe
    model = build_transformer_from_config(cfg_load, src_vs, tgt_vs).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # (文件名后缀, encoding_type 覆盖, 图标题说明)
    modes: List[Tuple[str, str, str]] = [
        (train_pe, train_pe, f"Inference: {train_pe} PE (as trained)"),
    ]
    if train_pe == "sin":
        modes.append(
            ("none", "none", "Inference: PE off (encoding_type=none, same weights)")
        )
    else:
        print(
            "提示：当前 checkpoint 为无 PE 训练；``pos_enc.pe`` 未分配正弦表，"
            "无法在不大改模块的前提下安全叠加 sin PE，故仅输出 none 一种热力图。"
        )

    print("德语原句:", de_text)
    print("Tokenizer IDs 长度:", len(src_ids), "| layer:", layer_idx, "| device:", device)

    for file_tag, enc_type, pe_label in modes:
        model.pos_enc.encoding_type = enc_type  # type: ignore[assignment]
        attn = encoder_self_attention_weights(model, src, src_mask, layer_idx)
        a0 = attn[0]  # [H, L, L]
        valid_len = _src_valid_length(src[0], pad_idx)
        tokens = _tokens_from_ids(src_ids, itos)
        md, mo, ratio = diagonal_concentration(a0, valid_len)
        ent = attention_entropy_nats(a0, valid_len)
        tr_ratio = trace_mass_ratio(a0, valid_len)
        print(
            f"[{file_tag}] mean(attn[i,i])={md:.4f}  mean(off-diag)={mo:.4f}  diag/off={ratio:.3f} | "
            f"Tr/sum={tr_ratio:.4f} | Attention Entropy={ent:.4f} nats"
        )

        title = f"Encoder self-attn layer {layer_idx} | {pe_label}\n{de_text[:80]}"
        if args.attn_colorbar == "fixed01":
            title += f"\n(colorbar fixed [{args.attn_vmin}, {args.attn_vmax}] viridis)"
        out_path = parent / f"{stem}_{file_tag}{suf}"
        plot_multihead_heatmap(
            a0,
            tokens,
            title,
            out_path,
            valid_len=valid_len,
            attention_entropy_nats=ent,
            colorbar_mode=args.attn_colorbar,
            attn_vmin=float(args.attn_vmin),
            attn_vmax=float(args.attn_vmax),
        )
        print("  -> 已保存", out_path)

    print(
        "\n说明：sin→none 的第二条曲线为**同一组权重**下仅关闭位置项的前向，用于观察注意力是否更依赖"
        "「相对位置/对角邻域」等结构；与「单独训练的无 PE 模型」不是同一对照实验。"
    )
    if len(modes) >= 2 and modes[0][0] == "sin":
        print(
            "\n对比提示：若 ``none`` 行的 Attention Entropy 高于 ``sin`` 行，与脚本顶部注释一致——"
            "关闭 PE 后键维分布更弥散，不确定性增大。"
        )


if __name__ == "__main__":
    main()
