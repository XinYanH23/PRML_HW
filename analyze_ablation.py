#!/usr/bin/env python3
"""
对比 Baseline 与「无残差」消融的 metrics.csv（前 4 个 epoch），输出 Markdown 表格与结论句。

默认：
  --baseline-csv: runs/baseline_reference_metrics.csv（与 BASELINE.md 一致的参考落盘）
  --ablation-csv: runs/run_1778600084145839068/metrics.csv

用法::

    cd hw_4
    python analyze_ablation.py
    python analyze_ablation.py --out-md ablation_table.md
    python analyze_ablation.py --out-fig report/fig_ablation_ep1_4.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


# 2.1 叠图 / 单 run 曲线图底注（与《说明.md》一致）
FIG_FOOTNOTE_21 = (
    "Both runs: experiment_mode, 50% train sentences per epoch (fixed seed). "
    "Compare sin PE vs no PE only; do not mix with full-data baseline on the same axis."
)


def read_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 metrics.csv: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first_n_epochs(rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        try:
            ep = int(r["epoch"])
        except (KeyError, ValueError):
            continue
        if 1 <= ep <= n:
            out.append(r)
    out.sort(key=lambda r: int(r["epoch"]))
    return out


def loss_slope_e1_e4(losses: List[float]) -> float:
    """按报告定义：$(Loss_{E1} - Loss_{E4}) / 4$。"""
    if len(losses) < 4:
        raise ValueError("需要至少 4 个 epoch 的 loss")
    return (losses[0] - losses[3]) / 4.0


def fmt_float(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _style_line_plots() -> None:
    """高对比度折线图样式（报告用图）。"""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "lines.linewidth": 2.75,
            "lines.markersize": 9,
            "axes.grid": True,
            "grid.alpha": 0.55,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.facecolor": "#f4f6f8",
            "axes.edgecolor": "#263238",
            "axes.linewidth": 1.1,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_ablation_figure(
    base_4: List[Dict[str, str]],
    abl_4: List[Dict[str, str]],
    none_pe_rows: Optional[Sequence[Dict[str, str]]],
    out_path: Path,
) -> None:
    """前若干 epoch：train loss / valid BLEU 对比曲线（可选叠加无 PE 组）。"""
    import matplotlib.pyplot as plt

    _style_line_plots()
    # 高饱和、易区分的配色（左轴 loss / 右轴 BLEU 各一组色相）
    C_LOSS_B, C_LOSS_NR, C_LOSS_PE = "#0d47a1", "#c62828", "#1b5e20"
    C_BLEU_B, C_BLEU_NR, C_BLEU_PE = "#e65100", "#6a1b9a", "#00695c"

    epochs = [int(r["epoch"]) for r in base_4]
    loss_b = [float(r["train_loss_token"]) for r in base_4]
    loss_a = [float(r["train_loss_token"]) for r in abl_4]
    bleu_b = [float(r["valid_bleu_x100"]) for r in base_4]
    bleu_a = [float(r["valid_bleu_x100"]) for r in abl_4]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"train loss / token", color=C_LOSS_B)
    ax1.plot(epochs, loss_b, "o-", color=C_LOSS_B, label="Baseline (residual+sin PE)")
    ax1.plot(epochs, loss_a, "s--", color=C_LOSS_NR, label="no residual (sin PE)")
    if none_pe_rows:
        ne = [int(r["epoch"]) for r in none_pe_rows]
        loss_n = [float(r["train_loss_token"]) for r in none_pe_rows]
        ax1.plot(ne, loss_n, "^-", color=C_LOSS_PE, label="no PE (residual, metrics)")
    ax1.tick_params(axis="y", labelcolor=C_LOSS_B, width=1.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"valid BLEU ($\times 100$)", color=C_BLEU_B)
    ax2.plot(epochs, bleu_b, "o-", color=C_BLEU_B, label="BLEU Baseline")
    ax2.plot(epochs, bleu_a, "s--", color=C_BLEU_NR, label="BLEU no residual")
    if none_pe_rows:
        bleu_n = [float(r["valid_bleu_x100"]) for r in none_pe_rows]
        ax2.plot(ne, bleu_n, "^-", color=C_BLEU_PE, label="BLEU no PE")
    ax2.tick_params(axis="y", labelcolor=C_BLEU_B, width=1.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labels + labels2,
        loc="center right",
        fontsize=8,
        framealpha=0.95,
        edgecolor="#37474f",
    )
    fig.suptitle("Ablation 2.3: first 4 epochs (metrics.csv)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_baseline_long_monitor_figure(
    base_rows: List[Dict[str, str]],
    out_path: Path,
    *,
    max_epoch: int = 15,
) -> None:
    """Baseline 全训练期 train loss / valid BLEU（来自 reference metrics.csv）。"""
    import matplotlib.pyplot as plt

    _style_line_plots()
    C_LOSS, C_BLEU, C_BAND = "#01579a", "#bf360c", "#ffcdd2"

    rows = [r for r in base_rows if 1 <= int(r["epoch"]) <= max_epoch]
    rows.sort(key=lambda r: int(r["epoch"]))
    if len(rows) < 3:
        raise ValueError("Baseline metrics 行数不足")
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss_token"]) for r in rows]
    bleu = [float(r["valid_bleu_x100"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.4, 4.4))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"train loss / token", color=C_LOSS)
    ax1.plot(epochs, loss, "o-", color=C_LOSS, markersize=8, label="train loss/token")
    ax1.axvspan(12, 15, alpha=0.35, color=C_BAND, label="late rebound (ep 12–15)")
    ax1.tick_params(axis="y", labelcolor=C_LOSS)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"valid BLEU ($\times 100$)", color=C_BLEU)
    ax2.plot(epochs, bleu, "D-", color=C_BLEU, markersize=7, label="valid BLEU")
    ax2.tick_params(axis="y", labelcolor=C_BLEU)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        edgecolor="#37474f",
    )
    fig.suptitle("Baseline (sin PE + residual): training monitor", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_single_metrics_figure(
    rows: List[Dict[str, str]],
    out_path: Path,
    *,
    title: str,
    footnote: str = "",
) -> None:
    """单条实验曲线：train loss + valid BLEU（双 y）。"""
    import matplotlib.pyplot as plt

    _style_line_plots()
    C_LOSS, C_BLEU = "#1565c0", "#b71c1c"

    rows = sorted(rows, key=lambda r: int(r["epoch"]))
    if len(rows) < 2:
        raise ValueError("至少需要 2 行 metrics")
    epochs = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss_token"]) for r in rows]
    bleu = [float(r["valid_bleu_x100"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"train loss / token", color=C_LOSS)
    ax1.plot(epochs, loss, "o-", color=C_LOSS, label="train loss/token")
    ax1.tick_params(axis="y", labelcolor=C_LOSS)

    if footnote:
        n_ep_max = max(epochs)
        ax_top = ax1.twiny()
        ax_top.set_xlim(ax1.get_xlim())
        ax_top.set_xticks(epochs)
        ax_top.set_xticklabels([f"{100 * int(e) / n_ep_max:.0f}%" for e in epochs])
        ax_top.set_xlabel("Relative training progress (% of plotted max epoch)")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"valid BLEU ($\times 100$)", color=C_BLEU)
    ax2.plot(epochs, bleu, "s-", color=C_BLEU, label="valid BLEU")
    ax2.tick_params(axis="y", labelcolor=C_BLEU)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        edgecolor="#37474f",
    )
    fig.suptitle(title, fontsize=10)
    if footnote:
        fig.tight_layout(rect=[0, 0.18, 1, 0.94])
        fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=7.5, color="#263238")
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_pe_paired_figure(
    sin_rows: List[Dict[str, str]],
    none_rows: List[Dict[str, str]],
    out_path: Path,
    *,
    footnote: str = "",
) -> None:
    """2.1：同数据量（50%）下 sin PE vs 无 PE，单变量为位置编码。"""
    import matplotlib.pyplot as plt

    _style_line_plots()
    C_LS, C_LN = "#1565c0", "#c62828"
    C_BS, C_BN = "#2e7d32", "#6a1b9a"

    n = min(len(sin_rows), len(none_rows))
    if n < 2:
        raise ValueError("两组 metrics 至少需要 2 个对齐的 epoch")
    epochs = [int(sin_rows[i]["epoch"]) for i in range(n)]
    loss_s = [float(sin_rows[i]["train_loss_token"]) for i in range(n)]
    loss_n = [float(none_rows[i]["train_loss_token"]) for i in range(n)]
    bleu_s = [float(sin_rows[i]["valid_bleu_x100"]) for i in range(n)]
    bleu_n = [float(none_rows[i]["valid_bleu_x100"]) for i in range(n)]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"train loss / token", color=C_LS)
    ax1.plot(epochs, loss_s, "o-", color=C_LS, label="50% data + sin PE")
    ax1.plot(epochs, loss_n, "s--", color=C_LN, label="50% data + no PE")
    ax1.tick_params(axis="y", labelcolor=C_LS)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"valid BLEU ($\times 100$)", color=C_BS)
    ax2.plot(epochs, bleu_s, "o-", color=C_BS, label="BLEU sin PE")
    ax2.plot(epochs, bleu_n, "s--", color=C_BN, label="BLEU no PE")
    ax2.tick_params(axis="y", labelcolor=C_BS)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
        edgecolor="#37474f",
    )
    fig.suptitle("2.1 PE ablation: matched 50% data per epoch", fontsize=10)
    if footnote:
        fig.tight_layout(rect=[0, 0.14, 1, 0.94])
        fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=7.5, color="#263238")
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_none_pe_early_figure(
    base_rows: List[Dict[str, str]],
    none_rows: List[Dict[str, str]],
    out_path: Path,
    *,
    subtitle: str = "",
    footnote: str = "",
) -> None:
    """（旧）全量 Baseline ref 与无 PE 叠图；正文请改用 save_pe_paired_figure。"""
    import matplotlib.pyplot as plt

    _style_line_plots()
    C_LB, C_LN = "#0d47a1", "#e65100"
    C_BB, C_BN = "#4a148c", "#006064"

    n = min(len(base_rows), len(none_rows))
    if n < 2:
        raise ValueError("两组 metrics 至少需要 2 个对齐的 epoch")
    epochs = [int(base_rows[i]["epoch"]) for i in range(n)]
    loss_b = [float(base_rows[i]["train_loss_token"]) for i in range(n)]
    loss_n = [float(none_rows[i]["train_loss_token"]) for i in range(n)]
    bleu_b = [float(base_rows[i]["valid_bleu_x100"]) for i in range(n)]
    bleu_n = [float(none_rows[i]["valid_bleu_x100"]) for i in range(n)]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"train loss / token", color=C_LB)
    ax1.plot(epochs, loss_b, "o-", color=C_LB, label="Baseline sin PE (ref.)")
    ax1.plot(epochs, loss_n, "s--", color=C_LN, label="no PE (this run)")
    ax1.tick_params(axis="y", labelcolor=C_LB)

    n_ep_max = max(epochs) if epochs else 1
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(epochs)
    ax_top.set_xticklabels([f"{100 * int(e) / n_ep_max:.0f}%" for e in epochs])
    ax_top.set_xlabel("Relative training progress (% of plotted max epoch)")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"valid BLEU ($\times 100$)", color=C_BB)
    ax2.plot(epochs, bleu_b, "o-", color=C_BB, label="BLEU ref.")
    ax2.plot(epochs, bleu_n, "s--", color=C_BN, label="BLEU no PE")
    ax2.tick_params(axis="y", labelcolor=C_BB)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        fontsize=8,
        ncol=1,
        framealpha=0.95,
        edgecolor="#37474f",
    )
    ttl = "2.1 Position encoding: early train loss / valid BLEU"
    if subtitle:
        ttl = f"{ttl}\n{subtitle}"
    fig.suptitle(ttl, fontsize=10)
    if footnote:
        fig.tight_layout(rect=[0, 0.16, 1, 0.94])
        fig.text(0.5, 0.02, footnote, ha="center", va="bottom", fontsize=7.5, color="#263238")
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_table_rows(
    name: str,
    slice_rows: List[Dict[str, str]],
) -> Tuple[List[str], float, float, float]:
    losses = [float(r["train_loss_token"]) for r in slice_rows]
    bleus_x100 = [float(r["valid_bleu_x100"]) for r in slice_rows]
    slope = loss_slope_e1_e4(losses)
    return (
        [
            name,
            fmt_float(losses[0]),
            fmt_float(losses[1]),
            fmt_float(losses[2]),
            fmt_float(losses[3]),
            fmt_float(bleus_x100[3]),
            fmt_float(slope, nd=6),
        ],
        losses[3],
        bleus_x100[3],
        slope,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="2.3 残差消融：metrics 对比与 Markdown 表")
    ap.add_argument(
        "--baseline-csv",
        type=str,
        default=str(_script_dir() / "runs" / "baseline_reference_metrics.csv"),
        help="Baseline（N=6, use_residual=True）metrics.csv",
    )
    ap.add_argument(
        "--ablation-csv",
        type=str,
        default=str(_script_dir() / "runs" / "run_1778600084145839068" / "metrics.csv"),
        help="无残差组 metrics.csv",
    )
    ap.add_argument(
        "--out-md",
        type=str,
        default="",
        help="若指定，将 Markdown 表格写入该文件",
    )
    ap.add_argument(
        "--out-fig",
        type=str,
        default="",
        help="若指定，保存前 4 epoch 的 loss/BLEU 对比图（PNG）",
    )
    ap.add_argument(
        "--none-pe-csv",
        type=str,
        default="",
        help="可选：无位置编码组 metrics.csv，用于在同一张图上叠加前期曲线",
    )
    ap.add_argument(
        "--out-fig-2-1",
        type=str,
        default="",
        help="若指定且提供 --none-pe-csv：绘制 Baseline 与无 PE 组前 --early-epochs 个 epoch 的 loss/BLEU 对照图",
    )
    ap.add_argument(
        "--early-epochs",
        type=int,
        default=5,
        help="与 --out-fig-2-1 配合：对齐前多少个 epoch（默认 5）",
    )
    ap.add_argument(
        "--out-fig-monitor",
        type=str,
        default="",
        help="若指定：根据 Baseline reference CSV 绘制 1–15 epoch 训练监控图",
    )
    ap.add_argument(
        "--out-fig-none-pe-only",
        type=str,
        default="",
        help="若指定且提供 --none-pe-csv：仅绘制该 run 的 loss/BLEU（不叠 Baseline）",
    )
    ap.add_argument(
        "--half-sin-pe-csv",
        type=str,
        default="",
        help="50%% data + sin PE control metrics.csv (pair with --none-pe-csv)",
    )
    ap.add_argument(
        "--out-fig-2-1-paired",
        type=str,
        default="",
        help="若指定且同时提供 --half-sin-pe-csv 与 --none-pe-csv：绘制同数据量 PE 对照图",
    )
    args = ap.parse_args()

    base_path = Path(args.baseline_csv).resolve()
    abl_path = Path(args.ablation_csv).resolve()

    base_all = read_metrics(base_path)
    abl_all = read_metrics(abl_path)

    base_4 = first_n_epochs(base_all, 4)
    abl_4 = first_n_epochs(abl_all, 4)
    if len(base_4) < 4:
        raise SystemExit(f"Baseline CSV 中 epoch 1–4 不完整: {base_path}")
    if len(abl_4) < 4:
        raise SystemExit(
            f"消融 CSV 中 epoch 1–4 不完整（当前 {len(abl_4)} 行）: {abl_path}\n"
            "无残差实验若早停于 epoch<4，无法按定义计算 E1–E4 斜率；请补跑或放宽早停。"
        )

    header = [
        "设置（$N=6$）",
        "$\\mathcal{L}^{\\mathrm{tok}}_{E1}$",
        "$\\mathcal{L}^{\\mathrm{tok}}_{E2}$",
        "$\\mathcal{L}^{\\mathrm{tok}}_{E3}$",
        "$\\mathcal{L}^{\\mathrm{tok}}_{E4}$",
        "Valid BLEU$_{E4}$ (×100)",
        "Loss 降幅斜率 $\\frac{\\mathcal{L}_{E1}-\\mathcal{L}_{E4}}{4}$",
    ]
    row_b, _, bleu_b4, slope_b = build_table_rows("Baseline（有残差）", base_4)
    row_a, _, bleu_a4, slope_a = build_table_rows("无残差（本仓库 run）", abl_4)

    lines = [
        "### 2.3 残差消融：前 4 Epoch 收敛对比（`metrics.csv`）",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        "| " + " | ".join(row_b) + " |",
        "| " + " | ".join(row_a) + " |",
        "",
        "**一句话（可写入报告）**：在六层堆叠下，残差提供近似恒等映射的优化捷径，使梯度能稳定流过深层；",
        "关闭残差后子层输出需从零逼近恒等分支，信号在前几轮难以有效下传，表现为 **train loss/token 下降极慢**、",
        "**验证 BLEU 长期接近随机水平**，与「深层几乎无法学习」的观测一致。",
        "",
        f"- Baseline：E4 BLEU≈{bleu_b4:.2f}，斜率≈{slope_b:.4f}（loss/token 每 epoch 平均下降量，按 $(\\mathcal{{L}}_{{E1}}-\\mathcal{{L}}_{{E4}})/4$）。",
        f"- 无残差：E4 BLEU≈{bleu_a4:.4f}，斜率≈{slope_a:.4f}（不足 Baseline 的一半量级时可强调「初期几乎学不动」）。",
        "",
    ]
    md = "\n".join(lines)
    print(md)
    if args.out_md:
        out_p = Path(args.out_md).resolve()
        out_p.write_text(md + "\n", encoding="utf-8")
        print(f"\n已写入: {out_p}")

    if args.out_fig:
        none_rows: Optional[List[Dict[str, str]]] = None
        if args.none_pe_csv.strip():
            none_all = read_metrics(Path(args.none_pe_csv).resolve())
            none_rows = first_n_epochs(none_all, 4)
            if len(none_rows) < 2:
                print(f"警告: --none-pe-csv 中 epoch 1–4 不足，图上将省略无 PE 曲线: {args.none_pe_csv}")
                none_rows = None
        fig_p = Path(args.out_fig).resolve()
        save_ablation_figure(base_4, abl_4, none_rows, fig_p)
        print(f"\n已保存对比图: {fig_p}")

    if args.out_fig_2_1.strip() and args.none_pe_csv.strip():
        ne = int(args.early_epochs)
        base_n = first_n_epochs(base_all, ne)
        none_all = read_metrics(Path(args.none_pe_csv).resolve())
        none_n = first_n_epochs(none_all, ne)
        p21 = Path(args.out_fig_2_1).resolve()
        save_none_pe_early_figure(
            base_n,
            none_n,
            p21,
            subtitle="(aligned by epoch index)",
            footnote=FIG_FOOTNOTE_21,
        )
        print(f"\n已保存 2.1 前期曲线: {p21}")

    if args.out_fig_monitor.strip():
        pm = Path(args.out_fig_monitor).resolve()
        save_baseline_long_monitor_figure(base_all, pm)
        print(f"\n已保存训练监控图: {pm}")

    if args.out_fig_none_pe_only.strip() and args.none_pe_csv.strip():
        ne = int(args.early_epochs)
        none_all_only = read_metrics(Path(args.none_pe_csv).resolve())
        none_only = first_n_epochs(none_all_only, ne)
        po = Path(args.out_fig_none_pe_only).resolve()
        save_single_metrics_figure(
            none_only,
            po,
            title="2.1 No positional encoding (this run): early loss / BLEU",
            footnote=FIG_FOOTNOTE_21,
        )
        print(f"\n已保存无 PE 单曲线: {po}")

    if args.out_fig_2_1_paired.strip():
        if not args.half_sin_pe_csv.strip() or not args.none_pe_csv.strip():
            raise SystemExit("--out-fig-2-1-paired 需要同时提供 --half-sin-pe-csv 与 --none-pe-csv")
        ne = int(args.early_epochs)
        sin_all = read_metrics(Path(args.half_sin_pe_csv).resolve())
        none_all_p = read_metrics(Path(args.none_pe_csv).resolve())
        sin_n = first_n_epochs(sin_all, ne)
        none_n_p = first_n_epochs(none_all_p, ne)
        pp = Path(args.out_fig_2_1_paired).resolve()
        save_pe_paired_figure(sin_n, none_n_p, pp, footnote=FIG_FOOTNOTE_21)
        print(f"\n已保存 2.1 配对对照图: {pp}")


if __name__ == "__main__":
    main()
