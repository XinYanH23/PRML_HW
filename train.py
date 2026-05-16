"""
Multi30k 德译英训练脚本：torchtext + spacy 分词、词表、DataLoader、掩码、BLEU。

**自动落盘**：每次运行创建 ``runs/run_<时间戳纳秒>/``，写入 ``metrics.csv``、``train.log``、
``config.json``、验证 BLEU 最高时的 ``best_model.pt`` + ``best_model_meta.json``，以及结束时的 ``summary.json``。
**早停**：``early_stopping_patience``（默认 3）内验证 BLEU 无提升则停止；``--no-early-stopping`` 关闭。
**experiment_mode**：``config`` 或 ``--experiment`` 开启时，``epochs=5``（可用 ``--epochs`` 覆盖）、训练集随机 **50%%**（固定 seed）；与 ``--quick`` 同时存在时以 ``--quick`` 为准。

依赖（建议 conda/venv）:
  pip install torch torchtext spacy nltk tqdm
  python -m spacy download de_core_news_sm
  python -m spacy download en_core_web_sm

若下载出现 ``SSL: UNEXPECTED_EOF``：脚本会依次尝试 **requests**、**curl**、带重试的 **urllib** 及 **jsDelivr 镜像**；仍失败时可浏览器手动下载 tar.gz 放到 ``<data_root>/Multi30k/`` 后再运行（见下方 ``_MULTI30K_URL_CANDIDATES`` 中的 URL）。

**CUDA 警告（driver too old）**：升级 NVIDIA 驱动，或安装与当前驱动匹配的 PyTorch 版本；脚本会在无法分配 GPU 张量时退回 CPU。

说明：``from torchtext.datasets import Multi30k`` 在旧版 torchtext 中是 ``TranslationDataset`` 子类，
不能用 ``root= / split=``；本脚本优先尝试新版 DataPipe（需 ``pip install torchdata``），
否则自动使用内置 HTTP 下载与解压逻辑。
"""
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import inspect
import math
import random
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from transformer_model import (
    LabelSmoothingLoss,
    NoamOpt,
    build_transformer_from_config,
    make_pad_mask,
    make_subsequent_mask,
)

# -----------------------------------------------------------------------------
# 实验配置：改字段即可做消融（与 notebook 第二部分对应）
# -----------------------------------------------------------------------------
config: Dict[str, Any] = {
    "num_layers": 6,
    "d_model": 512,
    "num_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_pe_len": 5000,
    # 2.3 残差消融：False 时子层为 LayerNorm(Dropout(Sublayer(x)))，无 x+ 捷径（见 transformer_model 注释）
    "use_residual": False,
    # 2.3 单变量：与 Baseline 一致用 sin PE；2.1 无 PE 请 --pos-encoding none
    "pos_encoding_type": "sin",
    "qkv_mode": "standard",  # 2.2："shared" 在自注意力上用单一线性投 QKV
    # 2.3：对比前 3 epoch loss 斜率默认跑 5 epoch（可用 --epochs 3）
    "batch_size": 128,
    "epochs": 5,
    "label_smoothing": 0.1,
    "warmup_steps": 4000,
    "noam_factor": 2,
    "min_freq": 2,
    "max_src_len": 128,
    "max_tgt_len": 128,
    "data_root": ".data",
    # 若指向 Multi30k/WMT16 task1 仓库根目录（含 data/task1/tok/），则**不联网**直接读本地 .tok 文件
    "local_dataset_repo": "",  # 留空则自动探测 hw_4/7c37f-main/dataset-master
    "seed": 42,
    "num_workers": 2,
    # 若驱动与 PyTorch CUDA 不匹配，会在 resolve_device 中自动退回 CPU
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # 对比实验加速：True 时 epochs=5，且训练集仅随机取 50%（固定 seed，便于看前若干轮 loss 差异）
    "experiment_mode": False,
    # 早停：验证 BLEU 连续 patience 个 epoch 未提升则停止（0 表示关闭）
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.0,
    # 日志与最佳模型输出目录（每次运行会建子目录 run_YYYYMMDD_HHMMSS）
    "runs_dir": "runs",
}

PAD_TOK = "<pad>"
SOS_TOK = "<sos>"
EOS_TOK = "<eos>"
UNK_TOK = "<unk>"


def _try_tqdm(iterable: Iterable, **kwargs: Any):
    """优先使用 tqdm；未安装时退化为普通迭代并提示一次。"""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        if not getattr(_try_tqdm, "_warned", False):
            print("提示: 安装 tqdm 可显示进度条与剩余时间估计: pip install tqdm")
            _try_tqdm._warned = True  # type: ignore[attr-defined]
        return iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str) -> torch.device:
    """避免「驱动过旧 / CUDA 初始化失败」时仍强行使用 cuda。"""
    if preference != "cuda":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception as exc:  # noqa: BLE001
        print(f"警告: CUDA 不可用 ({exc})，改用 CPU。")
        return torch.device("cpu")


# 每个 split 多个 URL：优先 jsdelivr（部分网络下比 raw.githubusercontent.com 更稳）
_MULTI30K_URL_CANDIDATES: Dict[str, List[str]] = {
    "train": [
        "https://cdn.jsdelivr.net/gh/neychev/small_DL_repo@master/datasets/Multi30k/training.tar.gz",
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz",
    ],
    "valid": [
        "https://cdn.jsdelivr.net/gh/neychev/small_DL_repo@master/datasets/Multi30k/validation.tar.gz",
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz",
    ],
    "test": [
        "https://cdn.jsdelivr.net/gh/neychev/small_DL_repo@master/datasets/Multi30k/mmt16_task1_test.tar.gz",
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz",
    ],
}
_MULTI30K_PREFIX = {"train": "train", "valid": "val", "test": "test"}


def _download_via_requests(url: str, dest: Path) -> bool:
    try:
        import requests
    except ImportError:
        return False
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 300),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return dest.is_file() and dest.stat().st_size > 1024
    except Exception:
        if dest.is_file():
            dest.unlink(missing_ok=True)
        return False


def _download_via_curl(url: str, dest: Path) -> bool:
    curl = shutil.which("curl")
    if not curl:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                curl,
                "-fL",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "30",
                "-o",
                str(dest),
                url,
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
        return dest.is_file() and dest.stat().st_size > 1024
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        if dest.is_file():
            dest.unlink(missing_ok=True)
        return False


def _download_via_wget(url: str, dest: Path) -> bool:
    wget = shutil.which("wget")
    if not wget:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [wget, "-q", "--tries=5", "--waitretry=2", "-O", str(dest), url],
            check=True,
            capture_output=True,
            timeout=600,
        )
        return dest.is_file() and dest.stat().st_size > 1024
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        if dest.is_file():
            dest.unlink(missing_ok=True)
        return False


def _download_via_urllib(url: str, dest: Path, attempts: int = 5) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PRML-hw4/1.0)"})
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
            if dest.is_file() and dest.stat().st_size > 1024:
                return True
        except (urllib.error.URLError, ssl.SSLError, OSError, TimeoutError) as e:
            if dest.is_file():
                dest.unlink(missing_ok=True)
            if attempt == attempts:
                print(f"  urllib 最终失败 ({attempt} 次): {e}")
            else:
                time.sleep(min(2 * attempt, 10))
    return False


def _download_file(url_candidates: Sequence[str], dest: Path) -> None:
    """依次尝试多种方式与多个镜像，解决 SSL EOF / 连接中断等环境问题。"""
    if dest.is_file() and dest.stat().st_size > 1024:
        return

    last_err: str | None = None
    for url in url_candidates:
        print(f"Downloading {dest.name} from:\n  {url}")
        if _download_via_requests(url, dest):
            return
        if _download_via_curl(url, dest):
            return
        if _download_via_wget(url, dest):
            return
        if _download_via_urllib(url, dest):
            return
        last_err = url

    raise RuntimeError(
        "无法下载 Multi30k 数据（已尝试 requests / curl / wget / urllib 及所有镜像）。\n"
        "可选解决办法：\n"
        "  1) pip install requests certifi 后重试；\n"
        "  2) 确认系统时间正确、代理/VPN 或换网络；\n"
        "  3) 用浏览器或 curl 手动下载上述 URL 中的 tar.gz，保存为:\n"
        f"     {dest}\n"
        "     然后重新运行 python train.py（已存在且大于 1KB 会跳过下载）。\n"
        f"  最后失败的 URL: {last_err}"
    )


def _read_parallel_lines(de_path: Path, en_path: Path) -> List[Tuple[str, str]]:
    with de_path.open(encoding="utf-8") as fde, en_path.open(encoding="utf-8") as fen:
        des = [ln.strip() for ln in fde]
        ens = [ln.strip() for ln in fen]
    if len(des) != len(ens):
        raise ValueError(f"行数不一致: {de_path} ({len(des)}) vs {en_path} ({len(ens)})")
    return list(zip(des, ens))


def _load_multi30k_from_wmt_task1_tok(repo_root: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    从 WMT16 multimodal task1 仓库（如 dataset-master）读取已 Moses 分词的 de/en 平行句。
    路径约定：``<repo>/data/task1/tok/{train,val,test_2016_flickr}.lc.norm.tok.{de,en}``
    """
    tok = repo_root / "data" / "task1" / "tok"
    paths = {
        "train": (tok / "train.lc.norm.tok.de", tok / "train.lc.norm.tok.en"),
        "val": (tok / "val.lc.norm.tok.de", tok / "val.lc.norm.tok.en"),
        "test": (tok / "test_2016_flickr.lc.norm.tok.de", tok / "test_2016_flickr.lc.norm.tok.en"),
    }
    out: List[List[Tuple[str, str]]] = []
    for name, (de_p, en_p) in paths.items():
        if not de_p.is_file() or not en_p.is_file():
            raise FileNotFoundError(
                f"本地数据缺少文件: {de_p} 或 {en_p}。请确认 local_dataset_repo 指向含 data/task1/tok 的仓库根目录。"
            )
        out.append(_read_parallel_lines(de_p, en_p))
        print(f"  已加载本地 {name}: {len(out[-1])} 句对 ({de_p.name})")
    return out[0], out[1], out[2]


def _load_multi30k_from_tarballs(root: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """不依赖 torchtext Multi30k 符号：直接下载 tar.gz 并读 train/val/test 平行句。"""
    base = root / "Multi30k"
    base.mkdir(parents=True, exist_ok=True)
    out: List[List[Tuple[str, str]]] = []
    for split in ("train", "valid", "test"):
        urls = _MULTI30K_URL_CANDIDATES[split]
        tar_name = urls[0].rsplit("/", maxsplit=1)[-1]
        tar_path = base / tar_name
        _download_file(urls, tar_path)
        extract_dir = base / split / "extracted"
        marker = extract_dir / ".extract_ok"
        if not marker.is_file():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                if sys.version_info >= (3, 12):
                    tar.extractall(extract_dir, filter="data")
                else:
                    tar.extractall(extract_dir)
            marker.write_text("ok", encoding="utf-8")
        prefix = _MULTI30K_PREFIX[split]
        de_files = sorted(extract_dir.rglob(f"{prefix}.de"))
        en_files = sorted(extract_dir.rglob(f"{prefix}.en"))
        if not de_files or not en_files:
            raise FileNotFoundError(
                f"在 {extract_dir} 下未找到 {prefix}.de / {prefix}.en，"
                f"请删除该目录后重试或检查 torchtext 缓存。"
            )
        out.append(_read_parallel_lines(de_files[0], en_files[0]))
    return out[0], out[1], out[2]


def load_multi30k_pairs(
    root: str,
    local_repo: str | None = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    加载 Multi30k (de, en)。

    0) 若 ``local_repo`` 非空且目录存在，从 ``data/task1/tok`` 读本地分词文件（不下载）。
    1) 若存在 ``torchtext.datasets.multi30k.Multi30k`` **函数** 且已安装 ``torchdata``，
       使用官方 DataPipe。
    2) 否则使用内置 HTTP + tar 读取逻辑。
    """
    if local_repo:
        repo = Path(local_repo).expanduser().resolve()
        if repo.is_dir():
            print(f"使用本地数据集目录（不下载）: {repo}")
            return _load_multi30k_from_wmt_task1_tok(repo)

    root_path = Path(root).expanduser().resolve()

    try:
        m = importlib.import_module("torchtext.datasets.multi30k")
        fn = getattr(m, "Multi30k", None)
        if fn is not None and not inspect.isclass(fn):
            importlib.import_module("torchdata")
            train = list(fn(root=str(root_path), split="train", language_pair=("de", "en")))
            valid = list(fn(root=str(root_path), split="valid", language_pair=("de", "en")))
            test = list(fn(root=str(root_path), split="test", language_pair=("de", "en")))
            return train, valid, test
    except Exception as exc:
        print(f"提示: 未使用 torchtext DataPipe 加载 Multi30k（{type(exc).__name__}: {exc}），改用内置下载。")

    return _load_multi30k_from_tarballs(root_path)


class MemoryPairDataset(Dataset):
    """内存中的 (de, en) 句对。"""

    def __init__(self, pairs: Sequence[Tuple[str, str]]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


def build_tokenizers():
    from torchtext.data.utils import get_tokenizer

    de = get_tokenizer("spacy", language="de_core_news_sm")
    en = get_tokenizer("spacy", language="en_core_web_sm")
    return de, en


def build_vocabs(train_pairs, tok_de, tok_en, min_freq: int):
    from collections import Counter

    from torchtext.vocab import Vocab

    specials = [PAD_TOK, SOS_TOK, EOS_TOK, UNK_TOK]
    counter_src: Counter[str] = Counter()
    counter_tgt: Counter[str] = Counter()
    for de, en in train_pairs:
        counter_src.update(tok_de(de))
        counter_tgt.update(tok_en(en))

    vocab_src = Vocab(counter_src, min_freq=min_freq, specials=specials, specials_first=True)
    vocab_tgt = Vocab(counter_tgt, min_freq=min_freq, specials=specials, specials_first=True)
    if hasattr(vocab_src, "set_default_index"):
        vocab_src.set_default_index(vocab_src[UNK_TOK])
        vocab_tgt.set_default_index(vocab_tgt[UNK_TOK])
    return vocab_src, vocab_tgt


def _vocab_itos(vocab) -> List[str]:
    if hasattr(vocab, "get_itos"):
        return list(vocab.get_itos())
    return list(vocab.itos)


def _apply_experiment_mode(
    train_pairs: List[Tuple[str, str]],
    cfg: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """experiment_mode：epochs=5，训练集随机 50%（可复现）。"""
    n = len(train_pairs) // 2
    rng = random.Random(int(cfg["seed"]))
    idx = list(range(len(train_pairs)))
    rng.shuffle(idx)
    chosen = idx[:n]
    out = [train_pairs[i] for i in chosen]
    print(f"experiment_mode: 使用 {len(out)}/{len(train_pairs)} 条训练句（50%），epochs={cfg['epochs']}")
    return out


def _create_run_dir(script_dir: Path, cfg: Dict[str, Any]) -> Path:
    base = script_dir / str(cfg.get("runs_dir", "runs"))
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"run_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _append_csv_row(csv_path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    new_file = not csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _append_text_log(log_path: Path, line: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _save_best_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    valid_bleu: float,
    cfg: Dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "valid_bleu": valid_bleu,
        "config": {
            k: cfg[k]
            for k in (
                "num_layers",
                "d_model",
                "num_heads",
                "d_ff",
                "dropout",
                "use_residual",
                "pos_encoding_type",
                "qkv_mode",
                "batch_size",
                "label_smoothing",
                "experiment_mode",
            )
            if k in cfg
        },
    }
    torch.save(payload, path)


def numericalize_src(text: str, tok, vocab, max_len: int) -> List[int]:
    ids = [vocab[t] for t in tok(text)]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids


def numericalize_tgt(text: str, tok, vocab, max_body_len: int, sos_idx: int, eos_idx: int) -> List[int]:
    body = [vocab[t] for t in tok(text)]
    if len(body) > max_body_len:
        body = body[:max_body_len]
    return [sos_idx] + body + [eos_idx]


def make_collate_fn(
    tok_de,
    tok_en,
    vocab_src,
    vocab_tgt,
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    max_src_len: int,
    max_tgt_len: int,
):
    """在 CPU 上组 batch，避免 DataLoader 多进程 + CUDA 冲突；训练循环再 .to(device)。"""

    def collate(batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        src_ids = [numericalize_src(de, tok_de, vocab_src, max_src_len) for de, _ in batch]
        tgt_ids = [numericalize_tgt(en, tok_en, vocab_tgt, max_tgt_len, sos_idx, eos_idx) for _, en in batch]

        src_tensor = pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in src_ids],
            batch_first=True,
            padding_value=pad_idx,
        )
        tgt_full = pad_sequence(
            [torch.tensor(x, dtype=torch.long) for x in tgt_ids],
            batch_first=True,
            padding_value=pad_idx,
        )

        dec_in = tgt_full[:, :-1].contiguous()
        dec_out = tgt_full[:, 1:].contiguous()

        src_mask = make_pad_mask(src_tensor, pad_idx)
        sub = make_subsequent_mask(dec_in.size(1), device=src_tensor.device)
        trg_mask = sub | make_pad_mask(dec_in, pad_idx)

        return {
            "src": src_tensor,
            "dec_in": dec_in,
            "dec_out": dec_out,
            "src_mask": src_mask,
            "trg_mask": trg_mask,
        }

    return collate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: NoamOpt,
    pad_idx: int,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    pbar = _try_tqdm(
        loader,
        desc=f"Train {epoch}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for batch in pbar:
        src = batch["src"].to(device, non_blocking=True)
        dec_in = batch["dec_in"].to(device, non_blocking=True)
        dec_out = batch["dec_out"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        trg_mask = batch["trg_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(src, dec_in, src_mask, trg_mask)
        loss = criterion(logits, dec_out)
        loss.backward()
        optimizer.step()

        ntok = dec_out.ne(pad_idx).sum().item()
        total_loss += loss.item() * ntok
        total_tokens += ntok
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss_tok=f"{total_loss / max(total_tokens, 1):.4f}",
                lr=f"{optimizer.rate():.2e}",
                refresh=False,
            )
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
) -> torch.Tensor:
    """单样本 greedy；src: [1, src_len]。"""
    model.eval()
    device = src.device
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, dtype=torch.long, device=device) * sos_idx
    for _ in range(max_len):
        sub = make_subsequent_mask(ys.size(1), device=device)
        pad_m = make_pad_mask(ys, pad_idx)
        trg_mask = sub | pad_m
        out = model.decode(ys, memory, src_mask, trg_mask)
        logits = model.generator(out[:, -1:])
        next_tok = logits.argmax(dim=-1)
        ys = torch.cat([ys, next_tok], dim=1)
        if (next_tok == eos_idx).all():
            break
    return ys


def ids_to_tokens(ids: List[int], vocab_itos: List[str], pad_idx: int, sos_idx: int, eos_idx: int) -> List[str]:
    toks = []
    for i in ids:
        if i in (pad_idx, sos_idx):
            continue
        if i == eos_idx:
            break
        if 0 <= i < len(vocab_itos):
            toks.append(vocab_itos[i])
    return toks


def corpus_bleu_simple(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """简单 corpus BLEU（NLTK + smoothing），便于课程实验。"""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as e:
        raise ImportError("评估 BLEU 需要: pip install nltk") from e

    refs_tok = [[r] for r in references]
    smooth = SmoothingFunction().method1
    return float(corpus_bleu(refs_tok, hypotheses, smoothing_function=smooth))


@torch.no_grad()
def evaluate_bleu(
    model: nn.Module,
    pairs: Sequence[Tuple[str, str]],
    tok_de,
    tok_en,
    vocab_src,
    vocab_tgt,
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    max_src_len: int,
    max_gen_len: int,
    device: torch.device,
    max_samples: int = 500,
) -> float:
    """在句对列表上贪心解码并计算 corpus BLEU（英文明细 token）。"""
    model.eval()
    itos_src = _vocab_itos(vocab_src)
    itos_tgt = _vocab_itos(vocab_tgt)

    hyps: List[List[str]] = []
    refs: List[List[str]] = []

    subset = list(pairs)[:max_samples]
    for de, en in _try_tqdm(
        subset,
        desc="BLEU eval",
        leave=False,
        dynamic_ncols=True,
        mininterval=0.3,
    ):
        src_ids = numericalize_src(de, tok_de, vocab_src, max_src_len)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_pad_mask(src, pad_idx)
        out_ids = greedy_decode(model, src, src_mask, max_gen_len, sos_idx, eos_idx, pad_idx)
        hyp = ids_to_tokens(out_ids[0].tolist(), itos_tgt, pad_idx, sos_idx, eos_idx)
        ref = tok_en(en)
        hyps.append(hyp)
        refs.append(ref)

    return corpus_bleu_simple(refs, hyps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument(
        "--local-repo",
        type=str,
        default=None,
        help="WMT16 task1 数据仓库根目录（含 data/task1/tok），设置后不联网下载",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="小模型 + 1 epoch 冒烟（验证管线）；与 --experiment 同时存在时以 --quick 为准",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="等价于 config['experiment_mode']=True：epochs=5（可用 --epochs 覆盖）、训练集随机 50%%",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="早停：验证 BLEU 连续多少个 epoch 无提升则停止（覆盖 config early_stopping_patience）",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="关闭早停（等价 patience=0）",
    )
    parser.add_argument(
        "--pos-encoding",
        type=str,
        choices=["sin", "none"],
        default=None,
        help='覆盖 config 中 pos_encoding_type（Baseline 用 "sin"，2.1 消融用 "none"）',
    )
    parser.add_argument(
        "--use-residual",
        action="store_true",
        help="2.3 对照：强制 use_residual=True（与默认无残差消融对比）",
    )
    args = parser.parse_args()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.local_repo is not None:
        config["local_dataset_repo"] = args.local_repo
    if args.experiment:
        config["experiment_mode"] = True
    if args.patience is not None:
        config["early_stopping_patience"] = int(args.patience)
    if args.no_early_stopping:
        config["early_stopping_patience"] = 0
    if args.pos_encoding is not None:
        config["pos_encoding_type"] = args.pos_encoding
    if args.use_residual:
        config["use_residual"] = True

    if args.quick:
        config["num_layers"] = 2
        config["d_model"] = 256
        config["num_heads"] = 4
        config["d_ff"] = 1024
        config["epochs"] = 1
        config["batch_size"] = min(int(config.get("batch_size", 64)), 64)
        config["num_workers"] = 0
        config["warmup_steps"] = 400
    elif config.get("experiment_mode"):
        if args.epochs is None:
            config["epochs"] = 5

    script_dir = Path(__file__).resolve().parent
    auto_local = script_dir / "7c37f-main" / "dataset-master"
    local_repo = (config.get("local_dataset_repo") or "").strip()
    if not local_repo and auto_local.is_dir():
        local_repo = str(auto_local)
        print(f"自动使用本地数据: {local_repo}")

    run_dir = _create_run_dir(script_dir, config)
    metrics_csv = run_dir / "metrics.csv"
    text_log = run_dir / "train.log"
    best_ckpt = run_dir / "best_model.pt"
    cfg_dump = {k: v for k, v in config.items() if isinstance(v, (bool, int, float, str))}
    (run_dir / "config.json").write_text(json.dumps(cfg_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"本次运行输出目录: {run_dir}")

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))

    print("Loading Multi30k + tokenizers...")
    train_pairs, valid_pairs, test_pairs = load_multi30k_pairs(
        config["data_root"],
        local_repo=local_repo or None,
    )
    if config.get("experiment_mode") and not args.quick:
        train_pairs = _apply_experiment_mode(train_pairs, config)

    tok_de, tok_en = build_tokenizers()

    print("Building vocabs...")
    vocab_src, vocab_tgt = build_vocabs(train_pairs, tok_de, tok_en, int(config["min_freq"]))
    pad_idx = vocab_src[PAD_TOK]
    sos_idx = vocab_src[SOS_TOK]
    eos_idx = vocab_src[EOS_TOK]
    assert pad_idx == vocab_tgt[PAD_TOK]
    assert sos_idx == vocab_tgt[SOS_TOK]
    assert eos_idx == vocab_tgt[EOS_TOK]
    assert len(vocab_src) > 0 and len(vocab_tgt) > 0

    collate = make_collate_fn(
        tok_de,
        tok_en,
        vocab_src,
        vocab_tgt,
        pad_idx,
        sos_idx,
        eos_idx,
        int(config["max_src_len"]),
        int(config["max_tgt_len"]),
    )

    train_ds = MemoryPairDataset(train_pairs)
    valid_ds = MemoryPairDataset(valid_pairs)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    n_train = len(train_ds)
    bs = int(config["batch_size"])
    steps_per_epoch = math.ceil(n_train / bs)
    print(
        f"训练规模: {n_train} 句/epoch, batch_size={bs} → 每 epoch 约 {steps_per_epoch} 个 batch。"
    )
    print(
        "耗时粗估: 同一套超参下，GPU 通常每 epoch 约数秒～数分钟；CPU（尤其 base 512/6 层）"
        "常见每 epoch 十分钟～数小时。进度条中的 it/s 与 ETA 为实测估计。"
    )

    patience = int(config.get("early_stopping_patience", 0))
    min_delta = float(config.get("early_stopping_min_delta", 0.0))
    if patience > 0:
        print(f"早停: patience={patience}, min_delta={min_delta}（验证 BLEU 为 0–1 尺度上的增量）")

    model = build_transformer_from_config(
        config,
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
    ).to(device)

    criterion = LabelSmoothingLoss(
        len(vocab_tgt), padding_idx=pad_idx, smoothing=float(config["label_smoothing"])
    )
    inner_opt = torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0
    )
    optimizer = NoamOpt(
        int(config["d_model"]),
        float(config["noam_factor"]),
        int(config["warmup_steps"]),
        inner_opt,
    )

    print("Config:", {k: config[k] for k in ("num_layers", "d_model", "use_residual", "pos_encoding_type", "qkv_mode")})
    if not config.get("use_residual", True):
        print(
            "【2.3 监控】无残差：子层为 LayerNorm(Dropout(Sublayer(x)))，无恒等捷径；请重点对比前 1–3 epoch "
            "train loss/token 相对 Baseline（--use-residual）是否更慢、更抖或居高不下，用于论证残差对 N=6 深层优化的作用。"
        )
    _append_text_log(
        text_log,
        f"run_dir={run_dir} experiment_mode={config.get('experiment_mode')} patience={patience}",
    )

    csv_fields = [
        "epoch",
        "train_loss_token",
        "ppl",
        "valid_bleu_0_1",
        "valid_bleu_x100",
        "lr",
        "is_best",
        "early_stopped",
    ]

    best_bleu: float = -1.0
    best_epoch = 0
    stalled = 0
    stopped_early = False
    total_epochs = int(config["epochs"])

    epoch_bar = _try_tqdm(
        range(1, total_epochs + 1),
        desc="Epochs",
        leave=True,
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        loss_tok = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            pad_idx,
            device,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        ppl = math.exp(min(loss_tok, 20.0))
        lr_now = optimizer.rate()
        if hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(train_loss_tok=f"{loss_tok:.4f}", ppl=f"{ppl:.2f}")

        line = f"Epoch {epoch}: train loss/token ≈ {loss_tok:.4f}, approx PPL {ppl:.2f}"
        print(line)
        _append_text_log(text_log, line)

        bleu = evaluate_bleu(
            model,
            valid_pairs,
            tok_de,
            tok_en,
            vocab_src,
            vocab_tgt,
            pad_idx,
            sos_idx,
            eos_idx,
            int(config["max_src_len"]),
            int(config["max_tgt_len"]),
            device,
            max_samples=300,
        )
        bleu_line = f"          valid BLEU (approx, 300 sents): {bleu * 100:.2f}"
        print(bleu_line)
        _append_text_log(text_log, bleu_line)

        improved = bleu > best_bleu + min_delta
        is_best = 1 if improved else 0
        _append_csv_row(
            metrics_csv,
            csv_fields,
            {
                "epoch": epoch,
                "train_loss_token": f"{loss_tok:.6f}",
                "ppl": f"{ppl:.4f}",
                "valid_bleu_0_1": f"{bleu:.6f}",
                "valid_bleu_x100": f"{bleu * 100:.4f}",
                "lr": f"{lr_now:.6e}",
                "is_best": is_best,
                "early_stopped": 0,
            },
        )

        if improved:
            best_bleu = bleu
            best_epoch = epoch
            stalled = 0
            _save_best_checkpoint(best_ckpt, model, epoch, bleu, config)
            meta = {"epoch": epoch, "valid_bleu_0_1": bleu, "valid_bleu_x100": bleu * 100}
            (run_dir / "best_model_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _append_text_log(text_log, f"  -> new best valid BLEU={bleu * 100:.2f}, saved {best_ckpt.name}")
        else:
            stalled += 1
            if patience > 0 and stalled >= patience:
                stopped_early = True
                msg = (
                    f"早停于 epoch {epoch}：已连续 {patience} 个 epoch 验证 BLEU 未超过历史最佳 "
                    f"{best_bleu * 100:.2f}（best @ epoch {best_epoch}）"
                )
                print(msg)
                _append_text_log(text_log, msg)
                break

    if best_ckpt.is_file():
        try:
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        vb = float(ckpt["valid_bleu"])
        print(f"测试阶段加载最佳权重: epoch={ckpt['epoch']}, valid_BLEU≈{vb * 100:.2f}")
        _append_text_log(
            text_log,
            f"Loaded best_model.pt for test: epoch={ckpt['epoch']} valid_bleu={vb * 100:.2f}",
        )

    print("Test BLEU (approx, best checkpoint):", end=" ")
    bleu_test = evaluate_bleu(
        model,
        test_pairs,
        tok_de,
        tok_en,
        vocab_src,
        vocab_tgt,
        pad_idx,
        sos_idx,
        eos_idx,
        int(config["max_src_len"]),
        int(config["max_tgt_len"]),
        device,
        max_samples=500,
    )
    print(f"{bleu_test * 100:.2f}")
    _append_text_log(text_log, f"Test BLEU (approx, 500 sents): {bleu_test * 100:.2f}")
    summary = {
        "best_valid_bleu_x100": best_bleu * 100,
        "best_epoch": best_epoch,
        "test_bleu_x100": bleu_test * 100,
        "early_stopped": stopped_early,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {metrics_csv.name}, {text_log.name}, {best_ckpt.name}, summary.json")


if __name__ == "__main__":
    main()
