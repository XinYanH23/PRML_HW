# Multi30k 德→英 Transformer Baseline（已跑完，无需重跑）

**说明**：本次运行的 **Loss / BLEU / 耗时** 均来自你粘贴的终端输出；脚本默认**不保存 checkpoint**，因此若需要「最佳 epoch 的模型权重」才需改代码后重训或单训若干 epoch；**整理实验数字不必重跑 3 小时**。

---

## 运行命令

```bash
cd "/home/hxy/桌面/2026PRML/hw_4"
python train.py --local-repo "/home/hxy/桌面/2026PRML/hw_4/7c37f-main/dataset-master"
```

- 数据：本地 `dataset-master`（train 29000 / val 1014 / test 1000）
- 环境：CUDA 驱动告警后实际为 **CPU** 训练（`resolve_device` 回退）
- `Config`: `num_layers=6`, `d_model=512`, `use_residual=True`, `pos_encoding_type=sin`, `qkv_mode=standard`
- `batch_size=128` → 每 epoch **227** 个 batch

---

## 1. 训练 Loss / PPL（按 epoch）

| Epoch | train loss/token ↓ | approx PPL | valid BLEU (300 sents) |
|------:|-------------------:|-------------:|----------------------:|
| 1 | 5.4213 | 226.18 | 0.17 |
| 2 | 3.6305 | 37.73 | 6.64 |
| 3 | 2.9902 | 19.89 | 11.11 |
| 4 | 2.6380 | 13.99 | 11.47 |
| 5 | 2.3908 | 10.92 | 14.70 |
| 6 | 2.2061 | 9.08 | **18.68** |
| 7 | 2.0624 | 7.86 | 16.44 |
| 8 | 1.9561 | 7.07 | **18.62** |
| 9 | 1.8742 | 6.52 | 18.15 |
| 10 | 1.8209 | 6.18 | 17.07 |
| 11 | 1.7837 | 5.95 | 16.74 |
| 12 | 1.7666 | **5.85** | 15.44 |
| 13 | 1.7749 | 5.90 | 12.41 |
| 14 | 1.8510 | 6.37 | 17.79 |
| 15 | 2.0040 | 7.42 | 10.28 |

- **最后一 epoch 训练 loss**：**2.0040**（PPL **7.42**）— 注意：训练 loss 在 epoch 12 附近最低，**末 epoch 回升**，常与 **过拟合 / 后期学习率偏大** 有关。
- **验证 BLEU 峰值**：约 **18.7**（epoch **6** 与 **8** 附近）；末 epoch 验证 BLEU 跌至 **10.28**。
- **Test BLEU（脚本末尾）**：**9.83**（约 500 句贪心）。

---

## 2. 收敛速度 / 总耗时

| 项 | 数值 |
|----|------|
| Epoch 1 单轮训练墙钟 | 约 **15 min 26 s**（227 batch，~4.08 s/it） |
| 典型 epoch（中后期） | 约 **12 min 40 s～13 min**（~3.4–3.5 s/it） |
| 15 epoch 总训练墙钟 | 约 **3 h 31 min**（日志 `3:30:56` 量级 + 末尾评估） |
| 外层 tqdm 显示 per-epoch | 约 **805–843 s/epoch**（与上表一致） |

---

## 3. 写报告时可用的简短结论

- **训练曲线**：前 12 个 epoch 训练 loss 持续下降至约 **1.77**；第 13–15 epoch 上升，末值 **2.00**，提示 **过拟合或优化末期不稳定**。
- **验证 BLEU**：最高约 **18.7**；**最后一轮验证 BLEU 不是最高**，若写「官方 Baseline 数字」宜报告 **best valid BLEU ~18.7 @ epoch 6–8**，并说明 **last epoch ≠ best**。
- **Test BLEU ~9.83**：低于峰值 valid，与贪心解码、仅 500 句、及末模型非最优等因素一致；作课程内 **相对消融基线** 足够。

---

## 4. 下次自动落盘（避免只靠剪贴板）

```bash
mkdir -p "/home/hxy/桌面/2026PRML/hw_4/runs"
cd "/home/hxy/桌面/2026PRML/hw_4"
python train.py --local-repo "/home/hxy/桌面/2026PRML/hw_4/7c37f-main/dataset-master" 2>&1 | tee "runs/baseline_$(date +%Y%m%d_%H%M%S).log"
```

---

## 5. 消融实验对照（复制下行填新跑结果）

| 实验 | 改动 | best valid BLEU | last train loss |
|------|------|-----------------|-----------------|
| Baseline | 上表 | ~18.7 (ep 6–8) | 2.0040 (ep15) |
| | | | |

---

## 6. 写报告可用：Epoch 12 之后训练 Loss 回升（训练监控）

**现象**：上表显示训练 **loss/token 在约 epoch 12 降至最低（≈1.77）后，第 13–15 epoch 明显回升**，末 epoch 达 **≈2.00**；同期 **验证 BLEU** 从峰值 **≈18.7** 滑落，末 epoch 仅 **≈10.3**。

**专业表述（可整段放入报告）**：

1. **过拟合**：中后期模型在训练集上继续「死记」噪声与特定样本模式，训练损失与验证指标背离，表现为 **valid BLEU 下降而 train loss 反而上升或震荡**（本日志中二者在 epoch 12 后同向变差，符合「泛化变差 + 优化轨迹漂移」的综合图景）。
2. **Noam 学习率阶段**：Warmup（本配置 `warmup_steps=4000`）结束后，学习率按 \(1/\sqrt{\text{step}}\) 进入衰减段；若有效步长相对当前损失曲面仍偏大，后期可能出现 **在最优点附近震荡、训练损失反弹**。可与 **早停于 valid BLEU 最佳 epoch**、**保存 best checkpoint 而非 last** 的工程做法一并讨论，体现对 **训练曲线、验证集与 checkpoint 策略** 的监控意识。

**结论写法**：报告 **best valid BLEU** 时以 **epoch 6–8** 为准，并说明 **末 epoch 不代表最优**；若课程要求分析「为何不训满 15 epoch 仍可能更好」，可引用上述两点。
