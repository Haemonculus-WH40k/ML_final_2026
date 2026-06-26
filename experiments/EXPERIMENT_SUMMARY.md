# 实验结果总结

本文件记录本项目已经完成的正式实验结果。实验结果来自 `experiments/tables/results.csv` 和 `experiments/tables/summary.csv`，不是手工估计。

## 1. 实验设置

统一配置：

- 输入窗口：90 天；
- 短期预测：未来 90 天；
- 长期预测：未来 365 天；
- 特征数：13；
- 训练/验证/测试划分：时间顺序 70% / 15% / 15%；
- 随机种子：`42, 2026, 7, 13, 99`；
- 每组任务和模型运行 5 次；
- 最大 epoch：100；
- early stopping patience：10；
- batch size：64；
- optimizer：AdamW；
- learning rate：0.001；
- weight decay：0.0001；
- 评估指标：MSE、MAE。

训练过程中保存的标准化预测会反标准化回 `global_active_power` 原始单位后计算最终报告指标。标准化空间指标仍保留在结果表中，便于调试和复核。

## 2. 完成的实验矩阵

| 任务 | 模型 | 独立运行次数 | 状态 |
| --- | --- | ---: | --- |
| 90 天预测 | LSTM | 5 | 已完成 |
| 90 天预测 | Transformer | 5 | 已完成 |
| 90 天预测 | CNN + Transformer | 5 | 已完成 |
| 365 天预测 | LSTM | 5 | 已完成 |
| 365 天预测 | Transformer | 5 | 已完成 |
| 365 天预测 | CNN + Transformer | 5 | 已完成 |

合计：30 次正式训练运行。

## 3. 汇总结果

以下结果为测试集上反标准化后的 `global_active_power` 指标。

| 任务 | 模型 | Runs | MSE Mean | MSE Std | MAE Mean | MAE Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 90 天预测 | LSTM | 5 | 287850.41 | 5995.96 | 429.60 | 5.29 |
| 90 天预测 | Transformer | 5 | 292620.39 | 25074.70 | 439.39 | 22.79 |
| 90 天预测 | CNN + Transformer | 5 | 212555.30 | 30997.75 | 362.90 | 30.66 |
| 365 天预测 | LSTM | 5 | 190709.40 | 16130.92 | 337.09 | 17.03 |
| 365 天预测 | Transformer | 5 | 178740.92 | 17516.37 | 325.48 | 18.32 |
| 365 天预测 | CNN + Transformer | 5 | 165483.01 | 12328.18 | 310.95 | 13.41 |

当前结果显示：

- 短期预测中，CNN + Transformer 的平均 MSE 和 MAE 均低于 LSTM 与 Transformer；
- 长期预测中，CNN + Transformer 同样取得最低平均 MSE 和 MAE；
- LSTM 在短期任务上比 Transformer 略稳定，但长期任务上 Transformer 的平均误差优于 LSTM；
- CNN + Transformer 的长期任务标准差也相对较小，说明改进模型在本次实验中更稳定。

## 4. 产物位置

逐轮结果：

- `experiments/tables/results.csv`

分组汇总：

- `experiments/tables/summary.csv`

预测曲线：

- `experiments/figures/`

模型 checkpoint 和原始预测数组：

- `experiments/checkpoints/`

其中 checkpoint 目录被 `.gitignore` 忽略，不上传到 GitHub；结果表和预测曲线会随项目提交。
