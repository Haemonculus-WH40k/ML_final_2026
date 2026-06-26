# 机器学习课程项目

本项目用于完成家庭用电多变量时间序列预测课程考核。

当前已完成：

- 原始用电数据检查；
- 分钟级用电数据到天级数据的处理代码；
- 法国 92 省月度天气数据处理与合并；
- 90->90 与 90->365 预测样本构造代码；
- LSTM、Transformer、CNN+Transformer 改进模型；
- 训练、评估、绘图和批量实验脚本；
- 3 个模型 x 2 个任务 x 5 个随机种子的 30 次正式实验；
- MSE、MAE 均值和标准差汇总；
- 反标准化预测曲线图。

GitHub 仓库：

```text
https://github.com/Haemonculus-WH40k/ML_final_2026.git
```

## 数据处理

运行完整数据处理流程：

```powershell
python src/data/run_data_pipeline.py --config configs/data_pipeline.json
```

该流程会生成：

- `data/processed/daily_power.csv`
- `data/processed/daily_power_weather.csv`
- `data/processed/short_horizon_samples.npz`
- `data/processed/long_horizon_samples.npz`
- `data/processed/feature_scaler.json`

当前项目已放入法国 92 省月度天气数据，`configs/data_pipeline.json` 中的 `weather_path` 已指向：

```text
data/raw/weather/MENSQ_92_previous-1950-2024.csv.gz
```

因此运行数据管线时会自动完成天气字段处理、月度天气合并和短期/长期样本生成。

## 模型实验

单次训练示例：

```powershell
python src/train.py `
  --samples data/processed/short_horizon_samples.npz `
  --model conv_transformer `
  --output-dir experiments/checkpoints/short_conv_transformer_seed42 `
  --epochs 50 `
  --batch-size 64 `
  --patience 8 `
  --seed 42
```

复现实验矩阵：

```powershell
python src/run_experiments.py --epochs 50 --batch-size 64 --patience 8 --skip-existing
```

结果文件：

- 逐轮结果：`experiments/tables/results.csv`
- 分组汇总：`experiments/tables/summary.csv`
- 实验说明：`experiments/EXPERIMENT_SUMMARY.md`
- 预测曲线：`experiments/figures/`
- 报告草稿：`report/final_report.md`

## 当前结果

测试集反标准化指标中，CNN+Transformer 在两个任务上均取得当前最好平均表现：

- 90 天预测：MSE 212555.30，MAE 362.90；
- 365 天预测：MSE 165483.01，MAE 310.95。
