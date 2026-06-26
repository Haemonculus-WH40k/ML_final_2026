# 机器学习课程项目

本项目用于完成家庭用电多变量时间序列预测课程考核。

当前已完成：

- 原始用电数据检查；
- 数据处理工作流文档；
- 分钟级用电数据到天级数据的处理代码；
- 90->90 与 90->365 预测样本构造代码；
- LSTM、Transformer、CNN+Transformer 改进模型代码骨架；
- 训练、评估、绘图入口代码。

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

当前天气数据尚未放入项目，因此流程会先使用用电数据生成样本。后续取得天气数据后，在 `configs/data_pipeline.json` 中设置 `weather_path` 即可合并。

## 注意

本阶段只进行数据处理和代码准备，不启动模型训练。
