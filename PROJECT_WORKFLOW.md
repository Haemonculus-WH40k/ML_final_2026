# 机器学习课程项目工作流

本文档用于记录项目当前进展，并指导从数据准备、模型实验到最终报告提交的完整流程。

更新时间：2026-06-26

## 1. 课程任务目标

项目任务是基于家庭用电多变量时间序列，使用过去 90 天数据预测未来用电曲线。

必须完成两个独立预测任务：

- 短期预测：过去 90 天预测未来 90 天；
- 长期预测：过去 90 天预测未来 365 天。

两个任务需要分别训练模型。长期预测模型参数不能复用于短期预测模型。

必须完成三类方法：

1. LSTM 预测模型；
2. Transformer 预测模型；
3. 自提出改进模型。本项目当前采用 CNN + Transformer。

最终评估至少包含：

- MSE；
- MAE；
- 每个模型、每个预测长度至少 5 轮实验；
- 报告 5 轮实验均值和标准差；
- 给出预测曲线与 Ground Truth 对比图。

## 2. 当前进展总览

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| GitHub 仓库 | 已完成 | 已上传至 `https://github.com/Haemonculus-WH40k/ML_final_2026.git` |
| 原始用电数据 | 已完成 | `household_power_consumption.txt` 已纳入 Git LFS |
| 天气数据 | 已完成 | 已使用 Hauts-de-Seine 省 92 号月度气象数据 |
| 数据预处理代码 | 已完成 | 已实现分钟级清洗、日聚合、天气合并 |
| 样本构造代码 | 已完成 | 已生成 90->90 与 90->365 两类 `.npz` 文件 |
| LSTM 模型 | 已完成 | 已完成短期、长期各 5 轮实验 |
| Transformer 模型 | 已完成 | 已完成短期、长期各 5 轮实验 |
| CNN + Transformer 改进模型 | 已完成 | 已完成短期、长期各 5 轮实验 |
| 训练入口 | 已完成 | 支持训练、验证、早停、保存 checkpoint、标准化和反标准化预测 |
| 评估入口 | 已完成 | 自动输出标准化和原始单位 MSE、MAE |
| 绘图入口 | 已完成 | 默认生成反标准化 Prediction vs Ground Truth 曲线 |
| 正式 30 次实验 | 已完成 | 已按 3 模型 x 2 任务 x 5 种子执行 |
| 结果汇总表 | 已完成 | 已生成 `experiments/tables/results.csv` 和 `summary.csv` |
| 最终报告 | 草稿已完成 | `report/final_report.md` 已写入事实内容；作者贡献和正式 PDF 待补 |

## 3. 当前项目结构

```text
ML_final/
  configs/
    data_pipeline.json
  data/
    raw/
      weather/
        MENSQ_92_previous-1950-2024.csv.gz
        MENSQ_descriptif_champs.csv
    processed/
      daily_power.csv
      daily_power.metadata.json
      daily_power_weather.csv
      weather_monthly_normalized.csv
      weather_monthly_normalized.metadata.json
      short_horizon_samples.npz
      long_horizon_samples.npz
      feature_scaler.json
  src/
    data/
      preprocess_power.py
      preprocess_weather.py
      make_windows.py
      run_data_pipeline.py
    models/
      lstm.py
      transformer.py
      improved_model.py
    utils/
      metrics.py
      seed.py
    train.py
    evaluate.py
    plot_predictions.py
  README.md
  DATA_SPLIT_PLAN.md
  PROJECT_WORKFLOW.md
  requirements.txt
  household_power_consumption.txt
```

建议后续新增：

```text
experiments/
  checkpoints/
  figures/
  logs/
  tables/
report/
  figures/
  final_report.pdf
```

## 4. 数据资源与当前处理结果

### 4.1 原始用电数据

文件：

- `household_power_consumption.txt`

数据概况：

- 原始行数：2,075,259；
- 原始时间范围：2006-12-16 17:24:00 至 2010-11-26 21:02:00；
- 分隔符：分号 `;`；
- 缺失标记：`?`；
- 原始特征：
  - `Global_active_power`
  - `Global_reactive_power`
  - `Voltage`
  - `Global_intensity`
  - `Sub_metering_1`
  - `Sub_metering_2`
  - `Sub_metering_3`

当前处理方式：

1. 合并 `Date` 和 `Time` 为时间戳；
2. 将 `?` 转为缺失值；
3. 数值列转为浮点数；
4. 按分钟重建完整时间索引；
5. 对缺失分钟使用时间插值，并做双向填充；
6. 计算剩余分表能耗：

```text
sub_metering_remainder =
  global_active_power * 1000 / 60
  - sub_metering_1
  - sub_metering_2
  - sub_metering_3
```

每日聚合结果：

- 输出文件：`data/processed/daily_power.csv`
- 元数据：`data/processed/daily_power.metadata.json`
- 日数据行数：1,441；
- 起始日期：2006-12-17；
- 结束日期：2010-11-26；
- 每日最低有效分钟数阈值：1,000；
- 当前不保留低于阈值的不完整日。

### 4.2 天气数据

课程要求融合法国月度基础气象数据。本项目选择家庭用电数据所在地附近的 Hauts-de-Seine 省，即法国省编号 `92`。

文件：

- 原始天气数据：`data/raw/weather/MENSQ_92_previous-1950-2024.csv.gz`
- 字段说明：`data/raw/weather/MENSQ_descriptif_champs.csv`
- 归一化天气文件：`data/processed/weather_monthly_normalized.csv`
- 元数据：`data/processed/weather_monthly_normalized.metadata.json`

天气字段：

| 字段 | 含义 | 当前处理 |
| --- | --- | --- |
| `RR` | 月累计降水高度 | 按课程要求除以 10 |
| `NBJRR1` | 当月日降水 >= 1 mm 的天数 | 保留月度值 |
| `NBJRR5` | 当月日降水 >= 5 mm 的天数 | 保留月度值 |
| `NBJRR10` | 当月日降水 >= 10 mm 的天数 | 保留月度值 |
| `NBJBROU` | 当月雾出现天数 | 插值并填补缺失 |

当前天气处理策略：

1. 读取 92 省全部可用气象站；
2. 每个月、每个字段在可用站点间取均值；
3. `RR` 除以 10；
4. 对聚合后缺失值按时间插值，并做前后向填充；
5. 按月份合并到每日用电数据。

当前统计：

- 使用气象站数量：20；
- `NBJBROU` 原始月度聚合后存在 625 个缺失，填补后为 0；
- 其他课程要求天气字段填补前后均无残留缺失。

### 4.3 合并后的建模数据

文件：

- `data/processed/daily_power_weather.csv`

当前建模特征共 13 个：

1. `global_active_power`
2. `global_reactive_power`
3. `sub_metering_1`
4. `sub_metering_2`
5. `sub_metering_3`
6. `sub_metering_remainder`
7. `voltage`
8. `global_intensity`
9. `RR`
10. `NBJRR1`
11. `NBJRR5`
12. `NBJRR10`
13. `NBJBROU`

预测目标：

- `global_active_power`

## 5. 数据划分与样本文件

课程 PDF 中提到的 `train.csv` 和 `tes.csv/test.csv` 当前未在项目目录提供，因此项目采用时间顺序划分。

划分原则：

- 训练集：前 70% 窗口样本；
- 验证集：中间 15% 窗口样本；
- 测试集：最后 15% 窗口样本；
- 不随机打乱时间顺序；
- 只在训练段拟合标准化参数；
- 验证集和测试集复用训练段标准化参数。

标准化器：

- `data/processed/feature_scaler.json`

### 5.1 短期预测样本

文件：

- `data/processed/short_horizon_samples.npz`

任务：

- 输入：过去 90 天；
- 输出：未来 90 天。

当前形状：

| 集合 | X 形状 | y 形状 |
| --- | --- | --- |
| train | `(883, 90, 13)` | `(883, 90)` |
| val | `(189, 90, 13)` | `(189, 90)` |
| test | `(190, 90, 13)` | `(190, 90)` |

### 5.2 长期预测样本

文件：

- `data/processed/long_horizon_samples.npz`

任务：

- 输入：过去 90 天；
- 输出：未来 365 天。

当前形状：

| 集合 | X 形状 | y 形状 |
| --- | --- | --- |
| train | `(690, 90, 13)` | `(690, 365)` |
| val | `(148, 90, 13)` | `(148, 365)` |
| test | `(149, 90, 13)` | `(149, 365)` |

更详细日期范围见 `DATA_SPLIT_PLAN.md`。

## 6. 可复现运行流程

### 6.1 安装依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

- `numpy`
- `pandas`
- `matplotlib`
- `torch`

### 6.2 重新生成数据

```powershell
python src/data/run_data_pipeline.py --config configs/data_pipeline.json
```

该命令会重新生成：

- `data/processed/daily_power.csv`
- `data/processed/daily_power.metadata.json`
- `data/processed/weather_monthly_normalized.csv`
- `data/processed/weather_monthly_normalized.metadata.json`
- `data/processed/daily_power_weather.csv`
- `data/processed/short_horizon_samples.npz`
- `data/processed/long_horizon_samples.npz`
- `data/processed/feature_scaler.json`

### 6.3 单次训练命令

短期 LSTM 示例：

```powershell
python src/train.py `
  --samples data/processed/short_horizon_samples.npz `
  --model lstm `
  --output-dir experiments/checkpoints/short_lstm_seed42 `
  --epochs 100 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --patience 10 `
  --seed 42
```

长期 Transformer 示例：

```powershell
python src/train.py `
  --samples data/processed/long_horizon_samples.npz `
  --model transformer `
  --output-dir experiments/checkpoints/long_transformer_seed42 `
  --epochs 100 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --patience 10 `
  --seed 42
```

改进模型示例：

```powershell
python src/train.py `
  --samples data/processed/short_horizon_samples.npz `
  --model conv_transformer `
  --output-dir experiments/checkpoints/short_conv_transformer_seed42 `
  --epochs 100 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --patience 10 `
  --seed 42
```

训练输出：

- `best_model.pt`
- `history.json`
- `metrics.json`
- `test_predictions.npz`
- `test_predictions_scaled.npz`

### 6.4 评估保存的预测

```powershell
python src/evaluate.py `
  --predictions experiments/checkpoints/short_lstm_seed42/test_predictions.npz `
  --output experiments/checkpoints/short_lstm_seed42/metrics_recomputed.json
```

`metrics.json` 同时包含标准化空间和反标准化原始单位的 MSE、MAE。最终报告应优先使用 `mse_original` 和 `mae_original`。

### 6.5 绘制预测曲线

```powershell
python src/plot_predictions.py `
  --predictions experiments/checkpoints/short_lstm_seed42/test_predictions.npz `
  --output experiments/figures/short_lstm_seed42_sample0.png `
  --sample-index 0 `
  --title "Short Horizon LSTM Seed 42"
```

绘图脚本默认使用反标准化后的 `global_active_power`。如需调试标准化空间曲线，可额外传入 `--scaled`。

## 7. 正式实验矩阵

课程要求每个模型、每个任务至少 5 轮实验。建议固定 5 个随机种子：

```text
42, 2026, 7, 13, 99
```

必须完成的 30 次训练：

| 任务 | 样本文件 | 模型 | 轮数 |
| --- | --- | --- | ---: |
| 短期 90 天 | `short_horizon_samples.npz` | `lstm` | 5 |
| 短期 90 天 | `short_horizon_samples.npz` | `transformer` | 5 |
| 短期 90 天 | `short_horizon_samples.npz` | `conv_transformer` | 5 |
| 长期 365 天 | `long_horizon_samples.npz` | `lstm` | 5 |
| 长期 365 天 | `long_horizon_samples.npz` | `transformer` | 5 |
| 长期 365 天 | `long_horizon_samples.npz` | `conv_transformer` | 5 |

正式实验训练配置：

- 最大 epoch：100；
- early stopping patience：10；
- batch size：64；
- learning rate：0.001；
- weight decay：0.0001。

推荐输出目录命名：

```text
experiments/checkpoints/{task}_{model}_seed{seed}/
```

其中：

- `{task}` 使用 `short` 或 `long`；
- `{model}` 使用 `lstm`、`transformer`、`conv_transformer`；
- `{seed}` 使用实际随机种子。

## 8. 结果汇总规范

建议建立主结果表：

- `experiments/tables/results.csv`

每一行对应一次训练，字段至少包含：

| 字段 | 说明 |
| --- | --- |
| `task` | `short` 或 `long` |
| `horizon` | `90` 或 `365` |
| `model` | `lstm`、`transformer`、`conv_transformer` |
| `seed` | 随机种子 |
| `epochs_ran` | 实际训练 epoch 数 |
| `best_val_loss` | 最佳验证集损失 |
| `test_mse_scaled` | 标准化空间测试 MSE |
| `test_mae_scaled` | 标准化空间测试 MAE |
| `test_mse_original` | 反标准化后测试 MSE |
| `test_mae_original` | 反标准化后测试 MAE |
| `checkpoint_dir` | 本轮输出目录 |
| `figure_path` | 预测曲线图路径 |

最终报告表格应按 `task + model` 聚合，报告：

- MSE mean；
- MSE std；
- MAE mean；
- MAE std；
- 最佳或代表性曲线图。

## 9. 模型说明要点

### 9.1 LSTM

当前实现：`src/models/lstm.py`

结构：

1. 输入 `[batch, 90, feature_dim]`；
2. 多层 LSTM 编码历史序列；
3. 取最后时间步隐状态；
4. LayerNorm + Dropout + Linear 输出预测长度。

报告中可作为经典循环神经网络基线。

### 9.2 Transformer

当前实现：`src/models/transformer.py`

结构：

1. Linear 投影到 `d_model`；
2. 加入正弦位置编码；
3. Transformer Encoder 建模时间依赖；
4. 对时间维做均值池化；
5. Linear 输出预测长度。

报告中可强调自注意力对长依赖建模的优势。

### 9.3 CNN + Transformer 改进模型

当前实现：`src/models/improved_model.py`

结构：

1. Conv1d 提取局部用电模式；
2. GELU 和 BatchNorm 稳定局部特征；
3. 加入位置编码；
4. Transformer Encoder 建模 90 天窗口内全局依赖；
5. 预测头输出未来曲线。

报告中可解释创新点：

- CNN 捕捉局部周期、尖峰和短期波动；
- Transformer 建模较长时间依赖；
- 二者结合比单纯 LSTM 或单纯 Transformer 更适合混合局部波动与季节变化的用电序列。

## 10. 实验结果摘要

正式实验已经完成，详细结果见：

- `experiments/EXPERIMENT_SUMMARY.md`
- `experiments/tables/results.csv`
- `experiments/tables/summary.csv`

反标准化测试集汇总结果：

| 任务 | 模型 | Runs | MSE Mean | MSE Std | MAE Mean | MAE Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 90 天预测 | LSTM | 5 | 287850.41 | 5995.96 | 429.60 | 5.29 |
| 90 天预测 | Transformer | 5 | 292620.39 | 25074.70 | 439.39 | 22.79 |
| 90 天预测 | CNN + Transformer | 5 | 212555.30 | 30997.75 | 362.90 | 30.66 |
| 365 天预测 | LSTM | 5 | 190709.40 | 16130.92 | 337.09 | 17.03 |
| 365 天预测 | Transformer | 5 | 178740.92 | 17516.37 | 325.48 | 18.32 |
| 365 天预测 | CNN + Transformer | 5 | 165483.01 | 12328.18 | 310.95 | 13.41 |

当前结果显示，CNN + Transformer 在短期和长期两个任务上均取得最低平均 MSE 和 MAE。最终报告需要围绕这一结果分析局部卷积特征提取和 Transformer 长依赖建模的互补性。

## 11. 报告写作框架

最终报告建议包括以下部分：

1. 问题介绍
   - 家庭用电预测背景；
   - 90->90 和 90->365 两个任务定义；
   - 数据来源和天气特征来源。

2. 数据处理
   - 原始分钟级数据清洗；
   - 缺失值处理；
   - 剩余分表能耗构造；
   - 日聚合规则；
   - 天气数据选择、字段解释和合并；
   - 时间顺序划分和标准化策略。

3. 模型方法
   - LSTM；
   - Transformer；
   - CNN + Transformer 改进模型；
   - 三种模型输入输出形式和核心超参数。

4. 实验设置
   - 训练、验证、测试划分；
   - 5 个随机种子；
   - batch size、learning rate、early stopping；
   - MSE 和 MAE 指标；
   - 反标准化评估说明。

5. 结果与分析
   - 短期任务结果表；
   - 长期任务结果表；
   - 三类模型均值和标准差对比；
   - Ground Truth vs Prediction 曲线；
   - 短期与长期预测难度差异。

6. 讨论
   - 天气特征的作用和局限；
   - 长期预测误差累积；
   - 数据缺失和样本量限制；
   - 改进模型为何有效或为何未达到预期；
   - 后续可改进方向。

7. 参考文献与工具声明
   - 数据集来源；
   - PyTorch、pandas、numpy 等工具；
   - 如使用 ChatGPT、DeepSeek 等辅助工具，按课程要求注明。

8. 组员贡献
   - 若为单人完成，明确说明；
   - 若为多人完成，列出每人负责模块。

## 12. 最终提交清单

提交前逐项检查：

- [x] 已整理 GitHub 仓库链接；
- [x] 已纳入原始用电数据；
- [x] 已纳入天气数据；
- [x] 已完成用电数据清洗和日聚合；
- [x] 已完成天气字段处理和合并；
- [x] 已构造短期 90->90 样本；
- [x] 已构造长期 90->365 样本；
- [x] 已实现 LSTM 模型；
- [x] 已实现 Transformer 模型；
- [x] 已实现 CNN + Transformer 改进模型；
- [x] 已实现基础训练入口；
- [x] 已实现基础评估入口；
- [x] 已实现基础绘图入口；
- [x] 已完成 LSTM 短期 5 轮实验；
- [x] 已完成 LSTM 长期 5 轮实验；
- [x] 已完成 Transformer 短期 5 轮实验；
- [x] 已完成 Transformer 长期 5 轮实验；
- [x] 已完成改进模型短期 5 轮实验；
- [x] 已完成改进模型长期 5 轮实验；
- [x] 已汇总 MSE、MAE 均值和标准差；
- [x] 已生成反标准化预测曲线图；
- [x] 已完成三种方法比较；
- [x] 已完成最终报告 Markdown 草稿；
- [ ] 已补充作者贡献并导出正式 PDF；
- [ ] 已写明参考文献；
- [ ] 已写明作者贡献；
- [ ] 已在 2026-07-15 中午 12 点前提交。

## 13. 建议下一步执行顺序

1. 根据 `experiments/EXPERIMENT_SUMMARY.md` 撰写最终报告结果与分析部分。
2. 从 `experiments/figures/` 中选择代表性预测曲线放入报告。
3. 在报告中说明官方 `train.csv` 和 `tes.csv/test.csv` 未提供，因此采用时间顺序划分。
4. 写明参考文献、工具声明和作者贡献。
5. 完成最终报告 PDF。
6. 按课程要求提交。
