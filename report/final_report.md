# 家庭用电多变量时间序列预测实验报告

## 1. 问题介绍

本项目面向家庭用电多变量时间序列预测任务。给定过去 90 天的每日用电和天气特征，分别预测未来 90 天和未来 365 天的 `global_active_power` 曲线。

项目完成两个独立任务：

- 短期预测：90 天输入预测未来 90 天；
- 长期预测：90 天输入预测未来 365 天。

两个任务分别训练模型，长期预测模型不复用短期预测模型参数。

项目比较三类方法：

1. LSTM；
2. Transformer；
3. CNN + Transformer 改进模型。

评估指标为 MSE 和 MAE。每个模型、每个任务使用 5 个随机种子独立训练，并报告均值和标准差。

## 2. 数据处理

### 2.1 用电数据

原始数据文件为 `household_power_consumption.txt`，包含 2,075,259 行分钟级家庭用电记录。原始时间范围为 2006-12-16 17:24:00 至 2010-11-26 21:02:00。

处理流程：

1. 合并 `Date` 和 `Time` 为时间戳；
2. 将缺失标记 `?` 转为缺失值；
3. 将数值列转换为浮点数；
4. 按分钟重建完整时间索引；
5. 对缺失分钟使用时间插值和双向填充；
6. 构造剩余分表能耗：

```text
sub_metering_remainder =
  global_active_power * 1000 / 60
  - sub_metering_1
  - sub_metering_2
  - sub_metering_3
```

随后将分钟级数据聚合为每日数据。求和字段包括 `global_active_power`、`global_reactive_power`、三类分表能耗和 `sub_metering_remainder`；求均值字段包括 `voltage` 和 `global_intensity`。

处理后每日数据：

- 文件：`data/processed/daily_power.csv`
- 日期范围：2006-12-17 至 2010-11-26
- 行数：1,441 天

### 2.2 天气数据

项目使用法国 Hauts-de-Seine 省，省编号 `92`，月度基础气象数据：

- `data/raw/weather/MENSQ_92_previous-1950-2024.csv.gz`
- `data/raw/weather/MENSQ_descriptif_champs.csv`

使用字段：

| 字段 | 含义 | 处理方式 |
| --- | --- | --- |
| `RR` | 月累计降水高度 | 按课程要求除以 10 |
| `NBJRR1` | 当月日降水 >= 1 mm 的天数 | 保留月度值 |
| `NBJRR5` | 当月日降水 >= 5 mm 的天数 | 保留月度值 |
| `NBJRR10` | 当月日降水 >= 10 mm 的天数 | 保留月度值 |
| `NBJBROU` | 当月雾出现天数 | 插值并前后向填充 |

多个气象站同月记录按字段取平均值，再按月份合并到每日用电数据。最终建模数据文件为 `data/processed/daily_power_weather.csv`。

### 2.3 特征和划分

建模特征共 13 个：

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

课程中提到的 `train.csv` 和 `tes.csv/test.csv` 当前未在项目目录中提供，因此本项目按时间顺序划分窗口样本：

- 训练集：前 70%；
- 验证集：中间 15%；
- 测试集：最后 15%。

标准化参数仅在训练段拟合，然后用于验证集和测试集，避免未来信息泄漏。

短期样本：

- 文件：`data/processed/short_horizon_samples.npz`
- train：`(883, 90, 13)`，`(883, 90)`
- val：`(189, 90, 13)`，`(189, 90)`
- test：`(190, 90, 13)`，`(190, 90)`

长期样本：

- 文件：`data/processed/long_horizon_samples.npz`
- train：`(690, 90, 13)`，`(690, 365)`
- val：`(148, 90, 13)`，`(148, 365)`
- test：`(149, 90, 13)`，`(149, 365)`

## 3. 模型方法

### 3.1 LSTM

LSTM 模型将 90 天多变量输入编码为隐藏状态，并取最后时间步表示，经 LayerNorm、Dropout 和全连接层直接输出未来预测序列。该模型作为循环神经网络基线。

### 3.2 Transformer

Transformer 模型先将输入特征投影到 `d_model` 维度，加入正弦位置编码，再通过 Transformer Encoder 建模时间依赖。编码结果在时间维上平均池化后输出未来曲线。

### 3.3 CNN + Transformer

改进模型先使用一维卷积提取局部用电模式，再使用 Transformer Encoder 建模 90 天窗口中的长依赖关系。CNN 能捕捉短期波动、尖峰和局部周期，Transformer 用于补充全局时序依赖，因此该结构适合同时存在局部波动和较长周期变化的家庭用电序列。

## 4. 实验设置

统一训练配置：

- 随机种子：`42, 2026, 7, 13, 99`
- 最大 epoch：100
- early stopping patience：10
- batch size：64
- optimizer：AdamW
- learning rate：0.001
- weight decay：0.0001
- loss：MSELoss

每组任务和模型运行 5 次，共 30 次正式训练。训练保存标准化预测，并反标准化回 `global_active_power` 原始单位后计算最终 MSE 和 MAE。

## 5. 实验结果

以下表格为测试集反标准化指标。

| 任务 | 模型 | Runs | MSE Mean | MSE Std | MAE Mean | MAE Std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 90 天预测 | LSTM | 5 | 287850.41 | 5995.96 | 429.60 | 5.29 |
| 90 天预测 | Transformer | 5 | 292620.39 | 25074.70 | 439.39 | 22.79 |
| 90 天预测 | CNN + Transformer | 5 | 212555.30 | 30997.75 | 362.90 | 30.66 |
| 365 天预测 | LSTM | 5 | 190709.40 | 16130.92 | 337.09 | 17.03 |
| 365 天预测 | Transformer | 5 | 178740.92 | 17516.37 | 325.48 | 18.32 |
| 365 天预测 | CNN + Transformer | 5 | 165483.01 | 12328.18 | 310.95 | 13.41 |

完整逐轮结果见 `experiments/tables/results.csv`，分组汇总见 `experiments/tables/summary.csv`。预测曲线见 `experiments/figures/`。

## 6. 结果分析

短期预测中，CNN + Transformer 的平均 MSE 和 MAE 均低于 LSTM 和 Transformer，说明局部卷积特征对 90 天预测具有明显帮助。LSTM 的短期结果比 Transformer 略稳定，但平均误差仍高于改进模型。

长期预测中，Transformer 的平均表现优于 LSTM，说明自注意力机制对较长预测跨度更有优势。CNN + Transformer 进一步降低了长期预测误差，并且 MAE 标准差低于 LSTM 和 Transformer，说明该改进模型在本实验设置下更稳定。

从曲线角度看，深度模型通常能捕捉整体均值水平和较平滑趋势，但对局部尖峰、突然下降和高频波动的拟合仍有限。长期预测任务虽然预测跨度更长，但测试窗口覆盖的目标区间更平滑，因此其平均 MAE 低于短期任务；这一现象与测试集本身分布有关，不能简单理解为长期预测天然更容易。

## 7. 讨论

本项目的主要限制包括：

1. 用电数据只覆盖单一家庭，样本量有限；
2. 天气数据为月度省级聚合，无法反映每日微观天气变化；
3. 预测头采用一次性多步输出，没有显式建模未来输出之间的自回归依赖；
4. 当前超参数未进行大规模搜索，结果仍有提升空间。

后续可改进方向：

- 使用日级或小时级天气数据；
- 加入星期、月份、节假日等时间特征；
- 引入多尺度卷积或趋势/季节分解模块；
- 使用更系统的超参数搜索；
- 对极端波动日单独分析误差来源。

## 8. 参考与工具说明

数据来源：

- UCI Individual household electric power consumption 数据集；
- Météo-France 月度基础气象数据。

主要工具：

- Python
- NumPy
- pandas
- PyTorch
- Matplotlib

AI 辅助说明：

- 本项目开发与文档整理过程中使用了 ChatGPT/Codex 辅助代码实现、实验流程梳理和文字草拟。最终内容、实验结果和提交材料需由提交者复核确认。

## 9. 作者贡献

待填写。请在正式提交前补充组员姓名、学号、分工和贡献说明；若为单人完成，请明确写明单人完成的模块范围。
