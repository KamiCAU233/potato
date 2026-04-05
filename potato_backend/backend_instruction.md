# new_model_helper.md

目的：向下一个 Agent（或后端开发者）清晰说明本次训练产物的用途与正确使用方法，确保“黑箱”模型被正确调用和集成。

一、产物清单（位于仓库根目录）
- `potato_blight_lstm_model.keras`：训练好的 Keras 模型（SavedModel 格式）。
- `scaler.pkl`：训练时使用的 `sklearn.preprocessing.MinMaxScaler`（通过 `joblib.dump` 保存）。


二、CSV 字段确认（请务必参考 `potato_blight_risk_regression_dataset.csv` 的表头）
CSV 表头（已确认）：

sample_id, avg_temp_7d, max_temp_7d, min_temp_7d, rh_over_90_hours, total_rainfall_7d, longest_wet_period_hours, risk_score, infection_level, cycle_completed, split

本模型实际使用的输入特征（严格的顺序和名称）：
1. `avg_temp_7d` — 过去 7 天的平均气温（训练数据单位/刻度需保持一致，通常为摄氏度）。
2. `max_temp_7d` — 过去 7 天的最高气温。
3. `min_temp_7d` — 过去 7 天的最低气温。
4. `rh_over_90_hours` — 相对湿度 ≥90% 的累计小时数（过去某段时间的累计小时）。
5. `total_rainfall_7d` — 过去 7 天的总降水量（通常单位为 mm）。
6. `longest_wet_period_hours` — 最长连续湿润/结露期的小时数。

目标标签（训练时使用）：
- `risk_score` — 连续值，范围 0~1，表征晚疫病发生风险（回归目标）。

注意：任何传入模型的输入都必须严格按照上面的顺序提供数值（数值类型为浮点数），且单位与训练数据一致。

三、`scaler.pkl` 的作用与使用方法
- 类型：`sklearn.preprocessing.MinMaxScaler`，在训练集上 `fit` 后保存，用于将每个特征缩放到 [0, 1] 区间（feature-wise）。
- 为什么需要：模型在训练时使用了缩放后的输入；推理阶段必须使用相同的缩放函数以避免分布不匹配导致预测错误。

如何检查 scaler 内容（示例 Python）：

```python
import joblib
scaler = joblib.load('scaler.pkl')
print('data_min_:', scaler.data_min_)
print('data_max_:', scaler.data_max_)
print('feature_range:', scaler.feature_range)
```

如何使用（单样本）：

```python
import numpy as np
import joblib

scaler = joblib.load('scaler.pkl')
# 假设 input_feats 与 CSV 中顺序一致
input_feats = [22.3, 25.0, 18.1, 12.0, 4.5, 48.0]
x = np.array(input_feats, dtype=float).reshape(1, -1)
x_scaled = scaler.transform(x)  # 结果范围应在 [0,1]
```

四、Keras 模型的输入/输出与调用细节
- 模型输入形状：`(batch_size, 1, 6)`，其中 `1` 表示时间步（我们将每条样本视为单时间步的序列），`6` 为特征数。
- 模型输出：单个浮点值，范围 `[0,1]`（模型最后一层为 `Dense(1, activation='sigmoid')`），表示 `risk_score` 的估计值。

示例：完整载入并预测单样本

```python
import joblib
import numpy as np
import tensorflow as tf

scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('potato_blight_lstm_model.keras')

input_feats = [22.3, 25.0, 18.1, 12.0, 4.5, 48.0]
X = np.array(input_feats, dtype=float).reshape(1, -1)
X_s = scaler.transform(X)
X_s = X_s.reshape(1, 1, X_s.shape[1])  # (batch, timesteps=1, features=6)
pred = model.predict(X_s)[0,0]
print('predicted risk_score =', float(pred))
```

批量预测示例：

```python
# X_batch shape: (N, 6)
X_batch_s = scaler.transform(X_batch)
X_batch_s = X_batch_s.reshape(-1, 1, X_batch_s.shape[1])
preds = model.predict(X_batch_s).flatten()  # shape (N,)
```

五、业务阈值与判断
- 默认高风险阈值：`0.6`（训练脚本与评估使用该阈值）。
- 输出解读：若 `pred >= 0.6` 则判定为“高风险”，建议进行预警/干预。

六、常见风险与排查建议
- 特征顺序错误：最常见错误。务必按上面列出的六个特征顺序传入。
- 缺失值/NaN：推理前请填补或拒绝含 NaN 的输入（模型未对 NaN 做特殊处理）。
- 缩放不一致：不要在推理时重复 fit scaler；一定要加载并使用 `scaler.pkl` 的 `transform`。
- TensorFlow 兼容性：模型以当前环境的 TensorFlow 保存（见 `requirements.txt` 的 `tensorflow>=2.11`）。若后端 TensorFlow 版本不同，建议部署为微服务以隔离环境，或导出为 ONNX（需额外验证）。

七、与原系统输入/输出不一致时的接入建议
- 如果原系统使用不同输入格式（例如逐小时时序或不同聚合指标），必须在接入层做特征映射/聚合：例如把小时温度序列聚合为 `avg_temp_7d/max/min` 等。
- 如果原系统期望分类输出（例如高/中/低），应该在接入层把模型的连续输出 `risk_score` 转换为类别（例如 `risk_score >= 0.6` → 高风险）。

八、快速自检清单（部署前）
1. 确认 `scaler.pkl` 与 `potato_blight_lstm_model.keras` 在同一可访问目录。
2. 对一小批已知样本（CSV 的一部分）做预测并与 `evaluation.txt` 中的统计进行核对。
3. 检查模型输出范围是否在 `[0,1]`。
4. 检查高风险召回率/误报率是否在可接受范围，若不满足需反馈给模型重训练或调整阈值。


-- 结束 --
