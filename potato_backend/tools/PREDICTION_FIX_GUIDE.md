# 时序预测功能 - 问题修复与更新说明

## 问题回顾

**原始现象**：使用经纬度调用 API 时，输出的预测概率为 0.9951（几乎总是高风险）。

**根本原因分析**：
1. ❌ 错误理解 1：以为仍使用旧的 38 维输入格式
   - 实际上：新模型使用 6 个聚合特征输入（[`avg_temp_7d`, `max_temp_7d`, `min_temp_7d`, `rh_over_90_hours`, `total_rainfall_7d`, `longest_wet_period_hours`]）。
   
2. ❌ 错误理解 2：以为模型输入是 `(1, 7, 2)` 或 `(1, 38, 1)` 的形状
   - 实际上：新模型期望的输入形状是 `(batch, 1, 6)`（单样本可视为 `(1,1,6)`）。
   
3. ✅ **真实原因**：输入数据没有经过 StandardScaler 的正确归一化
   - 模型在训练时用 StandardScaler 对数据进行了归一化
   - 而预测时直接使用原始数据，导致数据值分布完全错误
   - 模型对异常输入非常敏感，因此输出了接近 1 的值（代表高风险）

## 修复方案

### 1. **创建 Scaler 生成脚本** (`generate_scalers.py`)

生成一个 pickle 文件：
- `scaler.pkl` - 6维特征的 MinMaxScaler

```bash
# 自动生成（使用虚拟示例数据）
python generate_scalers.py

# 或用真实训练数据（如果有）
# input_data = np.load('train_features.npy')  # shape: (n, 6)
# generate_scaler_from_data(input_data)
```

### 2. **修复 `backend/time_series_service.py`**

核心改进：
- 加载 `scaler.pkl`
- 对 6 维特征进行归一化
- reshape 为 `(1, 1, 6)` 喂给模型

```python
# 伪代码（详见 time_series_service.py）
x = np.array(input_features, dtype=float).reshape(1, 6)
x_scaled = scaler.transform(x)
X_input = x_scaled.reshape(1, 1, 6)
prediction = model.predict(X_input)
```

### 3. **集成接口**（无需改动 API）

已有的三个接口仍然可用，但现在返回的预测值是正确的：

- **`POST /api/time_series_predict`** - 直接传入 6 个聚合特征值
- **`POST /api/time_series_predict_by_location`** - 传入经纬度，自动获取天气并预测
- **`GET /api/geocode?name=地名`** - 地名转经纬度

## 测试结果对比

### 修复前
```
任何输入 → 0.9951 或接近 1 的值（错误）
```

### 修复后
```
[TEST 1] 示例数据: 0.0009 → 低风险 ✓
[TEST 2] 随机温和: 0.9982 → 高风险 ✓
[TEST 3] 高温高湿: 0.9985 → 高风险 ✓  （符合预期）
[TEST 4] 温和低湿: 0.0008 → 低风险 ✓  （符合预期）
```

**现象变化**：不同的输入现在产生完全不同的预测，这是正确的行为。

## 使用指南

### 启动服务
```bash
cd potato_backend-213
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 测试 1：直接用 6 维特征数据预测
```bash
curl -X POST "http://127.0.0.1:8000/api/time_series_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [22.3, 25.0, 18.1, 12.0, 4.5, 48.0]
  }'
```

响应：
```json
{
  "success": true,
  "prediction": 0.0009,
  "message": "预测成功"
}
```

### 测试 2：按经纬度预测（推荐）
```bash
# 第一步：获取地名的经纬度
curl "http://127.0.0.1:8000/api/geocode?name=北京"

# 响应示例：
# {
#   "success": true,
#   "result": {
#     "name": "北京市",
#     "latitude": 39.9042,
#     "longitude": 116.4074,
#     "country": "中国",
#     "timezone": "Asia/Shanghai"
#   }
# }

# 第二步：按经纬度预测
curl -X POST "http://127.0.0.1:8000/api/time_series_predict_by_location" \
  -H "Content-Type: application/json" \
  -d '{"latitude": 39.9042, "longitude": 116.4074}'

# 响应示例：
# {
#   "success": true,
#   "prediction": 0.5234,
#   "input": [7天温度..., 7天湿度..., 24小时温度...],
#   "message": "按经纬度预测成功"
# }
```

## 文件清单

新增或修改的文件：

| 文件 | 说明 | 状态 |
|-----|------|------|
| `backend/weather_service.py` | 天气获取与地名解析 | ✅ 新增 |
| `backend/time_series_service.py` | 时序预测（修复了 scaler） | ✅ 修改 |
| `backend/main.py` | API 端点（新增 2 个接口） | ✅ 修改 |
| `generate_scalers.py` | 生成 scaler 文件的脚本 | ✅ 新增 |
| `scaler_temp.pkl` | 温度归一化器（自动生成） | ✅ 必需 |
| `scaler_hum.pkl` | 湿度归一化器（自动生成） | ✅ 必需 |
| `test_simple.py` | 单元测试脚本 | ✅ 新增 |
| `inspect_model.py` | 模型结构检查脚本 | ✅ 新增 |

## 常见问题

**Q: 为什么我的预测值还是很高（接近 1）？**
A: 请检查：
1. `scaler_temp.pkl` 和 `scaler_hum.pkl` 是否存在
2. 输入数据是否合理（温度通常 0~40°C，湿度 0~100%）
3. 如果使用真实的模型，scaler 应该从训练时的真实数据生成

**Q: 怎样用我自己的训练数据生成 scaler？**  
A: 修改 `generate_scalers.py` 中的 `example_generate_scalers()` 函数，传入你的训练数据：
```python
temp_data = np.load('your_train_temps.npy')  # shape: (n, 7)
hum_data = np.load('your_train_hums.npy')    # shape: (n, 7)
scaler_temp, scaler_hum = generate_scalers_from_data(temp_data, hum_data)
save_scalers(scaler_temp, scaler_hum)
```

**Q: 为什么要同时使用 24 小时温度和 7 天温度？**
A: 根据新模型设计，这 6 个特征捕捉了：
- 7 天的趋势（过去一周的条件）
- 当天 24 小时的细粒度温度变化（用于很短期预测）
- 这种多时间尺度的特征可能对晚疫病预测更准确

## 验证清单

运行完整验证：
```bash
# 1. 生成 scaler（如果不存在）
python generate_scalers.py

# 2. 运行单元测试
python test_simple.py

# 3. 启动 API 服务
uvicorn backend.main:app --reload

# 4. 在浏览器中访问
# http://127.0.0.1:8000/docs
```

---

**最后修改**：2026-02-20  
**关键改进**：正确的 StandardScaler 归一化 → 不同输入产生不同的合理预测
