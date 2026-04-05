本文件由AI生成，暂时没有什么用，预计以后可能也没有什么用。
本文件由AI生成，暂时没有什么用，预计以后可能也没有什么用。
本文件由AI生成，暂时没有什么用，预计以后可能也没有什么用。




# 后端部署指南

## 📦 打包前须知

### 必须包含的文件

在生成 `potato_backend_package.zip` 时，**必须** 包含以下文件：

```
potato_backend_package/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── predict_service.py
│   ├── time_series_service.py        ← 新增/更新
│   ├── weather_service.py            ← 新增
│   ├── llm_service.py
│   ├── monthly_tip.py
│   ├── db.py
│   ├── requirements.txt
│   └── __pycache__/
├── data/
│   └── scalers/                       ← 新增（必需！）
│       ├── scaler_temp.pkl
│       └── scaler_hum.pkl
├── potato_blight_model.h5            ← 已有
├── llm_extra_prompt.txt              ← 已有
├── llm_fewshot.txt                   ← 已有
├── llm_knowledge.txt                 ← 已有
├── backend_package_README.txt        ← 已有
├── 前端对接说明.md                   ← 已有
└── [其他现有文件...]
```

### ⚠️ 关键说明

- **`data/scalers/scaler_temp.pkl` 和 `scaler_hum.pkl`**：时序预测必需的数据，无需每次运行脚本重新生成
- **一旦打包，永久可用**：这两个 scaler 文件包含了标准化参数，与模型绑定
- **若修改了模型**：才需要重新运行 `tools/generate_scalers.py` 并更新这两个文件

---

## 🚀 部署步骤

### 前端用户

```bash
# 1. 解压包
unzip potato_backend_package.zip
cd potato_backend_package

# 2. 安装依赖（仅需一次）
pip install -r backend/requirements.txt

# 3. 启动服务
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 4. 验证
# 浏览器访问 http://127.0.0.1:8000/docs
```

### 服务器部署（生产环境）

#### 使用 Gunicorn + Nginx

```bash
# 1. 安装 gunicorn
pip install gunicorn

# 2. 启动服务（4 个 worker，监听 0.0.0.0:8000）
gunicorn -w 4 -b 0.0.0.0:8000 backend.main:app

# 3. 配置 Nginx 反向代理（可选）
# 见下方配置示例
```

#### Nginx 配置示例

```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 使用 Docker（推荐）

创建 `Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建与运行：

```bash
docker build -t potato-backend .
docker run -p 8000:8000 potato-backend
```

---

## 📋 文件检查清单

打包前，请确认以下文件都已存在：

- [ ] `data/scalers/scaler_temp.pkl`（通常 ~400 字节）
- [ ] `data/scalers/scaler_hum.pkl`（通常 ~400 字节）
- [ ] `potato_blight_model.h5`（模型文件）
- [ ] `backend/weather_service.py`（天气获取模块）
- [ ] `backend/main.py`（包含新增接口）
- [ ] `backend/time_series_service.py`（包含修复后的预测逻辑）
- [ ] `backend/requirements.txt`（包含所有依赖）

如果缺少任何文件，尤其是 scaler 文件，API 将无法正常工作。

---

## ✅ 验证清单

部署完成后，逐一验证：

```bash
# 1. 检查 API 文档是否可访问
curl http://your_server:8000/docs

# 2. 检查根路由
curl http://your_server:8000/

# 3. 测试图片识别
curl -X POST http://your_server:8000/api/predict \
  -F "file=@test_image.jpg"

# 4. 测试地名转经纬度
curl "http://your_server:8000/api/geocode?name=北京"

# 5. 测试时序预测
curl -X POST http://your_server:8000/api/time_series_predict_by_location \
  -H "Content-Type: application/json" \
  -d '{"latitude": 39.9042, "longitude": 116.4074}'
```

---

## 🔧 常见问题

### Q: 服务启动后无法加载时序模型？

**A:** 检查以下问题：
1. `potato_blight_lstm_model.keras` 是否存在（或通过 `TIME_SERIES_MODEL_PATH` 指定路径）
2. `scaler.pkl` 是否存在（`data/scalers/scaler.pkl`）
3. TensorFlow 是否正确安装（`pip install tensorflow`）

### Q: 时序预测总是返回错误？

**A:** 常见原因：
1. **缺少 scaler 文件**：确保 `scaler.pkl` 在项目根目录（`data/scalers/`）
2. **输入数据格式错误**：确保传入的是长度 6 的数组
3. **天气 API 无法访问**：确保服务器可以访问 Open-Meteo API（https://api.open-meteo.com）

### Q: 可以修改天气数据源吗？

**A:** 可以。编辑 `backend/weather_service.py` 的 `fetch_weather_for_model()` 函数，替换为其他天气 API（如高德、百度等）。

---

## 📞 技术支持

如有问题，请提供以下信息：
- 错误日志（完整的 traceback）
- 使用的 Python 版本（`python --version`）
- TensorFlow 版本（`pip show tensorflow`）
- 是否能访问 Open-Meteo API

---

**最后更新**：2026-02-20  
**关键点**：scaler 文件**永久可用**，一次打包后无需重新生成
