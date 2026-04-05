# 项目结构说明

整理后的项目目录结构如下：

```
potato_backend-213/
├── backend/                          # 后端核心代码
│   ├── __init__.py
│   ├── main.py                       # FastAPI 主应用
│   ├── config.py                     # 配置文件
│   ├── predict_service.py            # YOLO 图片识别
│   ├── time_series_service.py        # 时序预测（已修复scaler路径）
│   ├── weather_service.py            # 天气获取 + 地名解析
│   ├── llm_service.py                # 大模型问答
│   ├── monthly_tip.py                # 月份提示
│   ├── db.py                         # 数据库
│   ├── requirements.txt              # 依赖列表
│   └── __pycache__/
│
├── data/
│   ├── scalers/                      # ⭐ 时序模型的分步缩放器
│   │   ├── scaler_temp.pkl           # 温度归一化器（生成一次，永久使用）
│   │   └── scaler_hum.pkl            # 湿度归一化器（生成一次，永久使用）
│   └── [其他数据文件...]
│
├── tools/                            # 🛠️ 开发工具与测试脚本
│   ├── generate_scalers.py           # 生成scaler的脚本（高级用途，通常不需要用）
│   ├── inspect_model.py              # 查看模型结构
│   ├── test_simple.py                # 简单测试
│   ├── test_new_prediction.py        # 完整测试
│   ├── test_time_series.py           # 时序测试
│   ├── PREDICTION_FIX_GUIDE.md       # 预测修复详细说明
│   └── DEPLOYMENT.md                 # 部署指南
│
├── runs/                             # YOLO 训练结果
├── uploads/                          # 用户上传的图片
├── weights/                          # YOLO 预训练权重
│
├── potato_blight_model.h5            # 时序预测模型
├── llm_extra_prompt.txt              # LLM 提示词
├── llm_fewshot.txt                   # LLM 示例
├── llm_knowledge.txt                 # LLM 知识库
├── backend_package_README.txt        # 包信息
├── 前端对接说明.md                   # ⭐ 前端必读
├── .env                              # 环境变量配置
└── requirements.txt                  # （可选）全局依赖

```

## 📌 关键说明

### Scaler 文件
- 📍 **位置**：`data/scalers/scaler_*.pkl`
- 💾 **生成频率**：一次性生成，之后永久可用
- 🔧 **管理**：通常不需要操作，高级用途可用 `tools/generate_scalers.py` 重新生成

### 工具目录
- 📍 **位置**：`tools/` 目录
- 📄 **包含**：测试脚本、辅助脚本、详细文档
- 🚀 **用途**：开发和调试时使用，部署到生产环不需要发布

### 前端接口
- 📍 **文档**：`前端对接说明.md`（根目录）
- 🌐 **API 文档**：启动服务后访问 `http://127.0.0.1:8000/docs`

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r backend/requirements.txt

# 2. 启动服务
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 3. 访问 API 文档
# 浏览器打开 http://127.0.0.1:8000/docs
```

## 📦 打包部署

打包时需要包含以下**必要文件**：
- ✅ `backend/` - 核心代码
- ✅ `data/scalers/` - Scaler 文件（必需！）
- ✅ `potato_blight_model.h5` - 时序预测模型
- ✅ `llm_*.txt` - 提示词文件
- ✅ `backend_package_README.txt`
- ✅ `前端对接说明.md` - 使用文档

不需要打包：
- ❌ `tools/` - 开发工具，不需要发布
- ❌ `runs/` - 训练结果，按需包含
- ❌ `uploads/` - 用户数据，不需要打包
- ❌ `.env` - 本地配置，用户自行配置

## ✅ 验证清单

部署前确认：
- [ ] `data/scalers/scaler_temp.pkl` 存在
- [ ] `data/scalers/scaler_hum.pkl` 存在
- [ ] `backend/weather_service.py` 存在
- [ ] `potato_blight_model.h5` 存在
- [ ] `pip install -r backend/requirements.txt` 成功

---

**最后更新**：2026-02-20  
**整理内容**：将 test_*.py、scaler_*.pkl 等临时文件整理到 tools/ 和 data/scalers/ 目录
