========================================
  马铃薯病害识别 - 后端运行包（前端可本地运行）
========================================

【环境】
  Python 3.8+，建议 3.9/3.10

【解压后目录结构】
  当前文件夹/
    backend/           # 后端代码
    runs/train/xxx/weights/best.pt   # 识别模型（若打包时包含）
    weights/           # 预训练权重（可选）
    backend_package_README.txt   # 本说明

【运行步骤】
1. 打开终端，进入当前文件夹（与 backend 同级）
2. 安装依赖：pip install -r backend/requirements.txt
3. （可选）配置大模型：在当前文件夹新建 .env 文件，写入：
     LLM_BASE_URL=https://api.deepseek.com
     LLM_API_KEY=你的key
     LLM_MODEL=deepseek-chat
   不配置则识别可用，问答/解读接口会返回「大模型未启用」
4. （可选）配置扣子智能体：在 .env 文件中添加：
     ENABLE_COZE=true
     COZE_BOT_ID=你的机器人ID
     COZE_PAT=你的个人访问令牌
   不配置则扣子问答接口会返回「扣子智能体未启用」
5. （可选）配置时序预测模型：将你的 h5 模型文件放入项目根目录，命名为 potato_blight_model.h5，或在 .env 中设置 TIME_SERIES_MODEL_PATH=你的模型路径
6. 启动服务：uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
6. 浏览器访问：http://127.0.0.1:8000/docs 可测试所有接口

【接口列表】
  GET  /                    根路径说明
  GET  /api/llm_status      大模型是否可用（调试）
  GET  /api/coze_status     扣子智能体是否可用（调试）
  GET  /api/monthly_tip     当前月份易发病提示
  POST /api/predict         上传图片识别（file=图片），返回 results（最多1条）+ image_result + monthly_tip
  POST /api/ask             大模型问答，body: {"question":"..." , "history": 可选}，支持多轮
  POST /api/coze/ask        扣子智能体问答，body: {"question":"..." , "history": 可选}，支持多轮
  POST /api/predict_with_explanation  上传图片，识别+大模型解读，返回 results + image_result + explanation + monthly_tip
  POST /api/time_series_predict  时序预测发病风险，body: {"data": [float数组，长度38]}，返回 prediction

【前端调用】
  Base URL：本地 http://127.0.0.1:8000，部署后改为实际地址
  识别：POST /api/predict，multipart file 字段名 file
  大模型问答：POST /api/ask，JSON {"question":"字符串", "history": 可选数组}
  扣子智能体问答：POST /api/coze/ask，JSON {"question":"字符串", "history": 可选数组}
  识别+解读：POST /api/predict_with_explanation，同 predict 的 file 上传
  时序预测：POST /api/time_series_predict，JSON {"data": [38个float值：7天温度 + 7天湿度 + 24小时温度]}

【若无 best.pt】
  若解压后没有 runs/train/xxx/weights/best.pt，识别会退化为使用 weights/yolov8n.pt（未训练）。需在项目方处获取 best.pt 放入 runs/train/某目录/weights/ 下再运行。
