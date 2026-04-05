# 后端配置（端口、模型、推理参数等）
import os
from pathlib import Path

# 项目根目录（backend 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 从项目根目录的 .env 加载环境变量（若有），不依赖 python-dotenv
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and v:
                        os.environ.setdefault(k, v)
    except Exception:
        pass

# 服务
HOST = "0.0.0.0"
PORT = 8000

# 上传限制
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 推理（与 detect_app 一致；置信度调低可看到更多结果，列表不易为空）
IMGSZ = 640
CONF_THRESH = 0.15
IOU_THRESH = 0.45
# 设备选择：默认使用环境变量 `DEVICE`，若未设置则回退到 "cpu"。
# 若你有 GPU 并希望使用 CUDA，请在启动前设置环境变量为数字（例如 0，或 '0,1' 多卡）
DEVICE = os.environ.get("DEVICE", "cpu")

# 大模型（OpenAI 兼容 API，如 DeepSeek / 智谱 / Ollama）
# 不配置或 ENABLE_LLM=false 时，/api/ask 与 /api/predict_with_explanation 返回「大模型未启用」
ENABLE_LLM = os.environ.get("ENABLE_LLM", "true").lower() in ("1", "true", "yes")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip() or None  # 空则用 OpenAI 默认；Ollama 可填 http://127.0.0.1:11434/v1
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
# 问答时若用户问「你是谁」，用这个名字回答；可写在 .env：LLM_ASSISTANT_NAME=你起的名字
LLM_ASSISTANT_NAME = os.environ.get("LLM_ASSISTANT_NAME", "马铃薯病害小助手").strip() or "马铃薯病害小助手"

# 扣子智能体（Coze）配置
ENABLE_COZE = os.environ.get("ENABLE_COZE", "false").lower() in ("1", "true", "yes")
COZE_BOT_ID = os.environ.get("COZE_BOT_ID", "")
COZE_PAT = os.environ.get("COZE_PAT", "")

# 时序预测模型（新模型 potato_blight_lstm_model.keras，输入6维特征）
TIME_SERIES_MODEL_PATH = os.environ.get("TIME_SERIES_MODEL_PATH", str(PROJECT_ROOT / "potato_blight_lstm_model.keras"))
TIME_SERIES_INPUT_SIZE = 6  # avg_temp_7d,max_temp_7d,min_temp_7d,rh_over_90_hours,total_rainfall_7d,longest_wet_period_hours
