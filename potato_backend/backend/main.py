# 马铃薯病害识别 - 后端 API（FastAPI）
# 运行：在项目根目录执行  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
import base64
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import MAX_UPLOAD_MB, ALLOWED_EXTENSIONS, ENABLE_LLM, LLM_API_KEY, LLM_BASE_URL, ENABLE_COZE, COZE_BOT_ID, COZE_PAT
from backend.predict_service import predict_image
from backend.llm_service import is_llm_available, ask, explain_detection, is_coze_available, ask_coze
from backend.monthly_tip import get_monthly_tip
from backend.time_series_service import predict_time_series
from backend.weather_service import fetch_weather_for_model, geocode_place
from backend.db import (
    init_db,
    save_upload_record,
    list_upload_records,
    get_record_by_id,
    create_llm_session,
    add_message,
    list_llm_sessions,
    get_messages_by_session,
)
from fastapi.responses import FileResponse

app = FastAPI(
    title="马铃薯病害识别 API",
    description="上传土豆块茎或叶片图片，返回病害识别结果；可选大模型解读与病害问答。",
    version="1.1",
)

# 初始化数据库（若不存在则创建）
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


class ChatMessage(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str


class AskBody(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None  # 可选，上一轮对话，用于多轮上下文


class TimeSeriesPredictBody(BaseModel):
    data: List[float]  # 时序数据列表，长度6：[avg_temp_7d, max_temp_7d, min_temp_7d, rh_over_90_hours, total_rainfall_7d, longest_wet_period_hours]


class LocationBody(BaseModel):
    latitude: float
    longitude: float


@app.get("/", summary="根路径")
def root():
    """接口说明与文档入口。"""
    return {
        "message": "马铃薯病害识别 API",
        "docs": "/docs",
        "predict": "POST /api/predict",
        "predict_with_explanation": "POST /api/predict_with_explanation",
        "ask": "POST /api/ask",
        "coze_ask": "POST /api/coze/ask",
        "llm_status": "GET /api/llm_status",
        "coze_status": "GET /api/coze_status",
        "monthly_tip": "GET /api/monthly_tip",
        "time_series_predict": "POST /api/time_series_predict",
    }


@app.get("/api/coze_status", summary="扣子智能体配置状态（调试用）")
def api_coze_status():
    """查看当前后端是否读到 Coze 配置，不暴露 PAT。若 has_bot_id 与 has_pat 都为 true，说明已配置。"""
    return {
        "coze_available": is_coze_available(),
        "enable_coze": ENABLE_COZE,
        "has_bot_id": bool(COZE_BOT_ID),
        "has_pat": bool(COZE_PAT),
    }


@app.post("/api/predict", summary="上传图片识别病害")
async def api_predict(file: UploadFile = File(..., description="土豆/叶片图片（jpg、png 等）")):
    """上传一张图片，返回检测结果：病害类别（中文）、置信度、检测框坐标，以及带框结果图的 base64 编码。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持图片格式: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    body = await file.read()
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"文件大小超过 {MAX_UPLOAD_MB}MB 限制")
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    results, plotted, err = predict_image(body, return_plot=True)
    if err is not None:
        raise HTTPException(status_code=500, detail=err)

    image_result = None
    if plotted is not None:
        import cv2
        _, buf = cv2.imencode(".jpg", plotted)
        image_result = base64.b64encode(buf.tobytes()).decode("utf-8")
        plotted_bytes = buf.tobytes()
    else:
        plotted_bytes = None

    # 保存上传记录（图片字节、绘制后的图、检测结果）
    try:
        save_upload_record(file.filename, body, plotted_bytes, results, None)
    except Exception:
        # 不阻塞主流程：保存失败不影响返回
        pass

    return {
        "success": True,
        "results": results,
        "image_result": image_result,
        "monthly_tip": get_monthly_tip(),
    }


@app.get("/api/monthly_tip", summary="当前月份易发病提示")
def api_monthly_tip():
    """返回按季节/月份生成的「当前易发病」一句提示，供前端展示或播报。"""
    return {"monthly_tip": get_monthly_tip()}


@app.post("/api/ask", summary="病害问答（大模型，支持多轮上下文）")
async def api_ask(body: AskBody):
    """提交一个问题，返回大模型关于马铃薯病害/种植的回答。可传 history 为上一轮对话列表以实现连续追问。大模型未启用时返回 success=false 与 message。"""
    history = None
    if body.history:
        history = [{"role": m.role, "content": m.content} for m in body.history]

    # 创建一个会话并记录用户问题与模型返回
    session_id = None
    try:
        session_id = create_llm_session(user_id=None, metadata={"source": "api_ask"})
        add_message(session_id, "user", body.question)
    except Exception:
        session_id = None

    answer, err = ask(body.question, history=history)
    if err is not None:
        return {"success": False, "answer": None, "message": err}

    try:
        if session_id is not None:
            add_message(session_id, "assistant", answer)
    except Exception:
        pass

    return {"success": True, "answer": answer, "message": None, "session_id": session_id}


@app.post("/api/coze/ask", summary="扣子智能体问答（支持多轮上下文）")
async def api_coze_ask(body: AskBody):
    """提交一个问题，返回扣子智能体的回答。可传 history 为上一轮对话列表以实现连续追问。扣子智能体未启用时返回 success=false 与 message。"""
    history = None
    if body.history:
        history = [{"role": m.role, "content": m.content} for m in body.history]

    answer, err = ask_coze(body.question, history=history)
    if err is not None:
        return {"success": False, "answer": None, "message": err}

    return {"success": True, "answer": answer, "message": None}


@app.get("/api/chat/sessions", summary="列出 LLM 会话")
def api_list_chat_sessions(limit: int = 100, offset: int = 0):
    items = list_llm_sessions(limit=limit, offset=offset)
    return {"success": True, "items": items}


@app.get("/api/chat/{session_id}", summary="获取会话消息")
def api_get_chat_session(session_id: int):
    msgs = get_messages_by_session(session_id)
    return {"success": True, "session_id": session_id, "messages": msgs}


@app.post("/api/predict_with_explanation", summary="上传图片识别病害并生成解读")
async def api_predict_with_explanation(file: UploadFile = File(..., description="土豆/叶片图片（jpg、png 等）")):
    """上传一张图片：先做病害检测，再用大模型根据检测结果生成简要解读与防治建议。未启用大模型时仍返回检测结果，explanation 为 null、message 说明未启用。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持图片格式: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    body = await file.read()
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"文件大小超过 {MAX_UPLOAD_MB}MB 限制")
    if len(body) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    results, plotted, err = predict_image(body, return_plot=True)
    if err is not None:
        raise HTTPException(status_code=500, detail=err)

    image_result = None
    if plotted is not None:
        import cv2
        _, buf = cv2.imencode(".jpg", plotted)
        image_result = base64.b64encode(buf.tobytes()).decode("utf-8")
        plotted_bytes = buf.tobytes()
    else:
        plotted_bytes = None

    explanation = None
    msg = None
    if is_llm_available():
        explanation, msg = explain_detection(results)
    else:
        msg = "大模型未启用或未配置（请设置 ENABLE_LLM、LLM_API_KEY 或 LLM_BASE_URL）"

    # 保存上传记录（图片字节、绘制后的图、检测结果、解读）
    try:
        save_upload_record(file.filename, body, plotted_bytes, results, explanation)
    except Exception:
        pass

    return {
        "success": True,
        "results": results,
        "image_result": image_result,
        "explanation": explanation,
        "message": msg,
        "monthly_tip": get_monthly_tip(),
    }


@app.get("/api/uploads", summary="列出上传记录")
def api_list_uploads(limit: int = 100, offset: int = 0):
    """返回最近的上传记录列表（不包含图片二进制，只包含路径/元数据）。"""
    items = list_upload_records(limit=limit, offset=offset)
    return {"success": True, "items": items}


@app.get("/api/uploads/{record_id}", summary="获取单条上传记录详情")
def api_get_upload(record_id: int):
    r = get_record_by_id(record_id)
    if not r:
        raise HTTPException(status_code=404, detail="Record not found")
    return {
        "id": r.id,
        "original_filename": r.original_filename,
        "stored_path": r.stored_path,
        "plotted_path": r.plotted_path,
        "results": r.results_json,
        "explanation": r.explanation,
        "created_at": r.created_at.isoformat(),
    }


@app.get("/api/uploads/{record_id}/image", summary="下载上传的图片或绘制结果")
def api_get_upload_image(record_id: int, which: str = "original"):
    r = get_record_by_id(record_id)
    if not r:
        raise HTTPException(status_code=404, detail="Record not found")
    if which == "original":
        path = r.stored_path
    else:
        path = r.plotted_path or r.stored_path
    full = Path(__file__).resolve().parent.parent / path
    if not full.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(full)


@app.post("/api/time_series_predict", summary="时序预测发病风险")
async def api_time_series_predict(body: TimeSeriesPredictBody):
    """提交时序数据（新模型6维特征），返回发病风险预测值。
    数据格式：
      [avg_temp_7d, max_temp_7d, min_temp_7d, rh_over_90_hours, total_rainfall_7d, longest_wet_period_hours]
    """
    prediction, err = predict_time_series(body.data)
    if err is not None:
        raise HTTPException(status_code=500, detail=err)

    return {
        "success": True,
        "prediction": prediction,
        "message": "预测成功"
    }


@app.post("/api/time_series_predict_by_location", summary="按经纬度获取天气并预测发病风险")
async def api_time_series_predict_by_location(body: LocationBody):
    """接收经纬度，调用 Open-Meteo 获取气象数据并整合时序模型进行预测。
    返回预测概率与用于预测的输入数据（可用于调试）。
    """
    data, err = fetch_weather_for_model(body.latitude, body.longitude)
    if err is not None:
        raise HTTPException(status_code=500, detail=err)

    prediction, err = predict_time_series(data)
    if err is not None:
        raise HTTPException(status_code=500, detail=err)

    return {
        "success": True,
        "prediction": prediction,
        "input": data,
        "message": "按经纬度预测成功",
    }


@app.get("/api/geocode", summary="地名转经纬度（Open-Meteo geocoding）")
def api_geocode(name: str):
    """通过地名查询经纬度，返回第一个匹配结果（如无匹配返回 404）。"""
    result, err = geocode_place(name)
    if err is not None:
        raise HTTPException(status_code=404, detail=err)
    return {"success": True, "result": result}