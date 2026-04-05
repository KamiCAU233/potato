# 时序预测服务，加载h5模型、scaler 并进行预测
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import os
import pickle
import joblib

from backend.config import TIME_SERIES_MODEL_PATH, TIME_SERIES_INPUT_SIZE, PROJECT_ROOT

_model_ts = None
_scaler = None


def get_time_series_model():
    """单例加载时序模型"""
    global _model_ts
    if _model_ts is None:
        model_path = Path(TIME_SERIES_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"时序模型文件未找到: {model_path}")
        print(f"[backend] 加载时序模型: {model_path}")
        _model_ts = load_model(str(model_path))
    return _model_ts


def get_scaler():
    """单例加载 scaler（优先 joblib scaler.pkl, 兼容旧 scaler_temp+scaler_hum）。"""
    global _scaler
    if _scaler is None:
        scaler_path = PROJECT_ROOT / "data" / "scalers" / "scaler.pkl"
        if scaler_path.exists():
            print(f"[backend] 加载 scaler: {scaler_path}")
            try:
                _scaler = joblib.load(str(scaler_path))
            except Exception as e:
                # joblib 格式异常，尝试 pickle（兼容性）
                print(f"[backend] joblib.load 失败（尝试 pickle）: {e}")
                _scaler = pickle.load(open(str(scaler_path), 'rb'))
            return _scaler

        # 兼容旧版本：scaler_temp/scaler_hum
        scaler_temp_path = PROJECT_ROOT / "data" / "scalers" / "scaler_temp.pkl"
        scaler_hum_path = PROJECT_ROOT / "data" / "scalers" / "scaler_hum.pkl"
        if scaler_temp_path.exists() and scaler_hum_path.exists():
            print(f"[backend] 使用旧 scaler_temp/scaler_hum（按旧 38 维模型逻辑已弃用，此处仅为兼容）")
            scaler_temp = joblib.load(str(scaler_temp_path)) if scaler_temp_path.exists() else pickle.load(open(str(scaler_temp_path), 'rb'))
            scaler_hum = joblib.load(str(scaler_hum_path)) if scaler_hum_path.exists() else pickle.load(open(str(scaler_hum_path), 'rb'))
            # 返回特殊对象，predict_time_series 不再使用此案例（历史兼容）
            _scaler = (scaler_temp, scaler_hum)
            return _scaler

        raise FileNotFoundError(f"scaler 文件未找到: {scaler_path} 或 scaler_temp/scaler_hum")
    return _scaler


def predict_time_series(data: list):
    """
    对时序数据进行预测（新模型，使用 scaler.pkl 归一化）。
    输入: data - 长度为6的float列表
        [avg_temp_7d, max_temp_7d, min_temp_7d, rh_over_90_hours, total_rainfall_7d, longest_wet_period_hours]
    返回: (prediction, error_message 或 None)
    prediction: 预测的发病概率 (float，0~1 范围)
    模型期望输入形状: (1, 1, 6)
    """
    if len(data) != TIME_SERIES_INPUT_SIZE:
        return None, f"输入数据长度错误，期望{TIME_SERIES_INPUT_SIZE}，实际{len(data)}"

    try:
        x = np.array(data, dtype=float).reshape(1, -1)  # (1, 6)

        model = get_time_series_model()
        scaler = get_scaler()

        if isinstance(scaler, tuple):
            return None, "当前仅支持 scaler.pkl（joblib）格式的 6 维新模型输入，请提供 scaler.pkl 或更新模型配置。"

        x_scaled = scaler.transform(x)  # (1, 6)
        x_input = x_scaled.reshape(1, 1, x_scaled.shape[1])  # (1, 1, 6)

        prediction = model.predict(x_input, verbose=0)
        pred_value = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])

        return pred_value, None
    except FileNotFoundError as e:
        return None, str(e) + " — 请先准备模型与 scaler 文件"
    except Exception as e:
        return None, str(e)