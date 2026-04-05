# 封装模型加载与单张图推理，供 API 调用
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from backend.config import PROJECT_ROOT, IMGSZ, CONF_THRESH, IOU_THRESH, DEVICE

CLASS_NAMES_CN = {
    0: "块茎黑痂病", 1: "块茎黑胫病", 2: "块茎普通疮痂病", 3: "块茎干腐病",
    4: "健康块茎", 5: "块茎其他病害", 6: "块茎粉红腐病",
    7: "健康叶片", 8: "叶片早疫病", 9: "叶片晚疫病",
}


def _find_latest_best_pt():
    train_dir = PROJECT_ROOT / "runs" / "train"
    if not train_dir.is_dir():
        return None
    candidates = []
    for d in train_dir.iterdir():
        if not d.is_dir():
            continue
        w = d / "weights" / "best.pt"
        w_student = d / "weights" / "best_student.pt"  # distill 提取的纯学生，部署用
        w = w_student if w_student.exists() else w
        if w.exists():
            name = d.name
            if "distill" in name:
                try:
                    suffix = name.replace("potato_disease_10class_distill", "").strip() or "0"
                    num = 10000 + int(suffix)
                except ValueError:
                    num = 9999
            elif name == "potato_disease_10class":
                num = 0
            elif name.startswith("potato_disease_10class"):
                try:
                    num = int(name.replace("potato_disease_10class", "") or "0")
                except ValueError:
                    num = 0
            else:
                num = -1
            candidates.append((num, w))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


_model = None


def get_model():
    """单例加载模型"""
    global _model
    if _model is None:
        model_path = _find_latest_best_pt()
        if model_path is None:
            model_path = PROJECT_ROOT / "weights" / "yolov8n.pt"
            if not model_path.exists():
                model_path = PROJECT_ROOT / "yolov8n.pt"
        print(f"[backend] 使用模型: {model_path}")
        _model = YOLO(str(model_path))
    return _model


def predict_image(image_bytes: bytes, conf_thresh: float = CONF_THRESH, return_plot: bool = True):
    """
    对图片字节进行推理。
    返回: (results_list, plotted_bgr_numpy 或 None, error_message 或 None)
    results_list: [ {"class": "叶片早疫病", "confidence": 0.95, "box": [x1,y1,x2,y2] }, ... ]
    """
    try:
        import cv2
    except ImportError:
        return [], None, "请安装 opencv-python"

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [], None, "无法解析图片"

    model = get_model()
    results = model.predict(
        source=img,
        imgsz=IMGSZ,
        device=DEVICE,
        conf=conf_thresh,
        iou=IOU_THRESH,
        verbose=False,
    )
    r = results[0]
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cid = int(box.cls.item())
            conf = float(box.conf.item())
            name_cn = CLASS_NAMES_CN.get(cid, f"class_{cid}")
            xyxy = box.xyxy[0].tolist()
            out.append({
                "class": name_cn,
                "confidence": round(conf, 4),
                "box": [round(x, 2) for x in xyxy],
            })
    # 只保留置信度最高的一条，作为「最可能的病害」
    out.sort(key=lambda x: x["confidence"], reverse=True)
    out = out[:1] if out else []
    plotted = r.plot() if return_plot else None
    return out, plotted, None
