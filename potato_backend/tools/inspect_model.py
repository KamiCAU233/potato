"""
检查模型的真实输入/输出形状和结构
"""
import sys
sys.path.insert(0, r'c:/Users/18083/Desktop/potato_backend-213/potato_backend-213')

from tensorflow.keras.models import load_model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
model_path = PROJECT_ROOT / "potato_blight_model.h5"

if model_path.exists():
    model = load_model(str(model_path))
    print("=" * 70)
    print("模型结构说明")
    print("=" * 70)
    model.summary()
    print("\n" + "=" * 70)
    print("详细信息")
    print("=" * 70)
    print(f"模型名称: {model.name}")
    print(f"总层数: {len(model.layers)}")
    print(f"\n输入形状: {model.input_shape}")
    print(f"输出形状: {model.output_shape}")
    
    print(f"\n各层详情:")
    for idx, layer in enumerate(model.layers):
        print(f"  [{idx}] {layer.name:20s} - {layer.__class__.__name__:15s} - input: {layer.input_shape}, output: {layer.output_shape}")
else:
    print(f"模型文件未找到: {model_path}")
