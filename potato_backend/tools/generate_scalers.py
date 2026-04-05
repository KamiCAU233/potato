"""
生成时序模型所需的 scaler 文件（scaler_temp.pkl 和 scaler_hum.pkl）
此脚本应该在模型训练后运行，从训练数据中计算出归一化参数。

使用说明：
1. 确保已有训练的模型（potato_blight_model.h5）
2. 如果有原始训练数据（CSV 或 NPZ），请传给此脚本来生成 scaler
3. 如果没有原始数据，可以从模型权重或训练元数据中恢复 scaler（需要额外配置）

这是一个示例脚本；实际的归一化参数应该来自你的训练代码。
"""
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent

def generate_scalers_from_data(temp_data: np.ndarray, hum_data: np.ndarray):
    """
    根据训练数据生成 scaler。
    
    Args:
        temp_data: 形状 (n_samples, 7) 的温度数据（推荐使用训练集）
        hum_data: 形状 (n_samples, 7) 的湿度数据（推荐使用训练集）
    
    Returns:
        scaler_temp, scaler_hum: sklearn 的 StandardScaler 对象
    
    说明：
        scaler 被拟合在所有样本和时间步的组合数据上。
        例如，(100, 7) 的数据会被展平为 (700,) 然后拟合。
        这样在预测时，可以对任意长度的温度/湿度序列进行转换。
    """
    # 确保数据是 (n_samples, 7) 形状
    assert temp_data.shape[1] == 7, f"温度数据应为 (n, 7)，实际 {temp_data.shape}"
    assert hum_data.shape[1] == 7, f"湿度数据应为 (n, 7)，实际 {hum_data.shape}"
    
    # 将数据展平为 (n_samples * 7, 1) 的形状以拟合 scaler
    temp_flat = temp_data.flatten().reshape(-1, 1)
    hum_flat = hum_data.flatten().reshape(-1, 1)
    
    # 分别创建和拟合 scaler
    scaler_temp = StandardScaler()
    scaler_hum = StandardScaler()
    
    scaler_temp.fit(temp_flat)
    scaler_hum.fit(hum_flat)
    
    return scaler_temp, scaler_hum


def save_scalers(scaler_temp, scaler_hum, save_dir: Path = None):
    """保存 scaler 到 pkl 文件"""
    if save_dir is None:
        save_dir = PROJECT_ROOT / "data" / "scalers"
    temp_path = save_dir / "scaler_temp.pkl"
    hum_path = save_dir / "scaler_hum.pkl"
    
    with open(temp_path, 'wb') as f:
        pickle.dump(scaler_temp, f)
    
    with open(hum_path, 'wb') as f:
        pickle.dump(scaler_hum, f)
    
    print(f"✓ scaler_temp.pkl 已保存到 {temp_path}")
    print(f"✓ scaler_hum.pkl 已保存到 {hum_path}")


def example_generate_scalers():
    """
    示例：使用虚拟数据生成 scaler（用于演示）。
    实际应用时，应该使用真实的训练数据调用此函数。
    """
    # 虚拟示例数据（你应该用真实的训练数据替换这里）
    n_samples = 100
    temp_data = np.random.randn(n_samples, 7) * 10 + 15  # 均值15℃，标准差10
    hum_data = np.random.randn(n_samples, 7) * 20 + 60    # 均值60%，标准差20
    
    print(f"生成虚拟数据：{n_samples} 个样本")
    print(f"  温度数据范围：[{temp_data.min():.2f}, {temp_data.max():.2f}]")
    print(f"  湿度数据范围：[{hum_data.min():.2f}, {hum_data.max():.2f}]")
    
    scaler_temp, scaler_hum = generate_scalers_from_data(temp_data, hum_data)
    save_scalers(scaler_temp, scaler_hum)
    
    print("\n✓ scaler 生成完成！")
    print(f"  温度 scaler mean={scaler_temp.mean_}, scale={scaler_temp.scale_}")
    print(f"  湿度 scaler mean={scaler_hum.mean_}, scale={scaler_hum.scale_}")


if __name__ == "__main__":
    print("=" * 60)
    print("时序模型 Scaler 生成脚本")
    print("=" * 60)
    
    # 检查是否 scaler 已存在
    scaler_temp_path = PROJECT_ROOT / "data" / "scalers" / "scaler_temp.pkl"
    scaler_hum_path = PROJECT_ROOT / "data" / "scalers" / "scaler_hum.pkl"
    
    if scaler_temp_path.exists() and scaler_hum_path.exists():
        print(f"✓ scaler 文件已存在，跳过生成")
        print(f"  - {scaler_temp_path}")
        print(f"  - {scaler_hum_path}")
    else:
        print("scaler 文件不存在，开始生成...")
        print("\n注意：")
        print("  如果你有真实的训练数据文件（如 train_temps.npy, train_hums.npy），")
        print("  请调用：")
        print("    temp_data = np.load('train_temps.npy')  # shape: (n, 7)")
        print("    hum_data = np.load('train_hums.npy')    # shape: (n, 7)")
        print("    generate_scalers_from_data(temp_data, hum_data)")
        print("\n  现在使用虚拟示例数据生成...\n")
        
        example_generate_scalers()
