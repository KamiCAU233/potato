# 时序预测API测试示例
import requests

def test_time_series_predict():
    """测试时序预测API的示例函数"""

    # 示例数据：6个float值（新模型输入）
    # 格式：[avg_temp_7d, max_temp_7d, min_temp_7d, rh_over_90_hours, total_rainfall_7d, longest_wet_period_hours]

    # 这里使用示例数据
    sample_data = [22.0, 27.0, 16.0, 12.0, 8.5, 6.0]

    # API请求
    url = "http://127.0.0.1:8000/api/time_series_predict"
    payload = {"data": sample_data}

    try:
        response = requests.post(url, json=payload)
        result = response.json()

        if result["success"]:
            print(f"预测成功！发病风险值: {result['prediction']:.4f}")
        else:
            print(f"预测失败: {result['message']}")

    except Exception as e:
        print(f"请求错误: {e}")

if __name__ == "__main__":
    test_time_series_predict()