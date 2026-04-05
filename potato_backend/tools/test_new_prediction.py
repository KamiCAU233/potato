"""
时序模型预测测试脚本 - 验证新的 scaler 和预测逻辑
"""
import sys
sys.path.insert(0, r'c:/Users/18083/Desktop/potato_backend-213/potato_backend-213')

from backend.time_series_service import predict_time_series

print("=" * 70)
print("时序模型预测测试 (新模型 6 维特征)")
print("=" * 70)

# 测试用例 1：标准特征输入
print("\n[测试1] 标准特征输入")
input_raw_1 = [22.3, 25.0, 18.1, 10.0, 4.5, 48.0]

pred1, err1 = predict_time_series(input_raw_1)
if err1:
    print(f"  ❌ 预测失败: {err1}")
else:
    print(f"  ✓ 预测概率: {pred1:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险' if pred1 >= 0.6 else '🟢 低风险'}")

# 测试用例 2：极端高湿高雨
print("\n[测试2] 极端高湿高雨")
input_raw_2 = [28.0, 32.0, 18.0, 40.0, 60.0, 36.0]

pred2, err2 = predict_time_series(input_raw_2)
if err2:
    print(f"  ❌ 预测失败: {err2}")
else:
    print(f"  ✓ 预测概率: {pred2:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险' if pred2 >= 0.6 else '🟢 低风险'}")

# 测试用例 3：低风险条件
print("\n[测试3] 低风险条件")
input_raw_3 = [15.0, 18.0, 10.0, 2.0, 0.5, 2.0]

pred3, err3 = predict_time_series(input_raw_3)
if err3:
    print(f"  ❌ 预测失败: {err3}")
else:
    print(f"  ✓ 预测概率: {pred3:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险' if pred3 >= 0.6 else '🟢 低风险'}")

# 测试用例 4：异常输入长度检查
print("\n[测试4] 异常输入长度检查")
input_wrong_length = [20.0] * 20
pred4, err4 = predict_time_series(input_wrong_length)
if err4:
    print(f"  ✓ 正确捕获错误: {err4}")
else:
    print("  ❌ 未检测到错误!")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)

pred1, err1 = predict_time_series(input_raw_1)
if err1:
    print(f"  ❌ 预测失败: {err1}")
else:
    print(f"  ✓ 预测概率: {pred1:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险（发生晚疫病）' if pred1 > 0.5 else '🟢 低风险（不发生晚疫病）'}")

# 测试用例 2：生成一些随机数据，测试处理流程
print("\n[测试2] 随机构造的数据（温度10-30°C，湿度30-80%）")
temp_random = np.random.uniform(10, 30, 7).tolist()
hum_random = np.random.uniform(30, 80, 7).tolist()
hourly_temp_random = np.random.uniform(12, 28, 24).tolist()
input_raw_2 = temp_random + hum_random + hourly_temp_random

pred2, err2 = predict_time_series(input_raw_2)
if err2:
    print(f"  ❌ 预测失败: {err2}")
else:
    print(f"  ✓ 预测概率: {pred2:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险（发生晚疫病）' if pred2 > 0.5 else '🟢 低风险（不发生晚疫病）'}")
    print(f"  📊 输入数据统计:")
    print(f"     - 温度范围: [{min(temp_random):.1f}, {max(temp_random):.1f}]°C")
    print(f"     - 湿度范围: [{min(hum_random):.1f}, {max(hum_random):.1f}]%")

# 测试用例 3：模拟极端高温高湿的情况
print("\n[测试3] 极端条件（高温高湿 - 易发病）")
temp_hot_wet = [25.0] * 7  # 7天都是25°C
hum_hot_wet = [85.0] * 7   # 7天都是85%湿度
hourly_hot_wet = [24.0] * 24
input_raw_3 = temp_hot_wet + hum_hot_wet + hourly_hot_wet

pred3, err3 = predict_time_series(input_raw_3)
if err3:
    print(f"  ❌ 预测失败: {err3}")
else:
    print(f"  ✓ 预测概率: {pred3:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险（发生晚疫病）' if pred3 > 0.5 else '🟢 低风险（不发生晚疫病）'}")

# 测试用例 4：正常温度低湿度（不易发病）
print("\n[测试4] 正常条件（温和温度、低湿度 - 不易发病）")
temp_normal = [15.0] * 7   # 7天都是15°C
hum_normal = [40.0] * 7    # 7天都是40%湿度
hourly_normal = [14.0] * 24
input_raw_4 = temp_normal + hum_normal + hourly_normal

pred4, err4 = predict_time_series(input_raw_4)
if err4:
    print(f"  ❌ 预测失败: {err4}")
else:
    print(f"  ✓ 预测概率: {pred4:.6f}")
    print(f"  ✓ 风险判定: {'🔴 高风险（发生晚疫病）' if pred4 > 0.5 else '🟢 低风险（不发生晚疫病）'}")

# 测试用例 5：输入长度验证
print("\n[测试5] 验证输入长度检查")
input_wrong_length = [20.0] * 20  # 长度错误
pred5, err5 = predict_time_series(input_wrong_length)
if err5:
    print(f"  ✓ 正确捕获错误: {err5}")
else:
    print(f"  ❌ 未检测到错误!")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
