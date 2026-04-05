"""
时序模型预测简单测试 - 避免编码问题
"""
import sys
sys.path.insert(0, r'c:/Users/18083/Desktop/potato_backend-213/potato_backend-213')

from backend.time_series_service import predict_time_series
import numpy as np

print("="*70)
print("Time Series Model Prediction Test")
print("="*70)

# Test 1: Standard feature input for new model
print("\n[TEST 1] Standard feature input")
input_raw_1 = [22.3, 25.0, 18.1, 10.0, 4.5, 48.0]

pred1, err1 = predict_time_series(input_raw_1)
if err1:
    print(f"  ERROR: {err1}")
else:
    print(f"  SUCCESS: probability={pred1:.4f}")
    print(f"  Risk: {'HIGH (will develop)' if pred1 > 0.5 else 'LOW (safe)'}")

# Test 2: Random feature data
print("\n[TEST 2] Random feature data")
input_raw_2 = [np.random.uniform(10, 30), np.random.uniform(25, 35), np.random.uniform(5, 18), np.random.uniform(0, 20), np.random.uniform(0, 30), np.random.uniform(0, 24)]

pred2, err2 = predict_time_series(input_raw_2)
if err2:
    print(f"  ERROR: {err2}")
else:
    print(f"  SUCCESS: probability={pred2:.4f}")
    print(f"  Risk: {'HIGH (will develop)' if pred2 > 0.5 else 'LOW (safe)'}")

# Test 3: Hot and wet (high risk)
print("\n[TEST 3] Hot and wet (high risk condition)")
input_raw_3 = [28.0, 32.0, 20.0, 40.0, 60.0, 36.0]

pred3, err3 = predict_time_series(input_raw_3)
if err3:
    print(f"  ERROR: {err3}")
else:
    print(f"  SUCCESS: probability={pred3:.4f}")
    print(f"  Risk: {'HIGH (will develop)' if pred3 > 0.5 else 'LOW (safe)'}")

# Test 4: Cool and dry (low risk)
print("\n[TEST 4] Cool and dry (low risk condition)")
input_raw_4 = [16.0, 19.0, 12.0, 1.0, 2.0, 2.0]

pred4, err4 = predict_time_series(input_raw_4)
if err4:
    print(f"  ERROR: {err4}")
else:
    print(f"  SUCCESS: probability={pred4:.4f}")
    print(f"  Risk: {'HIGH (will develop)' if pred4 > 0.5 else 'LOW (safe)'}")

# Test 5: Wrong length
print("\n[TEST 5] Validation - wrong input length")
input_wrong = [20.0] * 20

pred5, err5 = predict_time_series(input_wrong)
if err5:
    print(f"  CORRECT ERROR CAUGHT: {err5}")
else:
    print(f"  ERROR: Should have caught length error!")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
