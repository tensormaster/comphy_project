# test_matmul.py
import cytnx

# 根據日誌創建 P_matrix_prev
P_prev_val = 0.19767681165408388
P_prev = cytnx.zeros((1,1), dtype=cytnx.Type.Double, device=cytnx.Device.cpu)
P_prev[0,0] = P_prev_val
print("P_prev:")
print(P_prev)

# 根據日誌創建 T_matrix_form
T_form_val1 = 0.19767681165408388
T_form_val2 = 0.9751703272075021
T_form = cytnx.zeros((1,2), dtype=cytnx.Type.Double, device=cytnx.Device.cpu)
T_form[0,0] = T_form_val1
T_form[0,1] = T_form_val2
print("\nT_form:")
print(T_form)

# 執行 Matmul
try:
    result = cytnx.linalg.Matmul(P_prev, T_form)
    print("\nResult of Matmul(P_prev, T_form):")
    print(result)
except Exception as e:
    print(f"\nError during Matmul: {e}")

# 預期結果 (手動計算)
expected_val1 = P_prev_val * T_form_val1
expected_val2 = P_prev_val * T_form_val2
print(f"\nManually expected result: [[{expected_val1}, {expected_val2}]]")