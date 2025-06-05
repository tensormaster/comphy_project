#filename:mdkl
import math
from typing import List, Callable, Tuple
import logging

# 導入您的專案檔案中的類別
import cytnx
from cytnx import Type, Device
from tensor_ci import TensorCI1, TensorCI1Param
from tensorfuc import TensorFunction
from tensor_train import TensorTrain


# --- 模擬 QuadratureGK15 (高斯-克朗羅德積分) ---
def QuadratureGK15(a: float, b: float) -> Tuple[List[float], List[float]]:
    """
    Gauss-Kronrod 15點正交規則的簡化佔位符。
    """
    num_points = 7
    xi = [a + (b - a) * (i + 0.5) / num_points for i in range(num_points)]
    wi = [(b - a) / num_points for i in range(num_points)]
    return xi, wi

# --- 主要執行邏輯 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    count = 0

    # 將 f 函式修改為返回實數
    def f(xs: List[float]) -> float:
        global count
        count += 1
        x_val = 0.0
        y_val = 0.0
        c_val = 0.0
        for xi_val in xs:
            c_val += 1
            x_val += c_val * xi_val
            y_val += xi_val * xi_val / c_val
        # 移除複數部分，只返回實部
        arg = 1.0 + (x_val + 2 * y_val + x_val * y_val) * math.pi
        return 1 + x_val + math.cos(arg)

    dim = 5

    # 獲取積分點和權重
    xi, wi = QuadratureGK15(0, 1)

    # 將函式 f 包裝到 TensorFunction 中
    fc_wrapped = TensorFunction(func=f)

    # 創建 TensorCI1 的參數
    phys_dims = [len(xi)] * dim
    ci_param = TensorCI1Param(weight=[wi] * dim)

    # 初始化 TensorCI1 實例
    ci = TensorCI1(fc=fc_wrapped, phys_dims=phys_dims, param=ci_param,
                    dtype=cytnx.Type.Double, device=cytnx.Device.cpu)

    # 打印表頭
    print(f"{'rank':<5} {'nEval':<10} {'LastSweepPivotError':<20} {'integral(f)':<20.12s}")
    print("-" * 55)

    # 執行迭代
    for i in range(1, 50):
        ci.iterate_one_full_sweep() # 將 .iterate() 修改為 .iterate_one_full_sweep()
        if i % 10 == 0:
            # 獲取 TensorTrain 實例並計算積分
            current_tt = ci.get_TensorTrain()
            integral_val = current_tt.sum(weights=[wi] * dim)

            # 打印結果，使用格式化字符串和 setprecision (這裡用 :.12f)
            print(f"{ci.current_rank:<5} {count:<10} {ci.pivotError[-1]:<20.12f} {integral_val:<20.12f}")

