# filename: IT.py
import logging
import numpy as np
import cytnx
import time
from typing import List, Callable, Tuple

# 確保您的專案模組可以被導入
from tensor_ci import TensorCI1, TensorCI1Param
from tensorfuc import TensorFunction
from tensor_train import TensorTrain

# --- 1. 設定日誌紀錄 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. 來自 mdkl.py 的低秩測試函數 ---
def high_dim_low_rank_func(xs: List[float]) -> float:
    """
    一個八維的可分離函數 f(x1,...,x8) = product_i cos(pi/2 * xi)，其結構使其具有低秩特性。
    這個函數是從 mdkl.py 的 case 2 複製過來的。
    """
    # 為了方便追蹤，我們在此處自行計數，不使用全域變數
    # 注意：TensorFunction 的快取機制會影響實際呼叫次數
    return np.prod(np.cos(np.array(xs) * (np.pi / 2.0)))

# --- 3. 來自 mdkl.py 的 TCI 積分演算法 ---
def gauss_legendre_quadrature(n: int, domain: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    """生成高斯-勒讓德正交點和權重。"""
    points, weights = np.polynomial.legendre.leggauss(n)
    a, b = domain
    scaled_points = 0.5 * (b - a) * points + 0.5 * (b + a)
    scaled_weights = 0.5 * (b - a) * weights
    return scaled_points, scaled_weights

def integrate_with_tci(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_quad_points: int,
    tci_reltol: float
) -> Tuple[float, float, int]:
    """
    使用 TCI 進行積分的核心邏輯，從 mdkl.py 中提取。
    """
    logger.info(f"--- 開始 TCI 積分 (內部收斂容忍度 reltol={tci_reltol:.1e}) ---")
    start_time = time.time()
    
    # 將 Python 函數包裝成 TensorFunction，啟用快取
    fc_wrapped = TensorFunction(func=func, use_cache=True)
    
    # 設定 TCI 參數
    ci_param = TensorCI1Param(
        nIter=200, 
        reltol=tci_reltol, 
        pivot1=[0] * dim, 
        fullPiv=False, 
        nRookIter=8
    )
    
    # 獲取積分節點和權重
    points, weights = gauss_legendre_quadrature(n_quad_points, domain)
    
    # 定義 TCI 需要的格點函數
    def grid_func(indices: Tuple[int, ...]) -> float:
        coords = [points[i] for i in indices]
        return fc_wrapped(coords)

    # 再次包裝 grid_func 以便追蹤和快取
    grid_fc_wrapped = TensorFunction(func=grid_func, use_cache=True)
    
    phys_dims = [n_quad_points] * dim
    
    # 初始化並執行 TCI
    ci_instance = TensorCI1(
        fc=grid_fc_wrapped, 
        phys_dims=phys_dims, 
        param=ci_param, 
        dtype=cytnx.Type.Double, 
        device=cytnx.Device.cpu
    )
    
    integral_value = float('nan')
    if ci_instance.done:
        # 獲取正規化的 Tensor Train
        final_tt = ci_instance.get_canonical_tt(center=0)
        
        # 準備權重列表以進行求和
        integration_weights = [weights.tolist()] * dim
        
        # 呼叫 TensorTrain.sum() 進行積分
        integral_value = final_tt.sum(weights=integration_weights)

    end_time = time.time()
    logger.info("--- TCI 積分結束 ---")
    
    # 返回：積分值、執行時間、函數求值次數
    return float(integral_value), (end_time - start_time), grid_fc_wrapped.cache_info().currsize

# --- 4. 測試主體 ---
if __name__ == "__main__":
    
    # --- 測試參數 ---
    DIMENSIONS = 8
    DOMAIN = (0.0, 1.0)
    TCI_RELTOL = 1e-9
    TCI_QUAD_POINTS = 16
    
    # --- 理論參考值 ---
    # Integral of cos(pi/2 * x) from 0 to 1 is [2/pi * sin(pi/2*x)]_0^1 = 2/pi
    # For 8 dimensions, the result is (2/pi)^8
    reference_value = (2.0 / np.pi) ** DIMENSIONS
    
    logger.info("正在執行 [8維 可分離/低秩函數] 的 TCI 積分測試...")
    
    # --- 執行 TCI 積分 ---
    tci_integral, tci_time, tci_evals = integrate_with_tci(
        func=high_dim_low_rank_func, 
        dim=DIMENSIONS, 
        domain=DOMAIN,
        n_quad_points=TCI_QUAD_POINTS, 
        tci_reltol=TCI_RELTOL
    )
    
    # --- 計算並輸出誤差 ---
    final_error = abs(tci_integral - reference_value)
    
    print("\n" + "="*70)
    print("                TCI 積分結果分析 (來自 IT.py)")
    print("="*70)
    print(f"{'項目':<25} | {'結果'}")
    print("-"*70)
    print(f"{'函數':<25} | {high_dim_low_rank_func.__name__}")
    print(f"{'維度':<25} | {DIMENSIONS}")
    print(f"{'理論積分值':<25} | {reference_value:.12f}")
    print(f"{'TCI 計算積分值':<25} | {tci_integral:.12f}")
    print(f"{'最終絕對誤差':<25} | {final_error:.6e}")
    print(f"{'總耗時 (秒)':<25} | {tci_time:.4f}")
    print(f"{'函數求值次數 (快取後)':<25} | {tci_evals:,}")
    print("="*70)

    # 斷言：如果誤差仍然很大，這裡會失敗，這符合我們的預期
    assert final_error < 1e-6, f"誤差過大 ({final_error:.6e})！TCI 積分邏輯存在問題。"