# filename: IT_with_MC.py
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
    return np.prod(np.cos(np.array(xs) * (np.pi / 2.0)))

# --- 3. 來自 IT.py 的 TCI 積分演算法 (保持不變) ---
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
    
    fc_wrapped = TensorFunction(func=func, use_cache=True)
    ci_param = TensorCI1Param(
        nIter=200, reltol=tci_reltol, pivot1=[0] * dim, fullPiv=False, nRookIter=8
    )
    points, weights = gauss_legendre_quadrature(n_quad_points, domain)
    
    def grid_func(indices: Tuple[int, ...]) -> float:
        coords = [points[i] for i in indices]
        return fc_wrapped(coords)

    grid_fc_wrapped = TensorFunction(func=grid_func, use_cache=True)
    phys_dims = [n_quad_points] * dim
    
    ci_instance = TensorCI1(
        fc=grid_fc_wrapped, phys_dims=phys_dims, param=ci_param, 
        dtype=cytnx.Type.Double, device=cytnx.Device.cpu
    )
    
    integral_value = float('nan')
    if ci_instance.done:
        final_tt = ci_instance.get_canonical_tt(center=0)
        integration_weights = [weights.tolist()] * dim
        integral_value = final_tt.sum(weights=integration_weights)

    end_time = time.time()
    logger.info("--- TCI 積分結束 ---")
    
    return float(integral_value), (end_time - start_time), grid_fc_wrapped.cache_info().currsize

# --- 4. 新增的蒙地卡羅積分演算法 ---
def integrate_with_mc(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_samples: int
) -> Tuple[float, float, int]:
    """
    使用標準蒙地卡羅方法進行積分。

    Args:
        func: 要積分的函數。
        dim: 積分維度。
        domain: 每個維度的積分區間 [a, b]。
        n_samples: 隨機抽樣點的數量。

    Returns:
        積分近似值、執行時間、函數求值次數。
    """
    logger.info(f"--- 開始 Monte Carlo 積分 (樣本數 n_samples={n_samples:,}) ---")
    start_time = time.time()
    
    a, b = domain
    volume = (b - a) ** dim
    
    total_sum = 0.0
    # 產生 n_samples 個在 [0, 1) 區間的隨機點，然後縮放到 [a, b)
    # np.random.rand(n_samples, dim) 產生 shape=(n_samples, dim) 的陣列
    random_points = a + (b - a) * np.random.rand(n_samples, dim)
    
    for i in range(n_samples):
        total_sum += func(random_points[i])
        
    integral_value = volume * (total_sum / n_samples)
    
    end_time = time.time()
    logger.info("--- Monte Carlo 積分結束 ---")
    
    # 對於標準蒙地卡羅，函數求值次數等於樣本數
    return integral_value, (end_time - start_time), n_samples


# --- 5. 測試主體與比較 ---
if __name__ == "__main__":
    
    # --- 測試參數 ---
    DIMENSIONS = 8
    DOMAIN = (0.0, 1.0)
    TCI_RELTOL = 1e-9
    TCI_QUAD_POINTS = 16
    MC_SAMPLES_1 = 1_000_000
    MC_SAMPLES_2 = 10_000_000
    
    # --- 理論參考值 ---
    # Integral of cos(pi/2 * x) from 0 to 1 is [2/pi * sin(pi/2*x)]_0^1 = 2/pi
    # For 8 dimensions, the result is (2/pi)^8
    reference_value = (2.0 / np.pi) ** DIMENSIONS
    
    # --- 執行 TCI 積分 ---
    logger.info("執行 [8維 可分離/低秩函數] 的 TCI 積分測試...")
    tci_integral, tci_time, tci_evals = integrate_with_tci(
        func=high_dim_low_rank_func, 
        dim=DIMENSIONS, 
        domain=DOMAIN,
        n_quad_points=TCI_QUAD_POINTS, 
        tci_reltol=TCI_RELTOL
    )
    tci_error = abs(tci_integral - reference_value)

    # --- 執行蒙地卡羅積分 ---
    logger.info(f"執行 Monte Carlo 積分測試 (N={MC_SAMPLES_1:,})...")
    mc1_integral, mc1_time, mc1_evals = integrate_with_mc(
        func=high_dim_low_rank_func,
        dim=DIMENSIONS,
        domain=DOMAIN,
        n_samples=MC_SAMPLES_1
    )
    mc1_error = abs(mc1_integral - reference_value)

    logger.info(f"執行 Monte Carlo 積分測試 (N={MC_SAMPLES_2:,})...")
    mc2_integral, mc2_time, mc2_evals = integrate_with_mc(
        func=high_dim_low_rank_func,
        dim=DIMENSIONS,
        domain=DOMAIN,
        n_samples=MC_SAMPLES_2
    )
    mc2_error = abs(mc2_integral - reference_value)

    # --- 格式化輸出比較結果 ---
    print("\n" + "="*80)
    print("                TCI vs. Monte Carlo 積分結果比較")
    print("="*80)
    print(f"{'方法 (Method)':<20} | {'積分結果 (Value)':<20} | {'絕對誤差 (Error)':<15} | {'執行時間 (s)':<12} | {'求值次數':<15}")
    print("-"*80)
    print(f"{'理論值 (Exact)':<20} | {reference_value:<20.12f} | {'-':<15} | {'-':<12} | {'-'}")
    print(f"{'TCI':<25} | {tci_integral:<20.12f} | {tci_error:<15.2e} | {tci_time:<12.4f} | {tci_evals:<15,}")
    print(f"{f'Monte Carlo (N={MC_SAMPLES_1:,})':<25} | {mc1_integral:<20.12f} | {mc1_error:<15.2e} | {mc1_time:<12.4f} | {mc1_evals:<15,}")
    print(f"{f'Monte Carlo (N={MC_SAMPLES_2:,})':<25} | {mc2_integral:<20.12f} | {mc2_error:<15.2e} | {mc2_time:<12.4f} | {mc2_evals:<15,}")
    print("="*80)