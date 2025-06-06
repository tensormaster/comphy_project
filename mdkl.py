# filename: mdkl.py (以目標誤差為基準進行比較)
import logging
import math
import time
from typing import List, Callable, Tuple

import cytnx
import numpy as np
from cytnx import Type, Device

# 導入您的專案檔案中的類別
from tensor_ci import TensorCI1, TensorCI1Param
from tensor_train import TensorTrain
from tensorfuc import TensorFunction

# --- 1. 全域設定與日誌配置 ---
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)
EVAL_COUNTER = 0

# --- 2. 要進行積分的多維度函式 ---
def multi_dim_func(xs: List[float]) -> float:
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    x, y, z, w = xs[0], xs[1], xs[2], xs[3]
    term1 = np.cos(np.pi * (x + 2*y - z*w))
    term2 = (x*y - 0.5*z + 1.2*w**2) * np.exp(-0.5 * (x + z))
    term3 = np.tanh(2*x - y + z - 2*w)
    return float(term1 + 0.5*term2 - 0.8*term3)

# --- 3. 積分演算法 ---

def gauss_legendre_quadrature(n: int, domain: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(n)
    a, b = domain
    scaled_points = 0.5 * (b - a) * points + 0.5 * (b + a)
    scaled_weights = 0.5 * (b - a) * weights
    return scaled_points, scaled_weights

def integrate_with_tci_to_target_error(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_quad_points: int,
    target_error: float
) -> Tuple[float, float, int]:
    """
    使用 TCI 進行積分，直到其內部誤差評估低於 target_error。
    """
    logger.info("--- 開始 TCI 積分 (目標誤差導向) ---")
    start_time = time.time()
    
    # TCI 演算法的 reltol 直接設為我們的目標誤差
    ci_param = TensorCI1Param(
        nIter=200,  # 設定一個較高的迭代上限
        reltol=target_error, # 關鍵參數！
        pivot1=[0] * dim,
        fullPiv=False,
        nRookIter=8
    )
    
    points, weights = gauss_legendre_quadrature(n_quad_points, domain)
    
    def grid_func(indices: Tuple[int, ...]) -> float:
        coords = [points[i] for i in indices]
        return func(coords)

    fc_wrapped = TensorFunction(func=grid_func, use_cache=True)
    phys_dims = [n_quad_points] * dim
    
    ci_instance = TensorCI1(
        fc=fc_wrapped, phys_dims=phys_dims, param=ci_param,
        dtype=cytnx.Type.Double, device=cytnx.Device.cpu
    )
    
    final_tt = ci_instance.get_canonical_tt(center=0)
    integration_weights = [weights.tolist()] * dim
    integral_value = final_tt.sum(weights=integration_weights)
    
    end_time = time.time()
    logger.info("--- TCI 積分結束 ---")
    
    return float(integral_value), (end_time - start_time), EVAL_COUNTER

def integrate_with_mc_to_target_error(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    target_error: float,
    reference_value: float,
    n_initial_samples: int = 1000,
    max_samples: int = 50_000_000
) -> Tuple[float, float, int]:
    """
    使用蒙地卡羅法進行積分，以迭代方式增加樣本數，直到誤差低於目標值。
    """
    logger.info("--- 開始蒙地卡羅積分 (目標誤差導向) ---")
    start_time = time.time()
    
    a, b = domain
    volume = (b - a)**dim
    
    n_samples = n_initial_samples
    total_sum = 0.0
    current_error = float('inf')

    while current_error > target_error:
        if n_samples > max_samples:
            logger.warning(f"蒙地卡羅取樣數已達上限 {max_samples:,}，但仍未達到目標誤差。")
            break
            
        # 為了避免重算，我們只計算新增的樣本
        # 這裡為了簡化，我們每次都重新計算，但會重置計數器以得到正確的總求值次數
        global EVAL_COUNTER
        EVAL_COUNTER = 0 # 重置計數器以匹配 n_samples
        
        # 執行一次完整的蒙地卡羅計算
        current_total_sum = 0
        for _ in range(n_samples):
            random_point = np.random.uniform(a, b, dim).tolist()
            current_total_sum += func(random_point)
            
        integral_value = volume * (current_total_sum / n_samples)
        current_error = abs(integral_value - reference_value)
        
        logger.debug(f"MC 迭代: n_samples={n_samples:<10,}, current_error={current_error:.4e}")

        # 如果未達標，增加樣本數以進行下一次迭代
        if current_error > target_error:
            n_samples = int(n_samples * 1.5) # 每次增加 50% 的樣本數

    end_time = time.time()
    logger.info("--- 蒙地卡羅積分結束 ---")

    return integral_value, (end_time - start_time), n_samples


# --- 4. 主程式執行與比較 ---
if __name__ == "__main__":
    
    output_buffer = []

    # --- 可調整的參數 ---
    DIMENSIONS = 4
    DOMAIN = (0.0, 1.0)
    REFERENCE_SAMPLES = 20_000_000
    
    # 這是本次比較的基準
    TARGET_ERROR = 5e-3 

    # TCI 專用參數
    TCI_QUAD_POINTS = 10
    
    # --- 步驟 1: 計算高精度參考值 ---
    output_buffer.append("正在使用大量取樣的蒙地卡羅法計算參考積分值...")
    EVAL_COUNTER = 0
    reference_value, _, _ = integrate_with_mc_to_target_error(
        func=multi_dim_func, dim=DIMENSIONS, domain=DOMAIN, 
        target_error=1e-4, # 為參考值設定一個極高的精度
        reference_value=0, # 第一次運行時，參考值設為0，讓它自己跑
        n_initial_samples=REFERENCE_SAMPLES
    )
    output_buffer.append(f"參考積分值 (高精度 MC): {reference_value:.12f}")
    output_buffer.append(f"目標誤差設為: {TARGET_ERROR:.1e}")
    output_buffer.append("-" * 40)

    # --- 步驟 2: 執行 TCI 積分 ---
    output_buffer.append(f"正在執行 TCI 積分，直到誤差低於 {TARGET_ERROR:.1e}...")
    EVAL_COUNTER = 0
    tci_integral, tci_time, tci_evals = integrate_with_tci_to_target_error(
        func=multi_dim_func, dim=DIMENSIONS, domain=DOMAIN,
        n_quad_points=TCI_QUAD_POINTS, target_error=TARGET_ERROR
    )
    output_buffer.append("TCI 積分完成。")

    # --- 步驟 3: 執行蒙地卡羅積分 ---
    output_buffer.append(f"正在執行蒙地卡羅積分，直到誤差低於 {TARGET_ERROR:.1e}...")
    EVAL_COUNTER = 0
    mc_integral, mc_time, mc_evals = integrate_with_mc_to_target_error(
        func=multi_dim_func, dim=DIMENSIONS, domain=DOMAIN,
        target_error=TARGET_ERROR, reference_value=reference_value
    )
    output_buffer.append("蒙地卡羅積分完成。")

    # --- 步驟 4: 準備最終的比較表格 ---
    tci_final_error = abs(tci_integral - reference_value)
    mc_final_error = abs(mc_integral - reference_value)
    
    table = [
        "\n\n" + "="*80,
        f"          比較報告：何種演算法能更快達到 {TARGET_ERROR:.1e} 的目標誤差?",
        "="*80,
        f"{'演算法':<15} | {'最終積分值':<20} | {'最終絕對誤差':<10} | {'耗時 (秒)':<12} | {'函式求值次數':<15}",
        "-"*80,
        f"{'TCI':<20} | {tci_integral:<25.12f} | {tci_final_error:<15.3e} | {tci_time:<12.4f} | {tci_evals:<15,}",
        f"{'Monte Carlo':<20} | {mc_integral:<25.12f} | {mc_final_error:<15.3e} | {mc_time:<12.4f} | {mc_evals:<15,}",
        "="*80,
    ]
    output_buffer.extend(table)
    
    # --- 步驟 5: 將所有收集到的訊息一次性印出 ---
    print("\n".join(output_buffer))