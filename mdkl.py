# filename: mdkl.py (版本 4，合併輸出並統一顯示最終誤差)
import logging
import math
import time
from typing import List, Callable, Tuple, Dict, Any

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

def multi_dim_func_high_rank(xs: List[float]) -> float:
    """一個四維的複雜函數，理論上需要較高的TT-rank才能精確表示。"""
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    x, y, z, w = xs[0], xs[1], xs[2], xs[3]
    term1 = np.cos(np.pi * (x + 2*y - z*w))
    term2 = (x*y - 0.5*z + 1.2*w**2) * np.exp(-0.5 * (x + z))
    term3 = np.tanh(2*x - y + z - 2*w)
    return float(term1 + 0.5*term2 - 0.8*term3)

def high_dim_low_rank_func(xs: List[float]) -> float:
    """一個八維的可分離函數 f(x1,...,x8) = product_i cos(pi/2 * xi)，其結構使其具有低秩特性。"""
    global EVAL_COUNTER
    EVAL_COUNTER += 1
    return np.prod(np.cos(np.array(xs) * (np.pi / 2.0)))

# --- 3. 積分演算法 (保持不變) ---

def gauss_legendre_quadrature(n: int, domain: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
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
    global EVAL_COUNTER
    EVAL_COUNTER = 0
    logger.info(f"--- 開始 TCI 積分 (內部收斂容忍度 reltol={tci_reltol:.1e}) ---")
    start_time = time.time()
    
    ci_param = TensorCI1Param(nIter=200, reltol=tci_reltol, pivot1=[0] * dim, fullPiv=False, nRookIter=8)
    points, weights = gauss_legendre_quadrature(n_quad_points, domain)
    
    def grid_func(indices: Tuple[int, ...]) -> float:
        coords = [points[i] for i in indices]
        return func(coords)

    fc_wrapped = TensorFunction(func=grid_func, use_cache=True)
    phys_dims = [n_quad_points] * dim
    
    ci_instance = TensorCI1(fc=fc_wrapped, phys_dims=phys_dims, param=ci_param, dtype=Type.Double, device=Device.cpu)
    
    integral_value = float('nan')
    if ci_instance.done:
        final_tt = ci_instance.get_canonical_tt(center=0)
        integration_weights = [weights.tolist()] * dim
        integral_value = final_tt.sum(weights=integration_weights)

    end_time = time.time()
    logger.info("--- TCI 積分結束 ---")
    return float(integral_value), (end_time - start_time), fc_wrapped.cache_info().currsize

def integrate_with_mc_to_target_error(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    target_error: float,
    reference_value: float,
    n_initial_samples: int = 1000,
    max_samples: int = 50_000_000
) -> Tuple[float, float, int]:
    global EVAL_COUNTER
    EVAL_COUNTER = 0
    logger.info("--- 開始蒙地卡羅積分 (目標誤差導向) ---")
    start_time = time.time()
    
    a, b = domain
    volume = (b - a)**dim
    
    n_samples = n_initial_samples
    integral_value = 0.0
    current_error = float('inf')

    while current_error > target_error:
        if n_samples > max_samples:
            logger.warning(f"蒙地卡羅取樣數已達上限 {max_samples:,}，但仍未達到目標誤差。")
            break
        EVAL_COUNTER = 0 
        current_total_sum = 0
        for _ in range(n_samples):
            random_point = np.random.uniform(a, b, dim).tolist()
            current_total_sum += func(random_point)
        integral_value = volume * (current_total_sum / n_samples)
        if reference_value != 0:
            current_error = abs(integral_value - reference_value)
        logger.debug(f"MC 迭代: n_samples={n_samples:<10,}, current_error={current_error:.4e}")
        if current_error > target_error:
            n_samples = int(n_samples * 1.5)
            
    end_time = time.time()
    logger.info("--- 蒙地卡羅積分結束 ---")
    return integral_value, (end_time - start_time), EVAL_COUNTER

# --- 4. 測試案例執行函式 ---

def run_test_case_1_high_rank() -> Dict[str, Any]:
    """執行四維高秩函數測試，並返回結果字典"""
    print("正在執行 [測試案例 1: 4維 高耦合/高秩函數]...")
    
    DIMENSIONS = 4
    DOMAIN = (0.0, 1.0)
    REFERENCE_SAMPLES = 20_000_000
    MC_TARGET_ERROR = 1e-3 
    TCI_RELTOL = 1e-9
    TCI_QUAD_POINTS = 10
    
    reference_value, _, _ = integrate_with_mc_to_target_error(
        func=multi_dim_func_high_rank, dim=DIMENSIONS, domain=DOMAIN, 
        target_error=1e-4, reference_value=0, n_initial_samples=REFERENCE_SAMPLES
    )
    
    tci_integral, tci_time, tci_evals = integrate_with_tci(
        func=multi_dim_func_high_rank, dim=DIMENSIONS, domain=DOMAIN,
        n_quad_points=TCI_QUAD_POINTS, tci_reltol=TCI_RELTOL
    )
    
    mc_integral, mc_time, mc_evals = integrate_with_mc_to_target_error(
        func=multi_dim_func_high_rank, dim=DIMENSIONS, domain=DOMAIN,
        target_error=MC_TARGET_ERROR, reference_value=reference_value
    )
    
    print("...測試案例 1 完成。")
    return {
        "title": "測試案例 1: 4維 高耦合/高秩函數",
        "reference_value": reference_value,
        "tci_results": (tci_integral, abs(tci_integral - reference_value), tci_time, tci_evals),
        "mc_results": (mc_integral, abs(mc_integral - reference_value), mc_time, mc_evals)
    }

def run_test_case_2_low_rank() -> Dict[str, Any]:
    """執行八維低秩函數測試，並返回結果字典"""
    print("正在執行 [測試案例 2: 8維 可分離/低秩函數]...")

    DIMENSIONS = 8
    DOMAIN = (0.0, 1.0)
    MC_TARGET_ERROR = 1e-3
    TCI_RELTOL = 1e-9
    TCI_QUAD_POINTS = 16 
    
    reference_value = (2.0 / np.pi) ** DIMENSIONS
    
    tci_integral, tci_time, tci_evals = integrate_with_tci(
        func=high_dim_low_rank_func, dim=DIMENSIONS, domain=DOMAIN,
        n_quad_points=TCI_QUAD_POINTS, tci_reltol=TCI_RELTOL
    )
    
    mc_integral, mc_time, mc_evals = integrate_with_mc_to_target_error(
        func=high_dim_low_rank_func, dim=DIMENSIONS, domain=DOMAIN,
        target_error=MC_TARGET_ERROR, reference_value=reference_value,
        n_initial_samples=50000
    )
    
    print("...測試案例 2 完成。")
    return {
        "title": "測試案例 2: 8維 可分離/低秩函數",
        "reference_value": reference_value,
        "tci_results": (tci_integral, abs(tci_integral - reference_value), tci_time, tci_evals),
        "mc_results": (mc_integral, abs(mc_integral - reference_value), mc_time, mc_evals)
    }

# --- 5. 主程式：執行所有測試並統一輸出結果 ---

if __name__ == "__main__":
    # 執行所有測試案例並收集結果
    results1 = run_test_case_1_high_rank()
    results2 = run_test_case_2_low_rank()

    # 統一格式化並輸出所有結果
    final_output_buffer = []
    
    for res_data in [results1, results2]:
        tci_res = res_data["tci_results"]
        mc_res = res_data["mc_results"]

        output_block = [
            "\n" + "="*90,
            f"      {res_data['title']}",
            "="*90,
            f"參考積分值: {res_data['reference_value']:.12f}",
            "-"*90,
            f"{'演算法':<15} | {'最終積分值':<20} | {'最終結果誤差':<20} | {'耗時 (秒)':<12} | {'函式求值次數':<15}",
            "-"*90,
            f"{'TCI':<15} | {tci_res[0]:<20.8f} | {tci_res[1]:<20.3e} | {tci_res[2]:<12.4f} | {tci_res[3]:<15,}",
            f"{'Monte Carlo':<15} | {mc_res[0]:<20.8f} | {mc_res[1]:<20.3e} | {mc_res[2]:<12.4f} | {mc_res[3]:<15,}",
            "="*90,
        ]
        final_output_buffer.extend(output_block)

    print("\n".join(final_output_buffer))