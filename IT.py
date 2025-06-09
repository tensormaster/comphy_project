##--------------------------------------------test1-------------------------------------------------------------
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
    MC_SAMPLES_1 = 1_000
    MC_SAMPLES_2 = 10_000
    
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
#--------------------------------------------test2-------------------------------------------------------------
# 案例 1: 2 維、Rank 2 函數
def func_rank2_cos_minus(xs: List[float]) -> float:
    """
    一個 2D 函數 f(x, y) = cos(x - y)，可分解為 cos(x)cos(y) + sin(x)sin(y)，秩為 2。
    """
    if len(xs) != 2:
        raise ValueError("This function is defined for 2 dimensions only.")
    x, y = xs[0], xs[1]
    return np.cos(x - y)

# 案例 2: 2 維、Rank 2 函數
def func_rank2_sum(xs: List[float]) -> float:
    """
    一個 2D 函數 f(x, y) = x*y + sin(x)*sin(y)，由兩個 Rank-1 項相加而成，秩為 2。
    """
    if len(xs) != 2:
        raise ValueError("This function is defined for 2 dimensions only.")
    x, y = xs[0], xs[1]
    return x * y + np.sin(x) * np.sin(y)


# --- 3. TCI 和 MC 積分演算法 (與原版相同) ---
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
    logger.info(f"--- 開始 TCI 積分 (維度 D={dim}, reltol={tci_reltol:.1e}) ---")
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
    logger.info(f"--- TCI 積分結束 (維度 D={dim}) ---")
    return float(integral_value), (end_time - start_time), grid_fc_wrapped.cache_info().currsize

def integrate_with_mc(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_samples: int
) -> Tuple[float, float, int]:
    logger.info(f"--- 開始 Monte Carlo 積分 (維度 D={dim}, 樣本數 n_samples={n_samples:,}) ---")
    start_time = time.time()
    a, b = domain
    volume = (b - a) ** dim
    total_sum = 0.0
    random_points = a + (b - a) * np.random.rand(n_samples, dim)
    for i in range(n_samples):
        total_sum += func(random_points[i])
    integral_value = volume * (total_sum / n_samples)
    end_time = time.time()
    logger.info(f"--- Monte Carlo 積分結束 (維度 D={dim}) ---")
    return integral_value, (end_time - start_time), n_samples


# --- 4. 測試主體與比較 ---
if __name__ == "__main__":

    # --- 全局測試參數 ---
    TCI_RELTOL = 1e-9
    TCI_QUAD_POINTS = 24  # 稍微增加格點數以提高精度
    MC_SAMPLES = 5_000_000

    # --- 【修改】定義測試案例列表 ---
    test_cases = [
        {
            "name": "Rank-2: cos(x-y) (2D)",
            "func": func_rank2_cos_minus,
            "dim": 2,
            "domain": (0.0, np.pi/2),
            "ref_value": 2.0  # 解析解
        },
        {
            "name": "Rank-2: xy + sin(x)sin(y) (2D)",
            "func": func_rank2_sum,
            "dim": 2,
            "domain": (0.0, 1.0),
            "ref_value": 0.25 + (1 - np.cos(1))**2  # 解析解
        }
    ]

    # --- 【修改】先執行所有計算，將結果儲存起來 ---
    all_results = []
    for case in test_cases:
        logger.info(f"\n{'='*20} 正在執行測試案例: {case['name']} {'='*20}")
        
        ref_value = case["ref_value"]

        # 執行 TCI 積分
        tci_integral, tci_time, tci_evals = integrate_with_tci(
            func=case['func'], dim=case['dim'], domain=case['domain'],
            n_quad_points=TCI_QUAD_POINTS, tci_reltol=TCI_RELTOL
        )
        tci_error = abs(tci_integral - ref_value)

        # 執行蒙地卡羅積分
        mc_integral, mc_time, mc_evals = integrate_with_mc(
            func=case['func'], dim=case['dim'], domain=case['domain'],
            n_samples=MC_SAMPLES
        )
        mc_error = abs(mc_integral - ref_value)

        # 收集結果
        all_results.append({
            "case_name": case['name'],
            "ref_value": ref_value,
            "tci_results": {
                "integral": tci_integral, "error": tci_error,
                "time": tci_time, "evals": tci_evals
            },
            "mc_results": {
                "integral": mc_integral, "error": mc_error,
                "time": mc_time, "evals": mc_evals
            }
        })
    
    # --- 在所有計算完成後，統一輸出結果 ---
    print("\n\n" + "="*100)
    print("                TCI vs. Monte Carlo 積分表現總結")
    print("="*100)

    for result in all_results:
        print(f"\n--- 測試案例: {result['case_name']} ---")
        print(f"{'方法 (Method)':<20} | {'積分結果 (Value)':<24} | {'絕對誤差 (Error)':<15} | {'執行時間 (s)':<15} | {'求值次數':<15}")
        print("-"*100)
        
        # 參考值
        print(f"{'理論/解析參考值':<20} | {result['ref_value']:<24.16f} | {'-':<15} | {'-':<15} | {'-'}")
        
        # TCI 結果
        tci_res = result['tci_results']
        print(f"{'TCI':<20} | {tci_res['integral']:<24.16f} | {tci_res['error']:<15.2e} | {tci_res['time']:<15.4f} | {tci_res['evals']:<15,}")
        
        # MC 結果
        mc_res = result['mc_results']
        print(f"MC (N={MC_SAMPLES:,}) | {mc_res['integral']:<24.16f} | {mc_res['error']:<15.2e} | {mc_res['time']:<15.4f} | {mc_res['evals']:<15,}")
        print("-"*100)
##--------------------------------------------test3-------------------------------------------------------------


# --- 2. 測試函數 (與原版相同) ---
def high_dim_low_rank_func(xs: List[float]) -> float:
    """
    一個 N 維的可分離函數 f(x1,...,xN) = product_i cos(pi/2 * xi)，其結構使其具有低秩特性。
    """
    return np.prod(np.cos(np.array(xs) * (np.pi / 2.0)))

# --- 3. TCI 和 MC 積分演算法 (與原版相同) ---
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
    使用 TCI 進行積分的核心邏輯。
    """
    logger.info(f"--- 開始 TCI 積分 (積分格點數 n_quad_points={n_quad_points}) ---")
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
    logger.info(f"--- TCI 積分結束 (積分格點數 n_quad_points={n_quad_points}) ---")
    
    return float(integral_value), (end_time - start_time), grid_fc_wrapped.cache_info().currsize

def integrate_with_mc(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_samples: int
) -> Tuple[float, float, int]:
    """
    使用標準蒙地卡羅方法進行積分。
    """
    logger.info(f"--- 開始 Monte Carlo 積分 (樣本數 n_samples={n_samples:,}) ---")
    start_time = time.time()
    
    a, b = domain
    volume = (b - a) ** dim
    
    total_sum = 0.0
    random_points = a + (b - a) * np.random.rand(n_samples, dim)
    
    for i in range(n_samples):
        total_sum += func(random_points[i])
        
    integral_value = volume * (total_sum / n_samples)
    
    end_time = time.time()
    logger.info("--- Monte Carlo 積分結束 ---")
    
    return integral_value, (end_time - start_time), n_samples


# --- 4. 測試主體與比較 ---
if __name__ == "__main__":
    
    # --- 全局測試參數 ---
    DIMENSIONS = 8
    DOMAIN = (0.0, 1.0)
    TCI_RELTOL = 1e-10 # 固定一個較高的精度要求
    MC_SAMPLES = 5_000_000
    
    # --- 【修改】定義一系列要測試的 TCI_QUAD_POINTS 值 ---
    QUAD_POINTS_TO_TEST = [8, 12, 16, 24, 32]
    
    # --- 計算理論參考值 ---
    reference_value = (2.0 / np.pi) ** DIMENSIONS
    
    # --- 【修改】執行一次 MC 作為性能基準 ---
    logger.info(f"\n{'='*20} 正在執行 Monte Carlo 基準測試 (維度 D={DIMENSIONS}) {'='*20}")
    mc_integral, mc_time, mc_evals = integrate_with_mc(
        func=high_dim_low_rank_func,
        dim=DIMENSIONS,
        domain=DOMAIN,
        n_samples=MC_SAMPLES
    )
    mc_error = abs(mc_integral - reference_value)

    # --- 【修改】遍歷不同的 TCI_QUAD_POINTS 進行計算 ---
    tci_results = []
    logger.info(f"\n{'='*20} 開始遍歷不同的 TCI 積分格點數進行測試 (維度 D={DIMENSIONS}) {'='*20}")
    
    for n_points in QUAD_POINTS_TO_TEST:
        tci_integral, tci_time, tci_evals = integrate_with_tci(
            func=high_dim_low_rank_func, 
            dim=DIMENSIONS, 
            domain=DOMAIN,
            n_quad_points=n_points,  # 使用當前迴圈的格點數
            tci_reltol=TCI_RELTOL
        )
        tci_error = abs(tci_integral - reference_value)
        
        tci_results.append({
            "quad_points": n_points,
            "integral": tci_integral,
            "error": tci_error,
            "time": tci_time,
            "evals": tci_evals
        })

    # --- 【修改】統一輸出結果 ---
    print("\n\n" + "="*100)
    print(f"                TCI 積分格點數 (Quad Points) 影響分析 (維度 D={DIMENSIONS})")
    print("="*100)
    print(f"理論參考值: {reference_value:.14f}")
    print("-"*100)
    
    # 先印出 MC 基準
    print("【Monte Carlo 性能基準】")
    print(f"{'方法 (Method)':<20} | {'積分結果 (Value)':<24} | {'絕對誤差 (Error)':<15} | {'執行時間 (s)':<15} | {'求值次數':<15}")
    print(f"MC (N={MC_SAMPLES:,}) | {mc_integral:<24.14f} | {mc_error:<15.2e} | {mc_time:<15.4f} | {mc_evals:<15,}")
    
    print("\n【TCI 性能分析】")
    print(f"{'積分格點數':<20} | {'積分結果 (Value)':<24} | {'絕對誤差 (Error)':<15} | {'執行時間 (s)':<15} | {'求值次數':<15}")
    
    # 再印出 TCI 的縱向對比表格
    for res in tci_results:
        print(f"{res['quad_points']:<20} | {res['integral']:<24.14f} | {res['error']:<15.2e} | {res['time']:<15.4f} | {res['evals']:<15,}")
        
    print("="*100)
#--------------------------------------------test4-------------------------------------------------------------

# --- 2. 測試函數 ---
def high_dim_low_rank_func(xs: List[float]) -> float:
    """
    一個 N 維的可分離函數 f(x1,...,xN) = product_i cos(pi/2 * xi)。
    此函數結構簡單，具有 Rank-1 的特性。
    """
    return np.prod(np.cos(np.array(xs) * (np.pi / 2.0)))

# --- 3. TCI 和 MC 積分演算法 (與原版相同) ---
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
    logger.info(f"--- 開始 TCI 積分 (維度 D={dim}, reltol={tci_reltol:.1e}) ---")
    start_time = time.time()
    fc_wrapped = TensorFunction(func=func, use_cache=True)
    # 增加 nIter 以應對更高維度的計算需求
    ci_param = TensorCI1Param(
        nIter=500, reltol=tci_reltol, pivot1=[0] * dim, fullPiv=False, nRookIter=8
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
    logger.info(f"--- TCI 積分結束 (維度 D={dim}) ---")
    return float(integral_value), (end_time - start_time), grid_fc_wrapped.cache_info().currsize

def integrate_with_mc(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_samples: int
) -> Tuple[float, float, int]:
    logger.info(f"--- 開始 Monte Carlo 積分 (維度 D={dim}, 樣本數 n_samples={n_samples:,}) ---")
    start_time = time.time()
    a, b = domain
    volume = (b - a) ** dim
    total_sum = 0.0
    # 為避免記憶體問題，分批次產生隨機點
    batch_size = 1_000_000
    num_batches = int(np.ceil(n_samples / batch_size))
    for i in range(num_batches):
        current_batch_size = min(batch_size, n_samples - i * batch_size)
        if current_batch_size <= 0:
            break
        random_points = a + (b - a) * np.random.rand(current_batch_size, dim)
        for j in range(current_batch_size):
            total_sum += func(random_points[j])

    integral_value = volume * (total_sum / n_samples)
    end_time = time.time()
    logger.info(f"--- Monte Carlo 積分結束 (維度 D={dim}) ---")
    return integral_value, (end_time - start_time), n_samples


# --- 4. 測試主體與比較 ---
if __name__ == "__main__":

    # --- 全局測試參數 ---
    TCI_RELTOL = 1e-12 # 要求更高的精度
    TCI_QUAD_POINTS = 16
    MC_SAMPLES = 1_000_000 # 增加 MC 樣本數以應對高維挑戰
    DOMAIN = (0.0, 1.0)

    # --- 【修改】定義要測試的更高維度列表 ---
    DIMENSIONS_TO_TEST = [30, 45, 60, 75, 90]

    all_results = []

    # --- 遍歷不同維度進行測試 ---
    for dim in DIMENSIONS_TO_TEST:
        logger.info(f"\n{'='*25} 開始測試 維度 D = {dim} {'='*25}")

        # 理論參考值
        reference_value = (2.0 / np.pi) ** dim

        # 執行 TCI 積分
        tci_integral, tci_time, tci_evals = integrate_with_tci(
            func=high_dim_low_rank_func, dim=dim, domain=DOMAIN,
            n_quad_points=TCI_QUAD_POINTS, tci_reltol=TCI_RELTOL
        )
        tci_rel_error = abs(tci_integral / reference_value - 1.0) if abs(reference_value) > 1e-300 else float('inf')

        # 執行蒙地卡羅積分
        mc_integral, mc_time, mc_evals = integrate_with_mc(
            func=high_dim_low_rank_func, dim=dim, domain=DOMAIN,
            n_samples=MC_SAMPLES
        )
        mc_rel_error = abs(mc_integral / reference_value - 1.0) if abs(reference_value) > 1e-300 else float('inf')

        # 收集結果
        all_results.append({
            "dim": dim,
            "ref_value": reference_value,
            "tci_results": {"integral": tci_integral, "error": tci_rel_error, "time": tci_time, "evals": tci_evals},
            "mc_results": {"integral": mc_integral, "error": mc_rel_error, "time": mc_time, "evals": mc_evals}
        })

    # --- 在所有計算完成後，統一輸出結果 ---
    print("\n\n" + "="*105)
    print("                TCI vs. Monte Carlo 在極高維度下的表現比較 (維度災難展示)")
    print("="*105)
    # --- 【修改】調整表格欄位與格式 ---
    print(f"{'維度 (D)':<8} | {'方法':<15} | {'積分結果':<24} | {'相對誤差':<15} | {'執行時間 (s)':<15} | {'函數求值次數':<15}")
    print("-"*105)

    for result in all_results:
        print(f"  {result['dim']:<6} | {'參考值':<15} | {result['ref_value']:<24.6e} | {'-':<15} | {'-':<15} | {'-'}")

        tci_res = result['tci_results']
        print(f"  {'':<6} | {'TCI':<15} | {tci_res['integral']:<24.6e} | {tci_res['error']:<15.2e} | {tci_res['time']:<15.4f} | {tci_res['evals']:<15,}")

        mc_res = result['mc_results']
        print(f"  {'':<6} | {'MC':<15} | {mc_res['integral']:<24.6e} | {mc_res['error']:<15.2e} | {mc_res['time']:<15.4f} | {mc_res['evals']:<15,}")

        if result != all_results[-1]:
            print("-" * 105)

    print("="*105)
    ##--------------------------------------------test5-------------------------------------------------------------


# --- 2. 新增的、秩可調整的測試函數 ---

def adjustable_rank_func(xs: List[float], rank: int) -> float:
    """
    一個2D函數，其秩 (Rank) 可以透過參數 'rank' 調整。
    f(x, y, R) = Σ_{k=1 to R} (1/k^2) * sin((k-0.5)πx) * sin((k-0.5)πy)
    """
    if len(xs) != 2:
        raise ValueError("This function is defined for 2 dimensions only.")
    x, y = xs[0], xs[1]
    
    total = 0.0
    for k in range(1, rank + 1):
        term_coeff = 1.0 / (k * k)
        angle = (k - 0.5) * np.pi
        total += term_coeff * np.sin(angle * x) * np.sin(angle * y)
        
    return total

def create_test_func(rank: int) -> Callable[[List[float]], float]:
    """工廠函數：創建一個只接受 xs 列表作為參數的特定秩的函數。"""
    def func(xs: List[float]) -> float:
        return adjustable_rank_func(xs, rank)
    return func

def calculate_reference_integral(rank: int) -> float:
    """計算該函數在 [0,1]x[0,1] 區間的定積分解析解。"""
    total_integral = 0.0
    for k in range(1, rank + 1):
        term_coeff = 1.0 / (k * k)
        integral_part = 1.0 / ((k - 0.5) * np.pi)
        total_integral += term_coeff * (integral_part ** 2)
    return total_integral

# --- 3. TCI 和 MC 積分演算法 (與原版相同) ---
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
    logger.info(f"--- 開始 TCI 積分 (目標Rank={func.__name__}, reltol={tci_reltol:.1e}) ---")
    start_time = time.time()
    fc_wrapped = TensorFunction(func=func, use_cache=True)
    ci_param = TensorCI1Param(
        nIter=500, reltol=tci_reltol, pivot1=[0] * dim, fullPiv=False, nRookIter=10
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
    logger.info(f"--- TCI 積分結束 ---")
    return float(integral_value), (end_time - start_time), grid_fc_wrapped.cache_info().currsize

def integrate_with_mc(
    func: Callable[[List[float]], float],
    dim: int,
    domain: Tuple[float, float],
    n_samples: int
) -> Tuple[float, float, int]:
    logger.info(f"--- 開始 Monte Carlo 積分 (目標Rank={func.__name__}, 樣本數 n_samples={n_samples:,}) ---")
    start_time = time.time()
    a, b = domain
    volume = (b - a) ** dim
    total_sum = 0.0
    random_points = a + (b - a) * np.random.rand(n_samples, dim)
    for i in range(n_samples):
        total_sum += func(random_points[i])
    integral_value = volume * (total_sum / n_samples)
    end_time = time.time()
    logger.info(f"--- Monte Carlo 積分結束 ---")
    return integral_value, (end_time - start_time), n_samples

# --- 4. 測試主體與比較 ---
if __name__ == "__main__":

    # --- 全局測試參數 ---
    TCI_RELTOL = 1e-9
    TCI_QUAD_POINTS = 32  # 使用較多格點以確保離散化精度
    MC_SAMPLES = 5_000_000
    DIMENSIONS = 2
    DOMAIN = (0.0, 1.0)
    
    # --- 【修改】定義要測試的函數秩列表 ---
    RANKS_TO_TEST = [10, 20, 40, 80, 160]
    
    all_results = []

    # --- 遍歷不同的秩進行測試 ---
    for rank in RANKS_TO_TEST:
        logger.info(f"\n{'='*25} 開始測試 Rank = {rank} {'='*25}")
        
        # 創建特定秩的函數及其積分參考值
        test_function = create_test_func(rank)
        test_function.__name__ = f'Rank-{rank}' # 方便日誌追蹤
        reference_value = calculate_reference_integral(rank)
        
        # 執行 TCI 積分
        tci_integral, tci_time, tci_evals = integrate_with_tci(
            func=test_function, dim=DIMENSIONS, domain=DOMAIN,
            n_quad_points=TCI_QUAD_POINTS, tci_reltol=TCI_RELTOL
        )
        tci_error = abs(tci_integral - reference_value)

        # 執行蒙地卡羅積分
        mc_integral, mc_time, mc_evals = integrate_with_mc(
            func=test_function, dim=DIMENSIONS, domain=DOMAIN,
            n_samples=MC_SAMPLES
        )
        mc_error = abs(mc_integral - reference_value)
        
        # 收集結果
        all_results.append({
            "rank": rank, "ref_value": reference_value,
            "tci_results": {"integral": tci_integral, "error": tci_error, "time": tci_time, "evals": tci_evals},
            "mc_results": {"integral": mc_integral, "error": mc_error, "time": mc_time, "evals": mc_evals}
        })

    # --- 在所有計算完成後，統一輸出結果 ---
    print("\n\n" + "="*105)
    print("                TCI vs. Monte Carlo 對於不同秩 (Rank) 函數的表現")
    print("="*105)
    print(f"{'目標秩 (R)':<10} | {'方法':<15} | {'積分結果':<24} | {'絕對誤差':<15} | {'執行時間 (s)':<15} | {'函數求值次數':<15}")
    print("-"*105)

    for result in all_results:
        print(f"  {result['rank']:<8} | {'參考值':<15} | {result['ref_value']:<24.12f} | {'-':<15} | {'-':<15} | {'-'}")
        
        tci_res = result['tci_results']
        print(f"  {'':<8} | {'TCI':<15} | {tci_res['integral']:<24.12f} | {tci_res['error']:<15.2e} | {tci_res['time']:<15.4f} | {tci_res['evals']:<15,}")
        
        mc_res = result['mc_results']
        print(f"  {'':<8} | {'MC':<15} | {mc_res['integral']:<24.12f} | {mc_res['error']:<15.2e} | {mc_res['time']:<15.4f} | {mc_res['evals']:<15,}")
        
        if result != all_results[-1]:
            print("-" * 105)

    print("="*105)

