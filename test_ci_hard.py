# filename: test_ci_hard.py
import logging
import numpy as np
import cytnx
from itertools import product
import time

# 確保您的專案模組可以被導入
from tensor_ci import TensorCI1, TensorCI1Param
from tensorfuc import TensorFunction
from tensor_train import TensorTrain

# --- 1. 設定日誌紀錄 ---
# 設定日誌格式，使其輸出更詳細的資訊，便於追蹤
logging.basicConfig(
    level=logging.INFO,  # 您可以改為 logging.DEBUG 來獲取最詳細的追蹤訊息
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. 定義一個難以逼近的函數 ---
# 這個函數包含多個耦合的、非線性的項，理論上需要較高的TT-rank才能精確表示
def hard_to_approximate_func(indices: tuple[int, ...]) -> float:
    """
    一個四維的複雜函數，用於測試 TCI 的極限。
    f(i,j,k,l) 包含了三角函數、指數與多項式的耦合。
    """
    # 對輸入索引做正規化，使其落在 [0, 1] 區間，避免數值過大
    i = indices[0] / 3.0
    j = indices[1] / 4.0
    k = indices[2] / 3.0
    l = indices[3] / 4.0

    # 組合 1: 三角函數耦合
    term1 = np.cos(np.pi * (i + 2*j - k*l))

    # 組合 2: 指數與多項式耦合
    term2 = (i*j - 0.5*k + 1.2*l**2) * np.exp(-0.5 * (i + k))

    # 組合 3: 另一種非線性耦合
    term3 = np.tanh(2*i - j + k - 2*l)

    return float(term1 + 0.5*term2 - 0.8*term3)

# --- 3. 測試主體 ---
if __name__ == "__main__":
    logger.info("--- 開始高難度 TensorCI 分解測試 ---")

    # --- 測試參數設定 ---
    phys_dims = [3, 4, 3, 4]  # 4個維度，且每個維度的物理大小不同
    D = len(phys_dims)
    target_dtype = cytnx.Type.Double
    target_device = cytnx.Device.cpu
    
    # 將Python函數包裝成 TensorFunction
    fc_wrapped = TensorFunction(func=hard_to_approximate_func, use_cache=True)

    # 設定 TensorCI1 參數
    # 我們給予較高的迭代次數上限，並設定一個實際的誤差目標
    ci_params = TensorCI1Param(
        nIter=50,             # 增加最大迭代次數，給予算法充分的時間去尋找主元
        reltol=1e-7,          # 設定一個合理的相對誤差容忍度
        pivot1=[0] * D,       # 從 (0,0,0,0) 開始尋找第一個主元
        fullPiv=False,        # 使用計算效率較高的 Rook Pivoting
        nRookIter=10          # 增加Rook Pivoting的迭代次數以找到更好的主元
    )

    # --- 執行 TCI 分解 ---
    logger.info(f"初始化 TensorCI1，維度 D={D}, 物理維度 phys_dims={phys_dims}")
    logger.info(f"最大迭代次數: {ci_params.nIter}, 目標誤差: {ci_params.reltol:.2e}")
    
    start_time = time.time()
    try:
        ci_instance = TensorCI1(
            fc=fc_wrapped,
            phys_dims=phys_dims,
            param=ci_params,
            dtype=target_dtype,
            device=target_device
        )
    except Exception as e:
        logger.error(f"TensorCI1 初始化失敗: {e}", exc_info=True)
        raise

    end_time = time.time()
    logger.info(f"TCI 分解完成，總耗時: {end_time - start_time:.4f} 秒")

    # --- 結果分析 ---
    logger.info("--- 分析分解結果 ---")
    
    # 1. 檢查收斂情況與秩的增長
    logger.info(f"算法是否已收斂 (done): {ci_instance.done}")
    logger.info(f"主元誤差衰減歷史 (errorDecay): {ci_instance.errorDecay}")
    for i in range(D - 1):
        final_rank = ci_instance.P_cross_data[i].rank()
        logger.info(f"鍵結 (bond) {i} 的最終TT-rank: {final_rank}")
        # 對於複雜函數，我們預期秩會大於1
        assert final_rank > 1, f"Bond {i} 的秩應大於1，當前為 {final_rank}"

    # 2. 計算並驗證全局重建誤差
    logger.info("計算全局最大重建誤差 (trueError)...")
    
    # 獲取最終的 Tensor Train 表示 (使用正規化形式)
    final_tt = ci_instance.get_canonical_tt(center=0)
    
    max_abs_error = 0.0
    total_elements = np.prod(phys_dims)
    logger.info(f"將遍歷所有 {total_elements} 個張量元素進行比較。")

    # 遍歷所有可能的索引組合
    all_indices = product(*[range(d) for d in phys_dims])

    for multi_index in all_indices:
        # 計算原始函數的精確值
        val_exact = hard_to_approximate_func(multi_index)
        
        # 計算 TCI 重建出來的近似值
        # TensorTrain.eval() 需要一個 list[int]
        val_approx = final_tt.eval(list(multi_index))
        
        current_error = abs(val_exact - val_approx)
        if current_error > max_abs_error:
            max_abs_error = current_error

    logger.info(f"全局最大絕對誤差 (Max Absolute Error): {max_abs_error:.6e}")

    # --- 最終斷言 ---
    # 最終的全局誤差應該在一個比 `reltol` 稍大的合理範圍內
    # 因為 `reltol` 控制的是主元選擇的局部誤差，而非全局誤差
    # 我們設定一個比 `reltol` 大兩個數量級的門檻作為通過標準
    error_threshold = ci_params.reltol * 100 
    
    if max_abs_error < error_threshold:
        logger.info(f"測試通過！全局誤差 {max_abs_error:.6e} 小於門檻 {error_threshold:.6e}")
    else:
        logger.error(f"測試失敗！全局誤差 {max_abs_error:.6e} 超過門檻 {error_threshold:.6e}")
        # 在CI/CD環境中，這裡應該拋出一個AssertionError
        assert max_abs_error < error_threshold, "全局重建誤差過大！"