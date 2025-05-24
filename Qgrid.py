# -*- coding: utf-8 -*-
# 導入必要的函式庫
import cytnx
from cytnx import * # 使用了 from cytnx import *
import numpy as np
import math
from itertools import product

# --------------------------------------------------------
# 網格生成函數
# --------------------------------------------------------
def uniform_grid(a, b, M):
    """生成一個均勻網格，包含 M 個點。"""
    if M <= 0: return cytnx.zeros(0, dtype=Type.Double)
    # 使用 linspace 計算 M 個點的座標，包含起點 a，終點為 a + (M-1)*step
    step = (b - a) / float(M)
    end_point = a + (M - 1) * step
    return cytnx.linspace(a, end_point, M, dtype=Type.Double)

# --------------------------------------------------------
# QuanticGrid 類別 (已簡化)
# --------------------------------------------------------
class QuanticGrid:
    """
    表示一個 Quantics 網格，用於將函數離散化為張量。
    邏輯與 C++ 版本一致。
    """
    def __init__(self, a=0.0, b=1.0, nBit=4, dim=1, fused=False, grid_method="uniform", custom_grid_func=None, f=None):
        if not isinstance(nBit, int) or nBit <= 0: raise ValueError("nBit 必須是正整數")
        if not isinstance(dim, int) or dim <= 0: raise ValueError("dim 必須是正整數")
        if b <= a: raise ValueError("b 必須大於 a")

        self.a = float(a); self.b = float(b); self.nBit = nBit; self.dim = dim; self.fused = fused
        self.M = 1 << nBit; self.deltaX = (self.b - self.a) / float(self.M) if self.M > 0 else 0
        self.grid_method = grid_method.lower(); self.custom_grid_func = custom_grid_func; self.f = f
        self.grids = [] # 儲存 cytnx.Tensor

        for i in range(dim):
            xs = None
            if self.custom_grid_func is not None and self.grid_method == "custom":
                xs_data, _ = self.custom_grid_func(self.a, self.b)
                if not isinstance(xs_data, Tensor):
                    xs = from_numpy(np.array(xs_data)) # 直接使用 from_numpy
                else:
                    xs = xs_data
                if xs.shape()[0] != self.M:
                    print(f"警告：自定義網格維度 {i} 的大小 {xs.shape()} 與 M={self.M} 不匹配")
            elif self.grid_method == "uniform":
                xs = uniform_grid(self.a, self.b, self.M)
                if xs.shape()[0] != self.M and self.M > 0:
                     # 警告表明 uniform_grid/linspace 的內部問題
                     print(f"嚴重警告：uniform_grid 未能產生預期的 {self.M} 個點 (got {xs.shape()[0]})")
            elif self.grid_method == "gk":
                raise ValueError("Gauss-Kronrod 網格與 Quantics 位元邏級不兼容。")
            else:
                raise ValueError("grid_method 必須是 'uniform', 'gk', 或 'custom'")

            if xs is None: # 如果未能生成 xs
                raise RuntimeError(f"未能為維度 {i} 生成網格數據。")
            if xs.dtype() != Type.Double:
                xs = xs.astype(Type.Double)
            self.grids.append(xs)

    def coord_to_int_index(self, coord):
        """座標 -> 整數索引 k (0 到 M-1)。"""
        if self.deltaX == 0: return 0
        # math.floor 比 int() 更明確處理負數 (雖然這裡 a<=coord<b)
        k = math.floor((coord - self.a) / self.deltaX)
        # 限制範圍處理邊界浮點數問題
        return max(0, min(self.M - 1, k))

    def coord_to_bin(self, coord):
        """座標 -> nBit 二進制列表。"""
        k = self.coord_to_int_index(coord)
        return [int(bit) for bit in format(k, f'0{self.nBit}b')]

    def get_cartesian_grid_coords(self):
        """生成所有笛卡爾網格點的座標組合。"""
        np_grids = [g.numpy() for g in self.grids] # 一次性轉換
        coords_comb = []
        # 使用 product 生成索引組合
        for indices in product(range(self.M), repeat=self.dim):
            coords_comb.append([np_grids[d][indices[d]] for d in range(self.dim)])
        return coords_comb

    def get_quantics_tensor(self):
        """計算函數值並返回初始 UniTensor。"""
        if self.f is None: raise ValueError("函數 f 未定義。")
        pts_coords = self.get_cartesian_grid_coords()
        try:
            # 確保函數輸出為 float64
            values = np.array([float(self.f(*pt)) for pt in pts_coords], dtype=np.float64)
        except Exception as e:
            print(f"評估函數 f 時出錯：{e}"); raise
        initial_shape = (self.M,) * self.dim
        T_block = from_numpy(values.reshape(initial_shape)) # 直接使用 from_numpy
        if T_block.dtype() != Type.Double: T_block = T_block.astype(Type.Double)
        # 初始標籤和 rowrank
        labels = [f'k{i}' for i in range(self.dim)]; T_uni = UniTensor(T_block, rowrank=0, labels=labels)
        return T_uni

    def reshape_to_mps(self, T_uni):
        """將 UniTensor 重塑為目標 Quantics 張量 (non-fused 或 fused)。"""
        if not isinstance(T_uni, UniTensor): raise TypeError("輸入必須是 cytnx.UniTensor")
        initial_shape = (self.M,) * self.dim
        if T_uni.shape() != list(initial_shape):
             raise ValueError(f"輸入 UniTensor 形狀 {T_uni.shape()} 與預期 {initial_shape} 不符")

        D = self.dim; N = self.nBit
        target_total_bits = D * N

        # 處理 nBit=0 或 dim=0 的邊界情況
        if target_total_bits == 0:
            return T_uni.clone() # 返回原始 (可能是空) Tensor

        if not self.fused:
            # --- Non-Fused ---
            initial_nonfused_shape = [2] * target_total_bits
            T_work = T_uni.clone()
            T_work.reshape_(*initial_nonfused_shape) # Reshape to (2, 2, ..., 2)

            # 計算置換索引 (將 dim-grouped 轉為 bit-grouped)
            permute_indices = [0] * target_total_bits
            for d_orig in range(D):
                for b_orig in range(N):
                    idx_orig = d_orig * N + b_orig; idx_target = b_orig * D + d_orig
                    permute_indices[idx_orig] = idx_target

            if T_work.rank() > 1: T_work.permute_(permute_indices) # 應用置換

            # 設置標籤
            final_labels = [f'b{bit}d{d}' for bit in range(N) for d in range(D)]
            if len(final_labels) == T_work.rank(): T_work.set_labels(final_labels)
            T_work.set_rowrank(0) # 設置默認 rowrank
            return T_work
        else:
            # --- Fused (手動 NumPy 映射) ---
            T_np = T_uni.get_block().numpy() # 獲取原始數據
            fused_dim_size = 2**D
            fused_shape_np = tuple([fused_dim_size] * N)
            T_fused_np = np.zeros(fused_shape_np, dtype=T_np.dtype) # 創建目標陣列

            powers_of_2_b = [1 << b for b in range(N)] # 預計算 2^b

            # 遍歷目標 fused 索引，計算來源 k 索引並賦值
            for fused_indices in product(range(fused_dim_size), repeat=N):
                k_indices = [0] * D
                for b in range(N):
                    fb = fused_indices[b]
                    for d in range(D):
                        sigma_db = (fb >> d) & 1 # 提取位元
                        k_indices[d] += sigma_db * powers_of_2_b[b] # 重建 k_d

                # 邊界檢查 (理論上不需要，但保留以防萬一)
                if any(k >= self.M for k in k_indices): continue
                try:
                    T_fused_np[fused_indices] = T_np[tuple(k_indices)] # 核心賦值
                except IndexError: continue # 處理可能的索引錯誤

            # 從 NumPy 轉換回 Cytnx Tensor
            T_block_fused = from_numpy(T_fused_np) # 直接使用 from_numpy
            if T_block_fused.dtype() != T_uni.dtype():
                 T_block_fused = T_block_fused.astype(T_uni.dtype())

            # 創建 UniTensor
            final_labels = [f'fused_b{bit}' for bit in range(N)]
            new_rowrank = 1 if N > 0 else 0 # 物理腿 rowrank=1
            T_work_fused = UniTensor(T_block_fused, rowrank=new_rowrank, labels=final_labels)
            return T_work_fused

# ============================================================
# 主程式：測試案例 (保持不變)
# ============================================================
def f_linear(x): return 2.0 * x + 1.0
def f_2d(x, y): return float(x) + 2.0 * float(y)
def f_3d(x, y, z): return float(x) + 2.0 * float(y) + 3.0 * float(z)

if __name__ == "__main__":
    np.set_printoptions(linewidth=150, precision=5, suppress=True)

    # --- 測試 1: 1D ---
    print("--- 測試 1: 1D (nBit=2, M=4) ---")
    qg1d = QuanticGrid(a=0.0, b=4.0, nBit=2, dim=1, fused=False, f=f_linear)
    T1d_uni = qg1d.get_quantics_tensor()
    print(f"初始張量 T (1D, shape={T1d_uni.shape()}):\n{T1d_uni.get_block().numpy()}")
    T1d_mps = qg1d.reshape_to_mps(T1d_uni.clone())
    print(f"重塑後張量 T_mps (1D, non-fused, shape={T1d_mps.shape()}, labels={T1d_mps.labels()}):\n{T1d_mps.get_block().numpy()}")
    # 預期: [[1. 5.] [3. 7.]]

    # --- 測試 2: 2D Non-Fused (nBit=1, M=2) ---
    print("\n--- 測試 2: 2D Non-Fused (nBit=1, M=2) ---")
    qg2d_nf = QuanticGrid(a=0.0, b=2.0, nBit=1, dim=2, fused=False, f=f_2d)
    T2d_uni = qg2d_nf.get_quantics_tensor()
    print(f"初始張量 T (2D, shape={T2d_uni.shape()}):\n{T2d_uni.get_block().numpy()}")
    T2d_mps_nf = qg2d_nf.reshape_to_mps(T2d_uni.clone())
    print(f"重塑後張量 T_mps (2D, non-fused, shape={T2d_mps_nf.shape()}, labels={T2d_mps_nf.labels()}):\n{T2d_mps_nf.get_block().numpy()}")
    # 預期: [[0. 2.] [1. 3.]]

    # --- 測試 3: 2D Fused (nBit=1, M=2) ---
    print("\n--- 測試 3: 2D Fused (nBit=1, M=2) ---")
    qg2d_f = QuanticGrid(a=0.0, b=2.0, nBit=1, dim=2, fused=True, f=f_2d)
    T2d_mps_f = qg2d_f.reshape_to_mps(T2d_uni.clone())
    print(f"重塑後張量 T_mps (2D, fused, shape={T2d_mps_f.shape()}, labels={T2d_mps_f.labels()}):\n{T2d_mps_f.get_block().numpy()}")
    # 預期: [0., 1., 2., 3.]

    # --- 測試 4: 2D Non-Fused (nBit=2, M=4) ---
    print("\n--- 測試 4: 2D Non-Fused (nBit=2, M=4) ---")
    qg2d_nf2 = QuanticGrid(a=0.0, b=4.0, nBit=2, dim=2, fused=False, f=f_2d)
    T2d_uni2 = qg2d_nf2.get_quantics_tensor()
    print(f"初始張量 T (2D, nBit=2, shape={T2d_uni2.shape()}):")
    T2d_mps_nf2 = qg2d_nf2.reshape_to_mps(T2d_uni2.clone())
    print(f"重塑後張量 T_mps (2D, non-fused, nBit=2, shape={T2d_mps_nf2.shape()}, labels={T2d_mps_nf2.labels()}):")
    print(f"展平輸出:\n{T2d_mps_nf2.get_block().numpy().flatten()}")
    expected_flat_nf = np.array([0., 2., 1., 3., 4., 6., 5., 7., 2., 4., 3., 5., 6., 8., 7., 9.])
    print(f"預期展平值 (基於 C++ 邏輯 & 實際輸出順序): {expected_flat_nf}")

    # --- 測試 5: 3D Fused (nBit=1, M=2) ---
    print("\n--- 測試 5: 3D Fused (nBit=1, M=2) ---")
    qg3d_f = QuanticGrid(a=0.0, b=1.0, nBit=1, dim=3, fused=True, f=f_3d)
    T3d_uni = qg3d_f.get_quantics_tensor()
    print(f"初始張量 T (3D, nBit=1, shape={T3d_uni.shape()}):\n{T3d_uni.get_block().numpy()}")
    T3d_mps_f = qg3d_f.reshape_to_mps(T3d_uni.clone())
    print(f"重塑後張量 T_mps (3D, fused, nBit=1, shape={T3d_mps_f.shape()}, labels={T3d_mps_f.labels()}):\n{T3d_mps_f.get_block().numpy()}")
    # 預期: [0.  0.5 1.  1.5 1.5 2.  2.5 3. ]