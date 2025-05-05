# -*- coding: utf-8 -*-
# 導入必要的函式庫
import cytnx
from cytnx import * # <--- 使用了 from cytnx import *
import numpy as np
import math
from itertools import product

# --------------------------------------------------------
# 網格生成函數
# --------------------------------------------------------
def uniform_grid(a, b, M):
    """生成一個均勻網格，包含 M 個點，不包含終點 b"""
    if M <= 0: return cytnx.zeros(0, dtype=Type.Double)
    step = (b - a) / float(M)
    end_point = a + (M - 1) * step
    return cytnx.linspace(a, end_point, M, dtype=Type.Double)

def gauss_kronrod_grid(a, b):
    """生成 Gauss-Kronrod 網格 (15點)"""
    print("警告：Gauss-Kronrod 網格與 Quantics 位元邏輯不兼容，建議使用 'uniform'")
    M = 15; xs = cytnx.linspace(a, b, M, dtype=Type.Double); ws = cytnx.ones(M, dtype=Type.Double) * ((b - a) / M); return xs, ws

# --------------------------------------------------------
# QuanticGrid 類別：生成量化張量 (使用直接的 from_numpy)
# --------------------------------------------------------
class QuanticGrid:
    """
    表示一個 Quantics 網格，用於將函數離散化為張量。
    邏輯已修正以匹配 C++ 版本。
    """
    def __init__(self, a=0.0, b=1.0, nBit=4, dim=1, fused=False, grid_method="uniform", custom_grid_func=None, f=None):
        if not isinstance(nBit, int) or nBit <= 0: raise ValueError("nBit 必須是正整數")
        if not isinstance(dim, int) or dim <= 0: raise ValueError("dim 必須是正整數")
        if b <= a: raise ValueError("b 必須大於 a")
        self.a = float(a); self.b = float(b); self.nBit = nBit; self.dim = dim; self.fused = fused
        self.M = 1 << nBit; self.deltaX = (self.b - self.a) / float(self.M) if self.M > 0 else 0
        self.grid_method = grid_method.lower(); self.custom_grid_func = custom_grid_func; self.f = f
        self.grids = []

        for i in range(dim):
            if self.custom_grid_func is not None and self.grid_method == "custom":
                xs_data, _ = self.custom_grid_func(self.a, self.b)
                if not isinstance(xs_data, cytnx.Tensor):
                    try:
                        # ----------- 使用直接的 from_numpy -----------
                        xs = from_numpy(np.array(xs_data)).astype(Type.Double)
                        # ----------- 修正結束 -----------
                    except NameError:
                         print("錯誤: 'from_numpy' 未定義。請確保 'from cytnx import *' 或相關導入正確。")
                         raise
                    except Exception as e:
                         raise TypeError(f"無法將自定義網格轉換為 cytnx.Tensor: {e}")
                else:
                    xs = xs_data # 如果已經是 Tensor，直接使用
                if xs.shape()[0] != self.M: print(f"警告：自定義網格維度 {i} 的大小 {xs.shape()} 與 M={self.M} 不匹配")
            elif self.grid_method == "uniform":
                xs = uniform_grid(self.a, self.b, self.M)
                if xs.shape()[0] != self.M and self.M > 0:
                     print(f"警告：uniform_grid 維度 {i} 產生了 {xs.shape()[0]} 個點，預期為 {self.M}")
                     end_point = self.a + (self.M - 1) * self.deltaX
                     xs = cytnx.linspace(self.a, end_point, self.M, dtype=Type.Double)
            elif self.grid_method == "gk": raise ValueError("Gauss-Kronrod 網格與嚴格的 Quantics 位元邏輯不兼容。請使用 'uniform'。")
            else: raise ValueError("grid_method 必須是 'uniform', 'gk', 或 'custom'")
            if xs.dtype() != Type.Double: xs = xs.astype(Type.Double)
            self.grids.append(xs)

    def coord_to_int_index(self, coord):
        if self.deltaX == 0: return 0
        k = math.floor((coord - self.a) / self.deltaX)
        return max(0, min(self.M - 1, k))

    def coord_to_bin(self, coord):
        k = self.coord_to_int_index(coord)
        return [int(bit) for bit in format(k, f'0{self.nBit}b')]

    # ... (get_bin_indices_for_dim, get_cartesian_* 保持不變) ...
    def get_bin_indices_for_dim(self, dim_index):
        if dim_index < 0 or dim_index >= self.dim: raise IndexError("維度索引超出範圍")
        arr = []; grid_points_np = self.grids[dim_index].numpy()
        for i in range(self.M): arr.append(self.coord_to_bin(grid_points_np[i]))
        return np.array(arr)
    def get_cartesian_grid_indices(self):
        return list(product(range(self.M), repeat=self.dim))
    def get_cartesian_grid_coords(self):
        indices_comb = self.get_cartesian_grid_indices(); coords_comb = []
        np_grids = [g.numpy() for g in self.grids]
        for indices in indices_comb: coords_comb.append([np_grids[d][indices[d]] for d in range(self.dim)])
        return coords_comb


    def get_quantics_tensor(self):
        if self.f is None: raise ValueError("函數 f 未在 QuanticGrid 中定義。")
        pts_coords = self.get_cartesian_grid_coords()
        try: values = np.array([float(self.f(*pt)) for pt in pts_coords], dtype=np.float64)
        except Exception as e: print(f"評估函數 f 時出錯：{e}"); raise
        initial_shape = (self.M,) * self.dim

        # ----------- 使用直接的 from_numpy -----------
        try:
            T_block = from_numpy(values.reshape(initial_shape))
        except NameError:
            print("錯誤: 'from_numpy' 未定義。請確保 'from cytnx import *' 或相關導入正確。")
            raise
        # ----------- 修正結束 -----------

        if T_block.dtype() != Type.Double: T_block = T_block.astype(Type.Double)
        labels = [f'k{i}' for i in range(self.dim)]; T_uni = UniTensor(T_block, rowrank=0, labels=labels)
        return T_uni

    def reshape_to_mps(self, T_uni):
        if not isinstance(T_uni, UniTensor): raise TypeError("輸入必須是 cytnx.UniTensor")
        initial_shape = (self.M,) * self.dim
        if T_uni.shape() != list(initial_shape):
             raise ValueError(f"輸入 UniTensor 形狀 {T_uni.shape()} 與預期形狀 {initial_shape} 不符")
        D = self.dim; N = self.nBit

        if not self.fused:
            # --- Non-Fused 模式 ---
            target_total_bits = D * N
            if target_total_bits == 0: # Handle nBit=0 or dim=0 edge case
                 return T_uni.clone() # Or return an empty UniTensor?
            initial_nonfused_shape = [2] * target_total_bits
            T_work = T_uni.clone()
            T_work.reshape_(*initial_nonfused_shape)
            permute_indices = [0] * target_total_bits
            for d_orig in range(D):
                for b_orig in range(N):
                    idx_orig = d_orig * N + b_orig; idx_target = b_orig * D + d_orig
                    permute_indices[idx_orig] = idx_target
            # Only permute if rank > 1
            if T_work.rank() > 1:
                 T_work.permute_(permute_indices)
            final_labels = [f'b{bit}d{d}' for bit in range(N) for d in range(D)]
            if len(final_labels) == T_work.rank(): # Check rank matches label count
                 T_work.set_labels(final_labels)
            T_work.set_rowrank(0)
            return T_work
        else:
            # --- Fused 模式 (手動 NumPy 映射) ---
            if N == 0:
                 T_block_fused = cytnx.zeros([], dtype=T_uni.dtype(), device=T_uni.device())
                 return UniTensor(T_block_fused, rowrank=0)

            T_np = T_uni.get_block().numpy()
            fused_dim_size = 2**D
            fused_shape_np = tuple([fused_dim_size] * N)
            T_fused_np = np.zeros(fused_shape_np, dtype=T_np.dtype)
            fused_indices_iterator = product(range(fused_dim_size), repeat=N)
            powers_of_2_b = [1 << b for b in range(N)]

            for fused_indices in fused_indices_iterator:
                k_indices = [0] * D; valid_k = True
                for b in range(N):
                    fb = fused_indices[b]
                    for d in range(D):
                        sigma_db = (fb >> d) & 1
                        k_indices[d] += sigma_db * powers_of_2_b[b]
                if any(k >= self.M for k in k_indices): continue
                try: T_fused_np[fused_indices] = T_np[tuple(k_indices)]
                except IndexError: continue

            # ----------- 使用直接的 from_numpy -----------
            try:
                T_block_fused = from_numpy(T_fused_np)
            except NameError:
                print("錯誤: 'from_numpy' 未定義。請確保 'from cytnx import *' 或相關導入正確。")
                raise
            # ----------- 修正結束 -----------

            if T_block_fused.dtype() != T_uni.dtype():
                 T_block_fused = T_block_fused.astype(T_uni.dtype())

            final_labels = [f'fused_b{bit}' for bit in range(N)]
            new_rowrank = 1 if N > 0 else 0
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
    # === 測試 1: 1D ===
    print("--- 測試 1: 1D (nBit=2, M=4) ---")
    qg1d = QuanticGrid(a=0.0, b=4.0, nBit=2, dim=1, fused=False, f=f_linear)
    T1d_uni = qg1d.get_quantics_tensor()
    print(f"初始張量 T (1D, shape={T1d_uni.shape()}):\n{T1d_uni.get_block().numpy()}")
    T1d_mps = qg1d.reshape_to_mps(T1d_uni.clone())
    print(f"重塑後張量 T_mps (1D, non-fused, shape={T1d_mps.shape()}, labels={T1d_mps.labels()}):\n{T1d_mps.get_block().numpy()}")
    # 預期: [[1. 5.] [3. 7.]]

    # === 測試 2: 2D Non-Fused (nBit=1, M=2) ===
    print("\n--- 測試 2: 2D Non-Fused (nBit=1, M=2) ---")
    qg2d_nf = QuanticGrid(a=0.0, b=2.0, nBit=1, dim=2, fused=False, f=f_2d)
    T2d_uni = qg2d_nf.get_quantics_tensor()
    print(f"初始張量 T (2D, shape={T2d_uni.shape()}):\n{T2d_uni.get_block().numpy()}")
    T2d_mps_nf = qg2d_nf.reshape_to_mps(T2d_uni.clone())
    print(f"重塑後張量 T_mps (2D, non-fused, shape={T2d_mps_nf.shape()}, labels={T2d_mps_nf.labels()}):\n{T2d_mps_nf.get_block().numpy()}")
    # 預期: [[0. 2.] [1. 3.]]

    # === 測試 3: 2D Fused (nBit=1, M=2) ===
    print("\n--- 測試 3: 2D Fused (nBit=1, M=2) ---")
    qg2d_f = QuanticGrid(a=0.0, b=2.0, nBit=1, dim=2, fused=True, f=f_2d)
    T2d_mps_f = qg2d_f.reshape_to_mps(T2d_uni.clone())
    print(f"重塑後張量 T_mps (2D, fused, shape={T2d_mps_f.shape()}, labels={T2d_mps_f.labels()}):\n{T2d_mps_f.get_block().numpy()}")
    # 預期: [0., 1., 2., 3.] (手動映射應該能得到這個結果)

    # === 測試 4: 2D Non-Fused (nBit=2, M=4) ===
    print("\n--- 測試 4: 2D Non-Fused (nBit=2, M=4) ---")
    qg2d_nf2 = QuanticGrid(a=0.0, b=4.0, nBit=2, dim=2, fused=False, f=f_2d)
    T2d_uni2 = qg2d_nf2.get_quantics_tensor()
    print(f"初始張量 T (2D, nBit=2, shape={T2d_uni2.shape()}):")
    T2d_mps_nf2 = qg2d_nf2.reshape_to_mps(T2d_uni2.clone())
    print(f"重塑後張量 T_mps (2D, non-fused, nBit=2, shape={T2d_mps_nf2.shape()}, labels={T2d_mps_nf2.labels()}):")
    print(T2d_mps_nf2.get_block().numpy().flatten())
    expected_flat_nf = np.array([0., 1., 2., 3., 2., 3., 4., 5., 4., 5., 6., 7., 6., 7., 8., 9.])
    print(f"預期展平值 (基於位元分組邏輯): {expected_flat_nf}")

    # === 測試 5: 3D Fused (nBit=1, M=2) ===
    print("\n--- 測試 5: 3D Fused (nBit=1, M=2) ---")
    qg3d_f = QuanticGrid(a=0.0, b=1.0, nBit=1, dim=3, fused=True, f=f_3d)
    T3d_uni = qg3d_f.get_quantics_tensor()
    print(f"初始張量 T (3D, nBit=1, shape={T3d_uni.shape()}):\n{T3d_uni.get_block().numpy()}")
    T3d_mps_f = qg3d_f.reshape_to_mps(T3d_uni.clone())
    print(f"重塑後張量 T_mps (3D, fused, nBit=1, shape={T3d_mps_f.shape()}, labels={T3d_mps_f.labels()}):\n{T3d_mps_f.get_block().numpy()}")
    # 預期: [0.  0.5 1.  1.5 1.5 2.  2.5 3. ] (手動映射應該能得到這個結果)