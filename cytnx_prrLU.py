import cytnx
from cytnx import *
import numpy as np
from IndexSet import *  # 使用全局的 IndexSet 類

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    """
    確保輸入的 Tensor t 至少是 2D 的。如果 t 是 1D，根據 axis 的值進行 reshape。
    """
    shape = t.shape()
    if len(shape) == 1:
        n = shape[0]
        if axis == 0:
            return t.reshape(n, 1)
        else:
            return t.reshape(1, n)
    return t

class RankRevealingLU:
    def __init__(self, T: UniTensor, verbose: bool = False, pivot_method: str = "full", tol: float = 1e-6):
        """
        初始化 RankRevealingLU 分解器。

        :param T: 輸入的 UniTensor
        :param verbose: 是否顯示詳細調試訊息
        :param pivot_method: 搜索 pivot 的方法，"full" 或 "rook"
        :param tol: 誤差容限，當剩餘部分誤差低於此值時提前停止分解
        """
        self.T = T
        self.M = T.get_block()  # 取得內部的 Tensor
        self.M = ensure_2d(self.M, axis=0)
        shape = self.M.shape()
        if verbose:
            print("Initial tensor shape:", shape)
        if len(shape) == 1:
            self.n = shape[0]
            self.m = 1
            self.M = self.M.reshape(self.n, self.m)
            if verbose:
                print("Reshaped tensor to (n, m):", self.n, self.m)
        elif len(shape) == 2:
            self.n, self.m = shape
        else:
            raise ValueError("The input tensor must be 1D or 2D for LU decomposition.")
        # 用輸入矩陣的維度決定分解步數：通常取 min(n, m)
        self.N = min(self.n, self.m)
        self.verbose = verbose
        self.pivot_method = pivot_method
        self.tol = tol  # 誤差容限
        self.L = cytnx.eye(self.n, dtype=Type.Double, device=Device.cpu)
        self.U = cytnx.eye(self.m, dtype=Type.Double, device=Device.cpu)
        self.D = cytnx.zeros((self.n, self.m), dtype=Type.Double, device=Device.cpu)
        self.perL = []
        self.perU = []
        self.error = None
        # 全局 pivot 記錄 (存 tuple (i0, j0))
        self.pivot_set = IndexSet()
        # 全局秩信息與步驟誤差記錄
        self.rank_info = []     # 每次迭代後的有效秩，例如 i+1
        self.step_errors = []   # 每次迭代後剩餘子矩陣的誤差

    def find_pivot_full(self, subT: Tensor):
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        i0, j0 = 0, 0
        maxv = 0.0
        for i in range(n):
            for j in range(m):
                val = abs(subT[i, j].item())
                if val > maxv:
                    maxv = val
                    i0, j0 = i, j
        if self.verbose:
            print(f"find_pivot_full: pivot at ({i0}, {j0}) with value {maxv}")
        return i0, j0

    def find_pivot_rook(self, subT: Tensor):
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        if n == 0 or m == 0:
            return 0, 0
        if n >= m:
            i0 = np.random.randint(0, n - 1)
            j0 = 0
            for j in range(m):
                if abs(subT[i0, j].item()) > abs(subT[i0, j0].item()):
                    j0 = j
        else:
            j0 = np.random.randint(0, m - 1)
            i0 = 0
            for i in range(n):
                if abs(subT[i, j0].item()) > abs(subT[i0, j0].item()):
                    i0 = i
        if self.verbose:
            print(f"find_pivot_rook: pivot at ({i0}, {j0}) with value {abs(subT[i0,j0].item())}")
        return i0, j0

    def swap_rows_inplace(self, tA: Tensor, r1: int, r2: int):
        tA = ensure_2d(tA, axis=0)
        if r1 == r2:
            return
        row1 = tA[r1, :].clone()
        row2 = tA[r2, :].clone()
        tA[r1, :] = row2
        tA[r2, :] = row1

    def swap_cols_inplace(self, tA: Tensor, c1: int, c2: int):
        tA = ensure_2d(tA, axis=0)
        if c1 == c2:
            return
        col1 = tA[:, c1].clone()
        col2 = tA[:, c2].clone()
        tA[:, c1] = col2
        tA[:, c2] = col1

    def PLDU(self, subT: Tensor):
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        T_local = subT.clone()

        # 使用所選的 pivot 搜索方法
        if self.pivot_method.lower() == "rook":
            i0, j0 = self.find_pivot_rook(T_local)
        else:
            i0, j0 = self.find_pivot_full(T_local)

        # 記錄原始 pivot 位置
        self.pivot_set.push_back((i0, j0))

        pivot_val = T_local[i0, j0].item()
        if abs(pivot_val) < 1e-14:
            if self.verbose:
                print("Pivot too small:", pivot_val)
            return -1, -1, None, None, None

        if i0 != 0:
            self.swap_rows_inplace(T_local, 0, i0)
        if j0 != 0:
            self.swap_cols_inplace(T_local, 0, j0)
        pivot_val = T_local[0, 0].item()

        L_local = cytnx.eye(n, dtype=Type.Double, device=Device.cpu)
        U_local = cytnx.eye(m, dtype=Type.Double, device=Device.cpu)
        for r in range(1, n):
            L_local[r, 0] = T_local[r, 0].item() / pivot_val
        for c in range(1, m):
            U_local[0, c] = T_local[0, c].item() / pivot_val

        for rr in range(1, n):
            for cc in range(1, m):
                T_local[rr, cc] = T_local[rr, cc].item() - L_local[rr, 0].item() * T_local[0, cc].item()
        return i0, j0, L_local, T_local, U_local

    def PrrLU(self):
        Temp = self.M.clone()
        for i in range(self.N):
            subN = self.n - i
            subM = self.m - i
            if subN < 1 or subM < 1:
                break
            subT = Temp[i:, i:].clone()
            subT = ensure_2d(subT, axis=0)
            i0, j0, L_step, newT, U_step = self.PLDU(subT)
            if i0 < 0 or j0 < 0:
                break
            # 將當前步驟的 pivot 已在 PLDU 中記錄到 pivot_set
            pivot_val = newT[0, 0].item()
            self.D[i, i] = pivot_val

            self.perL.insert(0, [i, i + i0])
            self.perU.insert(0, [i, i + j0])

            if i > 0:
                L_sub = self.L[i:, :i].clone()
                U_sub = self.U[:i, i:].clone()
                subN = self.n - i
                subM = self.m - i
                if i0 < subN:
                    self.swap_rows_inplace(L_sub, 0, i0)
                if j0 < subM:
                    self.swap_cols_inplace(U_sub, 0, j0)
                self.L[i:, :i] = L_sub
                self.U[:i, i:] = U_sub

            for rr in range(subN):
                self.L[i+rr, i] = L_step[rr, 0].item()
            for cc in range(subM):
                self.U[i, i+cc] = U_step[0, cc].item()

            if subN > 1 and subM > 1:
                for rr in range(1, subN):
                    for cc in range(1, subM):
                        Temp[i+rr, i+cc] = newT[rr, cc].item()

            # 記錄當前步驟的秩信息與誤差估計
            current_rank = i + 1
            self.rank_info.append(current_rank)
            current_error = cytnx.linalg.Norm(Temp[i:, i:]).item()
            self.step_errors.append(current_error)
            if self.verbose:
                print(f"[RankRevealingLU] after step i={i}")
                print("Temp=\n", Temp)
                print("L=\n", self.L)
                print("D=\n", self.D)
                print("U=\n", self.U)
                print(f"[RankRevealingLU] current rank = {current_rank}, current error = {current_error}")
            # 若當前誤差低於容限，提前停止
            if current_error <= self.tol:
                if self.verbose:
                    print(f"[RankRevealingLU] current error {current_error} <= tol {self.tol}, stopping iteration.")
                break

        for (x, y) in self.perL:
            self.swap_rows_inplace(self.L, x, y)
        for (x, y) in self.perU:
            self.swap_cols_inplace(self.U, x, y)

        LDU = self.L @ self.D @ self.U
        diff = self.M - LDU
        self.error = cytnx.linalg.Norm(diff).item()
        if self.verbose:
            print("[RankRevealingLU] final residual =", self.error)
        ut_L = UniTensor(self.L)
        ut_D = UniTensor(self.D)
        ut_U = UniTensor(self.U)
        return ut_L, ut_D, ut_U

    def get_error(self):
        return self.error

    def reconstruct(self):
        return self.L @ self.D @ self.U

    def get_rank_info(self):
        return self.rank_info
    
    def get_step_errors(self):
        return self.step_errors

if __name__=="__main__":
    # 測試用矩陣
    M, N = 6, 5
    np.random.seed(0)
    arr = np.random.rand(M, N)
    T = from_numpy(arr)
    uT = UniTensor(T, rowrank=2, is_diag=False)
    # 呼叫時不再傳入 N 參數，RankRevealingLU 內部會自動取 min(n, m)
    rrLU = RankRevealingLU(uT, verbose=True, pivot_method="rook", tol=1e-6)
    uT.print_diagram()
    L, D, U = rrLU.PrrLU()
    L.print_diagram()
    D.print_diagram()
    U.print_diagram()
    print("Error =", rrLU.get_error())
    rec = rrLU.reconstruct()
    print("Reconstructed =\n", rec)
    print("Global pivot set =", rrLU.pivot_set.get_all())
    print("Rank info =", rrLU.rank_info)
    print("Step errors =", rrLU.step_errors)
