import cytnx
from cytnx import *
import numpy as np
import random
from IndexSet import *  # Assuming IndexSet class is defined elsewhere

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    """
    Ensure the input Tensor t is at least 2D. If 1D, reshape it based on the axis value.
    
    :param t: Input Tensor
    :param axis: Axis to expand (0 for rows, 1 for columns)
    :return: Reshaped 2D Tensor
    """
    shape = t.shape()
    if len(shape) == 1:
        n = shape[0]
        if axis == 0:
            return t.reshape(n, 1)
        else:
            return t.reshape(1, n)
    return t

class AdaptiveLU:
    def __init__(self, n_rows: int, n_cols: int, verbose: bool = False, pivot_method: str = "full", tol: float = 1e-6):
        """
        Initialize the AdaptiveLU decomposition class.

        :param n_rows: Number of rows in the matrix
        :param n_cols: Number of columns in the matrix
        :param verbose: Whether to display detailed debugging information
        :param pivot_method: Pivot search method, either "full" or "rook"
        :param tol: Error tolerance; stops decomposition if residual error falls below this
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.verbose = verbose
        self.pivot_method = pivot_method
        self.tol = tol
        self.Iset = []  # Selected row indices
        self.Jset = []  # Selected column indices
        self.L = cytnx.eye(n_rows, dtype=Type.Double, device=Device.cpu)  # Lower triangular matrix
        self.U = cytnx.eye(n_cols, dtype=Type.Double, device=Device.cpu)  # Upper triangular matrix
        self.D = cytnx.zeros((n_rows, n_cols), dtype=Type.Double, device=Device.cpu)  # Diagonal matrix
        self.perL = []  # Row permutation records
        self.perU = []  # Column permutation records
        self.error = None
        self.rank_info = []  # Rank after each step
        self.step_errors = []  # Error after each step
        self.pivot_set = IndexSet()  # Global pivot record

    def find_pivot_full(self, subT: Tensor):
        """Find the pivot with the maximum absolute value in the submatrix."""
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        i0, j0, maxv = 0, 0, 0.0
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
        """Find a pivot using the rook strategy (random start, then max in row/column)."""
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        if n == 0 or m == 0:
            return 0, 0
        if n >= m:
            i0 = np.random.randint(0, n)
            j0 = 0
            for j in range(m):
                if abs(subT[i0, j].item()) > abs(subT[i0, j0].item()):
                    j0 = j
        else:
            j0 = np.random.randint(0, m)
            i0 = 0
            for i in range(n):
                if abs(subT[i, j0].item()) > abs(subT[i0, j0].item()):
                    i0 = i
        if self.verbose:
            print(f"find_pivot_rook: pivot at ({i0}, {j0}) with value {abs(subT[i0, j0].item())}")
        return i0, j0

    def swap_rows_inplace(self, tA: Tensor, r1: int, r2: int):
        """Swap two rows in the Tensor in-place."""
        tA = ensure_2d(tA, axis=0)
        if r1 != r2:
            row1 = tA[r1, :].clone()
            row2 = tA[r2, :].clone()
            tA[r1, :] = row2
            tA[r2, :] = row1

    def swap_cols_inplace(self, tA: Tensor, c1: int, c2: int):
        """Swap two columns in the Tensor in-place."""
        tA = ensure_2d(tA, axis=0)
        if c1 != c2:
            col1 = tA[:, c1].clone()
            col2 = tA[:, c2].clone()
            tA[:, c1] = col2
            tA[:, c2] = col1

    def add_pivot_row(self, i: int, row: Tensor):
        """Perform Gaussian elimination by adding a pivot row."""
        self.Iset.append(i)
        k = len(self.Iset) - 1
        new_U = cytnx.zeros((self.U.shape()[0] + 1, self.U.shape()[1]))
        new_U[:self.U.shape()[0], :] = self.U
        new_U[self.U.shape()[0], :] = row
        self.U = new_U
        print
        for l in range(k):
            self.U[k, :] -= self.U[l, :] * (self.L[self.Iset[k], l].item() * self.D[l, l].item())

    def add_pivot_col(self, j: int, col: Tensor):
        """Perform Gaussian elimination by adding a pivot column."""
        self.Jset.append(j)
        k = len(self.Jset) - 1
        new_L = cytnx.zeros((self.L.shape()[0]+1,self.L.shape()[1]))
        new_L[:self.L.shape()[0],:] = self.L
        new_L[self.L.shape()[0],:] = col
        self.L = new_L
        for l in range(k):
            self.L[:, k] -= self.L[:, l] * (self.U[l, self.Jset[k]].item() * self.D[l, l].item())
        self.D[k, k] = 1.0 / self.L[self.Iset[k], k].item()

    def PLDU(self, subT: Tensor):
        """Perform a local LU decomposition step with pivoting."""
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        T_local = subT.clone()

        # Select pivot based on method
        i0, j0 = (self.find_pivot_rook if self.pivot_method.lower() == "rook" else self.find_pivot_full)(T_local)
        self.pivot_set.push_back((i0, j0))

        pivot_val = T_local[i0, j0].item()
        if abs(pivot_val) < 1e-14:
            if self.verbose:
                print(f"Pivot too small: {pivot_val}")
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

    def PrrLU(self, M: Tensor):
        """Perform the rank-revealing LU decomposition."""
        Temp = M.clone()
        N = min(self.n_rows, self.n_cols)
        for i in range(N):
            subN, subM = self.n_rows - i, self.n_cols - i
            if subN < 1 or subM < 1:
                break
            subT = Temp[i:, i:].clone()
            i0, j0, L_step, newT, U_step = self.PLDU(subT)
            if i0 < 0 or j0 < 0:
                break

            pivot_val = newT[0, 0].item()
            self.D[i, i] = pivot_val
            self.perL.insert(0, [i, i + i0])
            self.perU.insert(0, [i, i + j0])

            if i > 0:
                L_sub = self.L[i:, :i].clone()
                U_sub = self.U[:i, i:].clone()
                if i0 < subN:
                    self.swap_rows_inplace(L_sub, 0, i0)
                if j0 < subM:
                    self.swap_cols_inplace(U_sub, 0, j0)
                self.L[i:, :i] = L_sub
                self.U[:i, i:] = U_sub

            for rr in range(subN):
                self.L[i + rr, i] = L_step[rr, 0].item()
            for cc in range(subM):
                self.U[i, i + cc] = U_step[0, cc].item()

            if subN > 1 and subM > 1:
                for rr in range(1, subN):
                    for cc in range(1, subM):
                        Temp[i + rr, i + cc] = newT[rr, cc].item()

            current_rank = i + 1
            self.rank_info.append(current_rank)
            current_error = cytnx.linalg.Norm(Temp[i:, i:]).item()
            self.step_errors.append(current_error)
            if self.verbose:
                print(f"[AdaptiveLU] Step {i}: rank = {current_rank}, error = {current_error}")
                print("Temp =\n", Temp)
                print("L =\n", self.L)
                print("D =\n", self.D)
                print("U =\n", self.U)
            if current_error <= self.tol:
                if self.verbose:
                    print(f"Error {current_error} <= tol {self.tol}, stopping.")
                break

        for x, y in self.perL:
            self.swap_rows_inplace(self.L, x, y)
        for x, y in self.perU:
            self.swap_cols_inplace(self.U, x, y)

        LDU = self.L @ self.D @ self.U
        diff = M - LDU
        self.error = cytnx.linalg.Norm(diff).item()
        if self.verbose:
            print(f"Final residual error = {self.error}")
        return UniTensor(self.L), UniTensor(self.D), UniTensor(self.U)

    def get_error(self):
        """Return the final decomposition error."""
        return self.error

    def reconstruct(self):
        """Reconstruct the matrix from L, D, and U."""
        return self.L @ self.D @ self.U

    def get_rank_info(self):
        """Return the rank information from each step."""
        return self.rank_info

    def get_step_errors(self):
        """Return the error from each step."""
        return self.step_errors

if __name__ == "__main__":
    # Test with a random matrix
    M, N = 6, 5
    np.random.seed(0)
    arr = np.random.rand(M, N)
    T = from_numpy(arr)
    adaptive_lu = AdaptiveLU(M, N, verbose=True, pivot_method="rook", tol=1e-6)
    L, D, U = adaptive_lu.PrrLU(T)
    print("\nL diagram:")
    L.print_diagram()
    print("D diagram:")
    D.print_diagram()
    print("U diagram:")
    U.print_diagram()
    print(f"Error = {adaptive_lu.get_error()}")
    rec = adaptive_lu.reconstruct()
    print("Reconstructed matrix =\n", rec)
    print("Global pivot set =", adaptive_lu.pivot_set.get_all())
    print("Rank info =", adaptive_lu.rank_info)
    print("Step errors =", adaptive_lu.step_errors)