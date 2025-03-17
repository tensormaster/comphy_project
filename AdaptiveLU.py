import logging
import cytnx
from cytnx import *
import numpy as np
import random
from IndexSet import *  # Assuming IndexSet class is defined elsewhere

# 初始化 logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def manual_hstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    """
    Manually stack two tensors horizontally using basic cytnx operations.
    
    Args:
        tensor1: First cytnx.Tensor with shape (m, n)
        tensor2: Second cytnx.Tensor with shape (m, p)
    
    Returns:
        cytnx.Tensor with shape (m, n + p)
    """
    m, n = tensor1.shape()
    p = tensor2.shape()[1]
    if tensor1.shape()[0] != tensor2.shape()[0]:
        raise ValueError("Tensors must have the same number of rows for horizontal stacking.")
    result = cytnx.zeros((m, n + p))
    result[:, :n] = tensor1
    result[:, n:n + p] = tensor2
    logger.debug(f"manual_hstack: stacked tensors with shapes {tensor1.shape()} and {tensor2.shape()} to {result.shape()}")
    return result

def manual_vstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    """
    Manually stack two tensors vertically using basic cytnx operations.
    
    Args:
        tensor1: First cytnx.Tensor with shape (m, n)
        tensor2: Second cytnx.Tensor with shape (p, n)
    
    Returns:
        cytnx.Tensor with shape (m + p, n)
    """
    m, n = tensor1.shape()
    p = tensor2.shape()[0]
    if tensor1.shape()[1] != tensor2.shape()[1]:
        raise ValueError("Tensors must have the same number of columns for vertical stacking.")
    result = cytnx.zeros((m + p, n))
    result[:m, :] = tensor1
    result[m:m + p, :] = tensor2
    logger.debug(f"manual_vstack: stacked tensors with shapes {tensor1.shape()} and {tensor2.shape()} to {result.shape()}")
    return result


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
        self.D = cytnx.zeros((min(self.n_rows, self.n_cols),), dtype=Type.Double, device=Device.cpu)
        self.perL = []  # Row permutation records
        self.perU = []  # Column permutation records
        self.error = None
        self.rank_info = []  # Rank after each step
        self.step_errors = []  # Error after each step
        self.pivot_set = IndexSet()  # Global pivot record
        self.rank = 0
        if self.verbose:
            logger.debug(f"AdaptiveLU initialized with n_rows={n_rows}, n_cols={n_cols}, pivot_method={pivot_method}, tol={tol}")
    
    def npivot(self) -> int:
        return self.rank
    
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
            logger.debug(f"find_pivot_full: pivot at ({i0}, {j0}) with value {maxv}")
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
            logger.debug(f"find_pivot_rook: pivot at ({i0}, {j0}) with value {abs(subT[i0, j0].item())}")
        return i0, j0

    def swap_rows_inplace(self, tA: Tensor, r1: int, r2: int):
        """Swap two rows in the Tensor in-place."""
        tA = ensure_2d(tA, axis=0)
        if r1 != r2:
            row1 = tA[r1, :].clone()
            row2 = tA[r2, :].clone()
            tA[r1, :] = row2
            tA[r2, :] = row1
            if self.verbose:
                logger.debug(f"swap_rows_inplace: Swapped rows {r1} and {r2}")

    def swap_cols_inplace(self, tA: Tensor, c1: int, c2: int):
        """Swap two columns in the Tensor in-place."""
        tA = ensure_2d(tA, axis=0)
        if c1 != c2:
            col1 = tA[:, c1].clone()
            col2 = tA[:, c2].clone()
            tA[:, c1] = col2
            tA[:, c2] = col1
            if self.verbose:
                logger.debug(f"swap_cols_inplace: Swapped columns {c1} and {c2}")

    def add_pivot_row(self, i: int, row: cytnx.Tensor):
        shape = row.shape()
        if len(shape) != 1 or shape[0] != self.n_cols:
            raise ValueError(f"Row must be 1D with length {self.n_cols}, got shape {shape}")
        row_2d = row.reshape(1, self.n_cols)
        self.Iset.append(i)
        k = self.npivot()
        self.U = manual_vstack(self.U, row_2d) if k > 0 else row_2d
        for l in range(k):
            factor = self.L[self.Iset[k], l].item() * self.D[l].item()
            self.U[k, :] -= self.U[l, :] * factor

    def add_pivot_col(self, j: int, col: cytnx.Tensor):
        shape = col.shape()
        if len(shape) != 1 or shape[0] != self.n_rows:
            raise ValueError(f"Column must be 1D with length {self.n_rows}, got shape {shape}")
        col_2d = col.reshape(self.n_rows, 1)
        self.Jset.append(j)
        k = self.rank
        self.L = manual_hstack(self.L, col_2d) if k > 0 else col_2d
        for l in range(k):
            factor = self.U[l, self.Jset[k]].item() * self.D[l].item()
            self.L[:, k] -= self.L[:, l] * factor
        new_D = cytnx.zeros((k + 1,))
        if k > 0:
            new_D[:k] = self.D
        self.D = new_D
        pivot = self.L[self.Iset[k], k].item()
        if abs(pivot) < 1e-10:
            raise ValueError("Pivot element too small, decomposition may be unstable")
        self.D[k] = 1.0 / pivot
        self.rank += 1

    def PLDU(self, subT: Tensor):
        """Perform a local LU decomposition step with pivoting."""
        subT = ensure_2d(subT, axis=0)
        n, m = subT.shape()
        T_local = subT.clone()

        # Select pivot based on method
        if self.pivot_method.lower() == "rook":
            i0, j0 = self.find_pivot_rook(T_local)
        else:
            i0, j0 = self.find_pivot_full(T_local)
        self.pivot_set.push_back((i0, j0))

        pivot_val = T_local[i0, j0].item()
        if abs(pivot_val) < 1e-14:
            if self.verbose:
                logger.debug(f"PLDU: Pivot too small: {pivot_val}")
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
        if self.verbose:
            logger.debug(f"PLDU: Completed local LU step with pivot ({i0}, {j0}), pivot value {pivot_val}")
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
            logger.debug(f"self.D shape before: {self.D.shape()}")
            current_size = self.D.shape()[0]
            if i >= current_size:
                new_size = i + 1
                new_D = cytnx.zeros((new_size,), dtype=Type.Double, device=Device.cpu)
                new_D[:current_size] = self.D  # Copy existing elements
                self.D = new_D  # Update self.D with the resized tensor
            self.D[i] = pivot_val
            logger.debug(f"self.D shape after: {self.D.shape()}")
            logging.debug(f"Resized self.D to shape: {self.D.shape()}")
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
                logger.debug(f"[AdaptiveLU] Step {i}: rank = {current_rank}, error = {current_error}")
                logger.debug(f"Temp matrix after step {i}:\n{Temp}")
                logger.debug(f"L matrix after step {i}:\n{self.L}")
                logger.debug(f"D matrix after step {i}:\n{self.D}")
                logger.debug(f"U matrix after step {i}:\n{self.U}")
            if current_error <= self.tol:
                if self.verbose:
                    logger.debug(f"Error {current_error} <= tol {self.tol}, stopping decomposition.")
                break

        for x, y in self.perL:
            self.swap_rows_inplace(self.L, x, y)
        for x, y in self.perU:
            self.swap_cols_inplace(self.U, x, y)

        # Handle reconstruction based on rank
        if self.rank > 0:
            # Perform slicing and reconstruction
            L_sub = self.L[:, :self.rank]
            U_sub = self.U[:self.rank, :]
            D_diag = cytnx.diag(self.D[:self.rank])
            LDU = L_sub @ D_diag @ U_sub
            # Log shapes safely
            logger.debug(f"Reconstruction shapes - L: {L_sub.shape()}, D: {D_diag.shape()}, U: {U_sub.shape()}")
        else:
            # If rank is 0, return a zero matrix of appropriate size
            LDU = cytnx.zeros((self.n_rows, self.n_cols), dtype=Type.Double, device=Device.cpu)
            logger.debug("Reconstruction: rank is 0, returning zero matrix")

        # Compute residual error
        diff = M - LDU
        self.error = cytnx.linalg.Norm(diff).item()
        if self.verbose:
            logger.debug(f"Final residual error = {self.error}")

        return UniTensor(self.L), UniTensor(self.D), UniTensor(self.U)

    def get_error(self):
        """Return the final decomposition error."""
        return self.error

    def reconstruct(self):
        if self.rank > 0:
            K = self.rank
            D_diag = cytnx.linalg.Diag(self.D[:K])  # Use only the first K elements
            return self.L[:, :K] @ D_diag @ self.U[:K, :]
        else:
            return cytnx.zeros((self.n_rows, self.n_cols), dtype=Type.Double, device=Device.cpu)

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
    logger.debug("\nL diagram:")
    L.print_diagram()
    logger.debug("D diagram:")
    D.print_diagram()
    logger.debug("U diagram:")
    U.print_diagram()
    logger.debug(f"Final error = {adaptive_lu.get_error()}")
    rec = adaptive_lu.reconstruct()
    logger.debug("Reconstructed matrix =\n" + str(rec))
    logger.debug("Global pivot set = " + str(adaptive_lu.pivot_set.get_all()))
    logger.debug("Rank info = " + str(adaptive_lu.rank_info))
    logger.debug("Step errors = " + str(adaptive_lu.step_errors))

    def test_add_multiple_pivots():
        # 測試矩陣 3x3
        M, N = 5, 5
        np.random.seed(42)
        arr = np.random.rand(M, N)
        A = from_numpy(arr)
        logger.debug("Test matrix A:")
        logger.debug(f"{A}")
        
        adaptive_lu = AdaptiveLU(M, N, verbose=True, pivot_method="full", tol=1e-6)
        
        # 依序對每個 pivot index 執行 add_pivot_row 與 add_pivot_col
        for idx in range(min(M,N)):
            row = A[idx, :]  # Extract the idx-th row
            adaptive_lu.add_pivot_row(idx, row)
            col = A[:, idx]  # Extract the idx-th column
            adaptive_lu.add_pivot_col(idx, col)
            logger.debug(f"After pivot index {idx}: Iset = {adaptive_lu.Iset}, Jset = {adaptive_lu.Jset}")
            logger.debug(f"U shape = {adaptive_lu.U.shape()}, L shape = {adaptive_lu.L.shape()}, D shape = {adaptive_lu.D.shape()}")
        
        logger.debug("Final L matrix:")
        logger.debug(f"{adaptive_lu.L}")
        logger.debug("Final D matrix:")
        logger.debug(f"{adaptive_lu.D}")
        logger.debug("Final U matrix:")
        logger.debug(f"{adaptive_lu.U}")
        
        try:
            LU = adaptive_lu.reconstruct()
            diff = A - LU
            err = cytnx.linalg.Norm(diff).item()
            logger.debug(f"Reconstruction error after all pivots: {err:.6e}")
        except Exception as e:
            logger.debug(f"Reconstruction error exception: {e}")
    
    test_add_multiple_pivots()