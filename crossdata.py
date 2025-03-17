import logging
import cytnx
from cytnx import *
import numpy as np
from AdaptiveLU import AdaptiveLU  # Assuming AdaptiveLU is in AdaptiveLU.py

# 初始化 logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    """
    Ensure the input Tensor t is at least 2D. If 1D, reshape it based on the axis value.
    """
    shape = t.shape()
    if len(shape) == 1:
        n = shape[0]
        if axis == 0:
            t2 = t.reshape(n, 1)
        else:
            t2 = t.reshape(1, n)
        logger.debug(f"ensure_2d: reshaped tensor from {shape} to {t2.shape()}")
        return t2
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

class CrossData:
    def __init__(self, n_rows: int, n_cols: int):
        """
        Initialize an empty CrossData instance.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.C = None  # Column submatrix
        self.R = None  # Row submatrix
        self.lu = AdaptiveLU(n_rows, n_cols)  # AdaptiveLU instance for decomposition
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}  # Cache for computed values
        logger.debug(f"Initialized CrossData with {n_rows} rows and {n_cols} columns.")

    def pivotMat(self) -> Tensor:
        """
        Return the pivot matrix, i.e., rows of C corresponding to lu.Iset.
        """
        if self.C is None or not self.lu.Iset:
            logger.debug("pivotMat: No pivot set found, returning None.")
            return None
        Iset_uvec = np.array(self.lu.Iset, dtype=np.uint32)
        pivot_matrix = self.C[Iset_uvec, :]
        logger.debug(f"pivotMat: pivot matrix shape: {pivot_matrix.shape()}")
        return pivot_matrix

    def leftMat(self) -> Tensor:
        """
        Return the left matrix L * D (cached).
        """
        k = self.rank()
        if self.cache['LD'] is None:
            logger.debug(f"leftMat: self.lu.D shape: {self.lu.D.shape()}")
            D_diag = cytnx.linalg.Diag(self.lu.D)[:k]
            logger.debug(f"leftMat: D_diag: {D_diag}")
            D_diag_matrix = cytnx.linalg.Diag(D_diag)
            logger.debug(f"leftMat: D_diag_matrix: {D_diag_matrix}")
            self.cache['LD'] = self.lu.L[:,:k] @ D_diag_matrix
            logger.debug("leftMat: Computed new left matrix (L * D).")
        else:
            logger.debug("leftMat: Using cached left matrix.")
        logger.debug(f"leftMat: shape: {self.cache['LD'].shape()}")
        return self.cache['LD']

    def rightMat(self) -> Tensor:
        """
        Return the right matrix U.
        """
        k = self.rank()
        logger.debug(f"rightMat: shape: {self.lu.U.shape()}")
        return self.lu.U[:k, :]

    def availRows(self) -> list:
        """
        Return available row indices not yet in lu.Iset (cached).
        """
        if self.cache['I_avail'] is None:
            self.cache['I_avail'] = [i for i in range(self.n_rows) if i not in self.lu.Iset]
            logger.debug(f"availRows: Computed available rows: {self.cache['I_avail']}")
        else:
            logger.debug("availRows: Using cached available rows.")
        return self.cache['I_avail']

    def availCols(self) -> list:
        """
        Return available column indices not yet in lu.Jset (cached).
        """
        if self.cache['J_avail'] is None:
            self.cache['J_avail'] = [j for j in range(self.n_cols) if j not in self.lu.Jset]
            logger.debug(f"availCols: Computed available columns: {self.cache['J_avail']}")
        else:
            logger.debug("availCols: Using cached available columns.")
        return self.cache['J_avail']

    def rank(self) -> int:
        """
        Return the current rank of the decomposition.
        """
        current_rank = len(self.lu.Iset)
        logger.debug(f"rank: Current rank is {current_rank}")
        return current_rank

    def firstPivotValue(self) -> float:
        """
        Return the value of the first pivot.
        """
        if self.C is None or not self.lu.Iset:
            logger.debug("firstPivotValue: No pivot found, returning 1.0")
            return 1.0
        pivot_value = self.C[self.lu.Iset[0], 0].item()
        logger.debug(f"firstPivotValue: The first pivot value is {pivot_value}")
        return pivot_value

    def eval(self, i: int, j: int) -> float:
        """
        Compute the approximate value at position (i, j) using the cross formula Aapprox = C * P^-1 * R.
        """
        if self.C is None or self.R is None:
            logger.debug(f"eval: Either C or R is None, returning 0.0 for position ({i}, {j})")
            return 0.0
        val = (self.leftMat()[i, :] @ self.rightMat()[:, j]).item()
        logger.debug(f"eval: Evaluated value at ({i}, {j}) is {val}")
        return val

    def addPivot(self, i: int, j: int, A: cytnx.Tensor):
        """
        Update the cross data by adding a new pivot at position (i, j) of matrix A.
        
        Args:
            i (int): Row index of the pivot.
            j (int): Column index of the pivot.
            A (cytnx.Tensor): Input tensor to be indexed.
        """
        logger.debug(f"addPivot: Adding pivot at position ({i}, {j}).")
        A = ensure_2d(A, axis=0)  # Ensure A is 2D (assuming this function exists)
        self.addPivotRow(i, A)    # Add row to R (implementation not shown)
        self.addPivotCol(j, A)    # Add column to C (implementation not shown)
        self.lu.Iset.append(i)    # Append row index to Iset
        self.lu.Jset.append(j)    # Append column index to Jset
        I, J = self.lu.Iset, self.lu.Jset
        k = len(I)  # Number of pivots
        self.P = cytnx.zeros((k, k))  # Initialize P as a k×k zero tensor
        for ii in range(k):
            for jj in range(k):
                self.P[ii, jj] = A[I[ii], J[jj]]  # Assign elements individually
        logger.debug(f"addPivot: Updated pivot matrix P shape: {self.P.shape()}")
        

    def addPivotRow(self, i: int, A: Tensor):
        """
        Add row i of matrix A to R and update lu.
        """
        logger.debug(f"addPivotRow: Adding row {i} to R.")
        A = ensure_2d(A, axis=0)
        row = A[i, :]
        logger.debug(f"addPivotRow: Row shape: {row.shape()}")
        if self.R is None:
            self.R = row.reshape(1, self.n_cols)
            logger.debug(f"addPivotRow: Initialized R with shape {self.R.shape()}.")
        else:
            row_2d = row.reshape(1, self.n_cols)
            self.R = manual_vstack(self.R, row_2d)
            logger.debug(f"addPivotRow: Updated R, new shape: {self.R.shape()}.")
        self.lu.add_pivot_row(i, row)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

    def addPivotCol(self, j: int, A: Tensor):
        """
        Add column j of matrix A to C and update lu.
        """
        logger.debug(f"addPivotCol: Adding column {j} to C.")
        A = ensure_2d(A, axis=0)
        col = A[:, j]
        logger.debug(f"addPivotCol: Column shape: {col.shape()}")
        if self.C is None:
            self.C = col.reshape(self.n_rows, 1)
            logger.debug(f"addPivotCol: Initialized C with shape {self.C.shape()}.")
        else:
            col_2d = col.reshape(self.n_rows, 1)
            self.C = manual_hstack(self.C, col_2d)
            logger.debug(f"addPivotCol: Updated C, new shape: {self.C.shape()}.")
        self.lu.add_pivot_col(j, col)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

    def mat(self, A: Tensor) -> Tensor:
        """
        Return the full approximated matrix Aapprox = C * P^-1 * R.
        
        Args:
            A (Tensor): The original matrix, needed to compute P.
        
        Returns:
            Tensor: Approximated matrix with shape (n_rows, n_cols).
        """
        if self.C is None or self.R is None:
            logger.debug("mat: Either C or R is None, returning zero matrix.")
            return cytnx.zeros((self.n_rows, self.n_cols))
        
        Iset = [int(i) for i in self.lu.Iset]
        Jset = [int(j) for j in self.lu.Jset]
        k = len(Iset)
        M, N = A.shape()[0], A.shape()[1]

        P = cytnx.zeros((k, k))
        for i in range(k):
            for j in range(k):
                P[i, j] = A[Iset[i], Jset[j]]
        logger.debug(f"mat: P shape: {P.shape()}")
        
        # 進一步檢查矩陣尺寸
        assert self.C.shape() == (M, k), f"C shape mismatch: {self.C.shape()} vs ({M}, {k})"
        assert self.R.shape() == (k, N), f"R shape mismatch: {self.R.shape()} vs ({k}, {N})"
        assert self.P.shape() == (k, k), f"P shape mismatch: {self.P.shape()} vs ({k}, {k})"
        
        P_inv = cytnx.linalg.Inv(self.P)
        logger.debug(f"mat: P_inv shape: {P_inv.shape()}")
        cond_P = np.linalg.cond(P.numpy())
        logger.debug(f"mat: Condition number of P: {cond_P}")
        
        left = self.C @ P_inv
        logger.debug(f"mat: left shape: {left.shape()}")
        
        approx_matrix = left @ self.R
        logger.debug(f"mat: Approximated matrix shape: {approx_matrix.shape()}")
        
        return approx_matrix

    def eval_multiple(self, rows: list, cols: list) -> list:
        """
        Compute approximate values for multiple positions.
        """
        logger.debug(f"eval_multiple: Evaluating multiple entries for rows {rows} and cols {cols}.")
        if self.C is None or self.R is None or len(rows) != len(cols):
            logger.debug("eval_multiple: Mismatch in rows and columns or C/R is None. Returning zeros.")
            return [0.0] * len(rows)
        left = self.leftMat()[rows, :]
        right = self.rightMat()[:, cols]
        values = []
        for i in range(len(rows)):
            val = (left[i, :] @ right[:, i]).item()
            logger.debug(f"eval_multiple: Value at position ({rows[i]}, {cols[i]}) is {val}")
            values.append(val)
        return values

    def row(self, i: int) -> cytnx.Tensor:
        """
        Return the approximate i-th row.
        """
        logger.debug(f"row: Retrieving row {i}.")
        if self.C is None or self.R is None:
            logger.debug("row: Either C or R is None, returning zero row.")
            return cytnx.zeros(self.n_cols)
        row_val = self.leftMat()[i, :] @ self.rightMat()
        logger.debug(f"row: Retrieved row {i} with shape: {row_val.shape()}")
        return row_val

    def col(self, j: int) -> cytnx.Tensor:
        """
        Return the approximate j-th column.
        """
        logger.debug(f"col: Retrieving column {j}.")
        if self.C is None or self.R is None:
            logger.debug("col: Either C or R is None, returning zero column.")
            return cytnx.zeros(self.n_rows)
        col_val = self.leftMat() @ self.rightMat()[:, j]
        logger.debug(f"col: Retrieved column {j} with shape: {col_val.shape()}")
        return col_val

    def submat(self, rows: list, cols: list) -> cytnx.Tensor:
        """
        Return the approximate submatrix defined by the given rows and columns.
        """
        logger.debug(f"submat: Extracting submatrix with rows {rows} and columns {cols}.")
        if self.C is None or self.R is None:
            logger.debug("submat: Either C or R is None, returning zero matrix.")
            return cytnx.zeros((len(rows), len(cols)))
        left_sub = self.leftMat()[rows, :]
        right_sub = self.rightMat()[:, cols]
        submatrix = left_sub @ right_sub
        logger.debug(f"submat: Extracted submatrix shape: {submatrix.shape()}")
        return submatrix

    def setRows(self, C_new: cytnx.Tensor, P: list):
        """
        Increase the rows of the matrix according to C_new, reordering old rows per P.
        """
        logger.debug("setRows: Setting new rows and reordering according to permutation P.")
        self.n_rows = C_new.shape[0]
        self.lu.Iset = [P[i] for i in self.lu.Iset]
        L_reordered = self.lu.L[P, :]
        new_L = cytnx.zeros((self.n_rows, self.lu.L.shape[1]))
        new_L[:len(P), :] = L_reordered
        Pc = [i for i in range(self.n_rows) if i not in P]
        for k in range(self.lu.rank()):
            new_L[Pc, k] = C_new[Pc, k]
            for l in range(k):
                new_L[Pc, k] -= new_L[Pc, l] * (self.lu.U[l, self.lu.Jset[k]].item() * self.lu.D[l, l].item())
        self.lu.L = new_L
        self.C = C_new
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"setRows: New L matrix shape: {self.lu.L.shape()}, C shape: {self.C.shape()}")

    def setCols(self, R_new: cytnx.Tensor, Q: list):
        """
        Increase the columns of the matrix according to R_new, reordering old columns per Q.
        """
        logger.debug("setCols: Setting new columns and reordering according to permutation Q.")
        self.n_cols = R_new.shape[1]
        self.lu.Jset = [Q[j] for j in self.lu.Jset]
        U_reordered = self.lu.U[:, Q]
        new_U = cytnx.zeros((self.lu.U.shape()[0], self.n_cols))
        new_U[:, :len(Q)] = U_reordered
        Qc = [j for j in range(self.n_cols) if j not in Q]
        for k in range(self.lu.rank()):
            new_U[k, Qc] = R_new[k, Qc]
            for l in range(k):
                new_U[k, Qc] -= (self.lu.L[self.lu.Iset[k], l].item() * self.lu.D[l, l].item()) * new_U[l, Qc]
        self.lu.U = new_U
        self.R = R_new
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"setCols: New U matrix shape: {self.lu.U.shape()}, R shape: {self.R.shape()}")

    def estimate_error(self, A: cytnx.Tensor, sample_size: int = 100) -> float:
        """
        Estimate the approximation error using a random sample of positions.
        """
        logger.debug("estimate_error: Estimating approximation error using random sampling.")
        import random
        import numpy as np
        if self.C is None or self.R is None:
            logger.debug("estimate_error: Either C or R is None, error is infinite.")
            return float('inf')
        errors = []
        for _ in range(sample_size):
            i = random.randint(0, self.n_rows - 1)
            j = random.randint(0, self.n_cols - 1)
            approx_val = self.eval(i, j)
            true_val = A[i, j].item()
            error_val = (approx_val - true_val) ** 2
            errors.append(error_val)
            logger.debug(f"estimate_error: At ({i}, {j}), approx: {approx_val}, true: {true_val}, squared error: {error_val}")
        rmse = np.sqrt(np.mean(errors))
        logger.debug(f"estimate_error: Estimated RMSE is {rmse}")
        return rmse

# Test script
if __name__ == "__main__":
    logger.debug("Starting debug of CrossData with random matrix.")
    M, N = 3, 3
    np.random.seed(1)
    arr = np.random.rand(M, N)
    A = from_numpy(arr)
    logger.debug("Original matrix A:")
    logger.debug(f"{A}")
    logger.debug(f"Input A dtype: {A.dtype()}, shape: {A.shape()}")
    
    cross_data = CrossData(M, N)
    logger.debug("Adding pivots and tracking approximation error:")
    for _ in range(min(A.shape()[0], A.shape()[1])):
        residual = A - cross_data.mat(A)
        residual_np = residual.numpy()
        i, j = np.unravel_index(np.argmax(np.abs(residual_np)), residual_np.shape)
        cross_data.addPivot(i, j, A)
    
    final_approx_matrix = cross_data.mat(A)
    final_error_matrix = A - final_approx_matrix
    max_error = final_error_matrix.Abs().Max().item()

    logger.debug(f"Final maximum absolute error: {max_error:.6e}")
    logger.debug("Final approximated matrix:")
    logger.debug(f"{final_approx_matrix}")
    logger.debug("Original matrix:")
    logger.debug(f"{A}")
