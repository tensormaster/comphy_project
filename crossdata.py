import logging
import numpy as np
import cytnx
from cytnx import *
from AdaptiveLU import AdaptiveLU
import logging
import numpy as np
import cytnx
# Initialize logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    """
    Ensure the input Tensor is at least 2D. If 1D, reshape it based on the axis.
    
    Args:
        t (Tensor): Input tensor.
        axis (int): Axis to reshape along (0 for rows, 1 for columns).
    
    Returns:
        Tensor: 2D tensor.
    """
    shape = t.shape()
    if len(shape) == 1:
        n = shape[0]
        if axis == 0:
            t2 = t.reshape(n, 1)
        else:
            t2 = t.reshape(1, n)
        logger.debug(f"ensure_2d: Reshaped tensor from {shape} to {t2.shape()}")
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
    logger.debug(f"manual_hstack: Result tensor content:\n{result}")
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
    logger.debug(f"manual_vstack: Result tensor content:\n{result}")
    return result

class CrossData:
    def __init__(self, n_rows: int, n_cols: int):
        """
        Initialize an empty CrossData instance.
        
        Args:
            n_rows (int): Number of rows in the matrix.
            n_cols (int): Number of columns in the matrix.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.C = None  # Column submatrix
        self.R = None  # Row submatrix
        self.lu = AdaptiveLU(n_rows, n_cols)  # Placeholder for AdaptiveLU
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"Initialized CrossData with {n_rows} rows and {n_cols} columns.")

    def pivotMat(self) -> Tensor:
        """
        Return the pivot matrix (rows of C corresponding to lu.Iset).
        
        Returns:
            Tensor: Pivot matrix, or None if no pivots exist.
        """
        if self.C is None or not self.lu.Iset:
            logger.debug("pivotMat: No pivot set found, returning None.")
            return None
        Iset_uvec = np.array(self.lu.Iset, dtype=np.uint32)
        pivot_matrix = self.C[Iset_uvec, :]
        logger.debug(f"pivotMat: Pivot matrix shape: {pivot_matrix.shape()}")
        logger.debug(f"pivotMat: Pivot matrix content:\n{pivot_matrix}")
        return pivot_matrix

    def leftMat(self) -> Tensor:
        """
        Return the left matrix L * D (cached).
        
        Returns:
            Tensor: Left matrix L * D.
        """
        k = self.rank()
        if self.cache['LD'] is None:
            D_diag = cytnx.linalg.Diag(self.lu.D[:k])
            self.cache['LD'] = self.lu.L[:, :k] @ D_diag
            logger.debug("leftMat: Computed new left matrix (L * D).")
        else:
            logger.debug("leftMat: Using cached left matrix.")
        logger.debug(f"leftMat: Shape: {self.cache['LD'].shape()}")
        logger.debug(f"leftMat: Matrix content:\n{self.cache['LD']}")
        return self.cache['LD']

    def rightMat(self) -> Tensor:
        """
        Return the right matrix U.
        
        Returns:
            Tensor: Right matrix U.
        """
        k = self.rank()
        logger.debug(f"rightMat: Shape: {self.lu.U.shape()}")
        logger.debug(f"rightMat: Matrix content:\n{self.lu.U[:k, :]}")
        return self.lu.U[:k, :]

    def availRows(self) -> list:
        """
        Return available row indices not yet in lu.Iset (cached).
        
        Returns:
            list: List of available row indices.
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
        
        Returns:
            list: List of available column indices.
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
        
        Returns:
            int: Current rank.
        """
        current_rank = len(self.lu.Iset)
        logger.debug(f"rank: Current rank is {current_rank}")
        return current_rank

    def firstPivotValue(self) -> float:
        """
        Return the value of the first pivot.
        
        Returns:
            float: First pivot value, or 1.0 if no pivots exist.
        """
        if self.C is None or not self.lu.Iset:
            logger.debug("firstPivotValue: No pivot found, returning 1.0")
            return 1.0
        pivot_value = self.C[self.lu.Iset[0], 0].item()
        logger.debug(f"firstPivotValue: First pivot value is {pivot_value}")
        return pivot_value

    def eval(self, i: int, j: int) -> float:
        """
        Compute the approximate value at position (i, j).
        
        Args:
            i (int): Row index.
            j (int): Column index.
        
        Returns:
            float: Approximate value at (i, j).
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
            A (Tensor): Input matrix.
        """
        logger.debug(f"addPivot: Adding pivot at position ({i}, {j}).")
        A = ensure_2d(A, axis=0)
        self.addPivotRow(i, A)
        self.addPivotCol(j, A)
        # Removed P and P_inv computation as they are not needed

    def addPivotRow(self, i: int, A: Tensor):
        """
        Add row i of matrix A to R and update lu.
        
        Args:
            i (int): Row index to add.
            A (Tensor): Input matrix.
        """
        logger.debug(f"addPivotRow: Adding row {i} to R.")
        A = ensure_2d(A, axis=0)
        row = A[i, :]
        logger.debug(f"[addPivotRow] Fetching row {i} from A, shape={A.shape()}")
        if self.R is None:
            self.R = row.reshape(1, self.n_cols)
            logger.debug(f"addPivotRow: Initialized R with shape {self.R.shape()}.")
        else:
            row_2d = row.reshape(1, self.n_cols)
            self.R = manual_vstack(self.R, row_2d)
            logger.debug(f"addPivotRow: Updated R, new shape: {self.R.shape()}.")
        self.lu.add_pivot_row(i, row)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"addPivotRow: Current R content:\n{self.R}")

    def addPivotCol(self, j: int, A: Tensor):
        """
        Add column j of matrix A to C and update lu.
        
        Args:
            j (int): Column index to add.
            A (Tensor): Input matrix.
        """
        logger.debug(f"addPivotCol: Adding column {j} to C.")
        A = ensure_2d(A, axis=0)
        col = A[:, j]
        if self.C is None:
            self.C = col.reshape(self.n_rows, 1)
            logger.debug(f"addPivotCol: Initialized C with shape {self.C.shape()}.")
        else:
            col_2d = col.reshape(self.n_rows, 1)
            self.C = manual_hstack(self.C, col_2d)
            logger.debug(f"addPivotCol: Updated C, new shape: {self.C.shape()}.")
        self.lu.add_pivot_col(j, col)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"addPivotCol: Current C content:\n{self.C}")

    def mat(self) -> Tensor:
        Aapprox = self.lu.reconstruct()
        logger.debug(f"mat: Approximated matrix shape: {Aapprox.shape()}")
        logger.debug(f"mat: Approximated matrix content:\n{Aapprox}")
        return Aapprox

    def setRows(self, C_new: cytnx.Tensor, P: list):
        """
        Increase the rows of the matrix according to C_new, reordering old rows per P.
        
        Args:
            C_new (Tensor): New column submatrix.
            P (list): Permutation of old row indices.
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
                new_L[Pc, k] -= new_L[Pc, l] * (self.lu.U[l, self.lu.Jset[k]].item() * self.lu.D[l].item())
        self.lu.L = new_L
        self.C = C_new
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"setRows: Updated C content:\n{self.C}")

    def setCols(self, R_new: cytnx.Tensor, Q: list):
        """
        Increase the columns of the matrix according to R_new, reordering old columns per Q.
        
        Args:
            R_new (Tensor): New row submatrix.
            Q (list): Permutation of old column indices.
        """
        logger.debug("setCols: Setting new columns and reordering according to permutation Q.")
        self.n_cols = R_new.shape[1]
        self.lu.Jset = [Q[j] for j in self.lu.Jset]
        U_reordered = self.lu.U[:, Q]
        new_U = cytnx.zeros((self.lu.U.shape[0], self.n_cols))
        new_U[:, :len(Q)] = U_reordered
        Qc = [j for j in range(self.n_cols) if j not in Q]
        for k in range(self.lu.rank()):
            new_U[k, Qc] = R_new[k, Qc]
            for l in range(k):
                new_U[k, Qc] -= (self.lu.L[self.lu.Iset[k], l].item() * self.lu.D[l].item()) * new_U[l, Qc]
        self.lu.U = new_U
        self.R = R_new
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"setCols: Updated R content:\n{self.R}")

    def reconstruct(self):
        if self.rank > 0:
            K = self.rank
            D_diag = cytnx.linalg.Diag(self.D[:K])  # Use only the first K elements
            return self.L[:, :K] @ D_diag @ self.U[:K, :]
        else:
            return cytnx.zeros((self.n_rows, self.n_cols), dtype=Type.Double, device=Device.cpu)
        
# Test script
if __name__ == "__main__":
    logger.debug("Starting debug of CrossData with random matrix.")
    M, N = 3, 3
    np.random.seed(0)
    arr = np.random.rand(M, N)
    A = from_numpy(arr)
    logger.debug("Original matrix A:")
    logger.debug(f"Matrix content:\n{A}")
    logger.debug(f"Input A dtype: {A.dtype()}, shape: {A.shape()}")

    cross_data = CrossData(M, N)
    logger.debug("Adding pivots and tracking approximation error:")
    for _ in range(min(A.shape()[0], A.shape()[1])):
        # Compute current approximation and residual
        Aapprox = cross_data.mat()  # Uses self.lu.reconstruct() internally
        residual = A - Aapprox

        # Mask already selected rows and columns
        for i_used in cross_data.lu.Iset:
            residual[i_used, :] = 0
        for j_used in cross_data.lu.Jset:
            residual[:, j_used] = 0

        # Find the maximum absolute value in the residual
        abs_residual = cytnx.linalg.Abs(residual)
        flat_residual = abs_residual.reshape(-1)  # Flatten to 1D
        max_idx = 0
        max_val = flat_residual[0].item()
        for idx in range(1, flat_residual.shape()[0]):
            val = flat_residual[idx].item()
            if val > max_val:
                max_val = val
                max_idx = idx

        # Convert 1D index back to 2D coordinates
        N = abs_residual.shape()[1]  # Number of columns
        i = max_idx // N
        j = max_idx % N

        # Add the pivot
        cross_data.addPivot(i, j, A)

    final_approx_matrix = cross_data.mat()
    final_error_matrix = A - final_approx_matrix
    max_error = final_error_matrix.Abs().Max().item()

    logger.debug(f"Final maximum absolute error: {max_error:.6e}")
    logger.debug("Final approximated matrix:")
    logger.debug(f"Matrix content:\n{final_approx_matrix}")
    logger.debug("Original matrix:")
    logger.debug(f"Matrix content:\n{A}")
