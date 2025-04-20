# filename: crossdata.py (Corrected for NumPy index casting error)

import logging
import numpy as np
import cytnx
from cytnx import *
from AdaptiveLU import AdaptiveLU # Assuming AdaptiveLU.py is available
from typing import Optional
import numpy as np # Ensure numpy is imported if used for array conversion

# Initialize logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    """
    Ensure the input Tensor is at least 2D. If 1D, reshape it based on the axis.
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

# Using these custom functions now
def manual_hstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    """
    Manually stack two tensors horizontally using basic cytnx operations.
    """
    m, n = tensor1.shape()
    p = tensor2.shape()[1]
    if tensor1.shape()[0] != tensor2.shape()[0]:
        raise ValueError("Tensors must have the same number of rows for horizontal stacking.")
    result = cytnx.zeros((m, n + p), dtype=tensor1.dtype(), device=tensor1.device()) # Match dtype/device
    result[:, :n] = tensor1
    result[:, n:n + p] = tensor2
    logger.debug(f"manual_hstack: stacked tensors with shapes {tensor1.shape()} and {tensor2.shape()} to {result.shape()}")
    return result

def manual_vstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    """
    Manually stack two tensors vertically using basic cytnx operations.
    """
    m, n = tensor1.shape()
    p = tensor2.shape()[0]
    if tensor1.shape()[1] != tensor2.shape()[1]:
        raise ValueError("Tensors must have the same number of columns for vertical stacking.")
    result = cytnx.zeros((m + p, n), dtype=tensor1.dtype(), device=tensor1.device()) # Match dtype/device
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
        self.C = None
        self.R = None
        self.lu = AdaptiveLU(n_rows, n_cols)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"Initialized CrossData with {n_rows} rows and {n_cols} columns.")

    def pivotMat(self) -> Optional[Tensor]:
        """
        Return the pivot matrix A[Iset, Jset] (extracted from C or R).
        """
        if self.C is None or not isinstance(self.lu.Iset, (list, tuple)) or not self.lu.Iset or not isinstance(self.lu.Jset, (list, tuple)) or not self.lu.Jset:
            logger.debug("pivotMat: No pivot set found or C/R is None, returning None.")
            return None
        Iset_indices = np.array(self.lu.Iset, dtype=np.uint32)
        try:
            if np.any(Iset_indices >= self.C.shape()[0]):
                 logger.error(f"pivotMat: Iset indices {self.lu.Iset} out of bounds for C shape {self.C.shape()}")
                 return None
            pivot_matrix = self.C[Iset_indices, :]
            logger.debug(f"pivotMat: Pivot matrix shape: {pivot_matrix.shape()}")
            return pivot_matrix
        except Exception as e:
             logger.error(f"pivotMat: Unexpected error: {e}")
             return None

    def leftMat(self) -> Optional[Tensor]:
        """ Return the left matrix L * D (cached). """
        k = self.rank()
        if k == 0: return None
        if self.cache.get('LD') is None:
            try:
                if self.lu.D is None or self.lu.L is None: return None
                if self.lu.D.shape()[0] < k or self.lu.L.shape()[1] < k: return None
                D_diag = cytnx.linalg.Diag(self.lu.D[:k])
                L_k = self.lu.L[:, :k]
                if L_k.shape()[1] != k or D_diag.shape()[0] != k or D_diag.shape()[1] != k: return None
                self.cache['LD'] = L_k @ D_diag
                logger.debug("leftMat: Computed new left matrix (L * D).")
            except Exception as e:
                 logger.error(f"leftMat: Error computing L*D: {e}")
                 self.cache['LD'] = None
                 return None
        if self.cache.get('LD') is not None: logger.debug(f"leftMat: Shape: {self.cache['LD'].shape()}")
        return self.cache.get('LD')

    def rightMat(self) -> Optional[Tensor]:
        """ Return the right matrix U. """
        k = self.rank()
        if k == 0 or self.lu.U is None: return None
        if self.lu.U.shape()[0] < k: return None
        U_k = self.lu.U[:k, :]
        if U_k.shape()[0] != k: return None
        logger.debug(f"rightMat: Shape: {U_k.shape()}")
        return U_k

    def availRows(self) -> list:
        """ Return available row indices not yet in lu.Iset (cached). """
        if self.cache.get('I_avail') is None:
            current_Iset = set(self.lu.Iset) if isinstance(self.lu.Iset, (list, tuple)) else set()
            self.cache['I_avail'] = [i for i in range(self.n_rows) if i not in current_Iset]
            logger.debug(f"availRows: Computed available rows: {len(self.cache['I_avail'])}")
        return self.cache.get('I_avail', [])

    def availCols(self) -> list:
        """ Return available column indices not yet in lu.Jset (cached). """
        if self.cache.get('J_avail') is None:
            current_Jset = set(self.lu.Jset) if isinstance(self.lu.Jset, (list, tuple)) else set()
            self.cache['J_avail'] = [j for j in range(self.n_cols) if j not in current_Jset]
            logger.debug(f"availCols: Computed available columns: {len(self.cache['J_avail'])}")
        return self.cache.get('J_avail', [])

    def rank(self) -> int:
        """ Return the current rank of the decomposition. """
        current_rank = len(self.lu.Iset) if isinstance(self.lu.Iset, (list, tuple)) else 0
        return current_rank

    def firstPivotValue(self) -> float:
        """ Return the value of the first pivot. """
        if self.C is None or not isinstance(self.lu.Iset, (list, tuple)) or not self.lu.Iset or not isinstance(self.lu.Jset, (list, tuple)) or not self.lu.Jset:
            logger.debug("firstPivotValue: No valid pivot set found or C is None, returning 1.0")
            return 1.0
        try:
             if self.lu.Iset[0] >= self.C.shape()[0]:
                  logger.error(f"firstPivotValue: First pivot row index {self.lu.Iset[0]} out of bounds for C shape {self.C.shape()}.")
                  return 1.0
             pivot_value = self.C[self.lu.Iset[0], 0].item()
             logger.debug(f"firstPivotValue: First pivot value is {pivot_value}")
             return pivot_value
        except Exception as e:
             logger.error(f"firstPivotValue: Error getting pivot value: {e}")
             return 1.0

    def eval(self, i: int, j: int) -> float:
        """ Compute the approximate value at position (i, j) using A_approx = L @ D @ U. """
        if self.rank() == 0: return 0.0
        LD = self.leftMat()
        U = self.rightMat()
        if LD is None or U is None: return 0.0
        try:
             if i >= LD.shape()[0] or j >= U.shape()[1]: return 0.0
             LD_row_i = LD[i, :]
             U_col_j = U[:, j]
             if LD_row_i.shape()[0] != U_col_j.shape()[0]: return 0.0
             val = cytnx.linalg.Dot(LD_row_i, U_col_j).item()
             return val
        except Exception as e:
             logger.error(f"eval: Error during evaluation at ({i},{j}): {e}")
             return 0.0

    # --- Restored Old addPivotRow/Col with manual stacking ---
    def addPivotRow(self, i: int, A: Tensor):
        """ Add row i of matrix A to R and update lu. """
        logger.debug(f"addPivotRow: Adding row {i} using full matrix A.")
        A = ensure_2d(A, axis=0)
        try: row = A[i, :]
        except IndexError:
            logger.error(f"addPivotRow: Row index {i} out of bounds for matrix A with shape {A.shape()}.")
            return
        logger.debug(f"[addPivotRow] Extracted row {i} from A, shape={row.shape()}")
        row_2d = row.reshape(1, self.n_cols)
        if self.R is None:
            self.R = row_2d.clone()
            logger.debug(f"addPivotRow: Initialized R with shape {self.R.shape()}.")
        else:
            self.R = manual_vstack(self.R, row_2d) # Use custom vstack
            logger.debug(f"addPivotRow: Updated R, new shape: {self.R.shape()}.")
        self.lu.add_pivot_row(i, row)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

    def addPivotCol(self, j: int, A: Tensor):
        """ Add column j of matrix A to C and update lu. """
        logger.debug(f"addPivotCol: Adding column {j} using full matrix A.")
        A = ensure_2d(A, axis=0)
        try: col = A[:, j]
        except IndexError:
             logger.error(f"addPivotCol: Column index {j} out of bounds for matrix A with shape {A.shape()}.")
             return
        logger.debug(f"[addPivotCol] Extracted col {j} from A, shape={col.shape()}")
        col_2d = col.reshape(self.n_rows, 1)
        if self.C is None:
            self.C = col_2d.clone()
            logger.debug(f"addPivotCol: Initialized C with shape {self.C.shape()}.")
        else:
            self.C = manual_hstack(self.C, col_2d) # Use custom hstack
            logger.debug(f"addPivotCol: Updated C, new shape: {self.C.shape()}.")
        self.lu.add_pivot_col(j, col)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

    # --- Restored addPivot convenience method ---
    def addPivot(self, i: int, j: int, A: cytnx.Tensor):
        """ Update the cross data by adding a new pivot at position (i, j) of matrix A. """
        logger.debug(f"addPivot: Adding pivot at position ({i}, {j}).")
        self.addPivotRow(i, A)
        self.addPivotCol(j, A)

    # --- Other methods ---
    def mat(self) -> Tensor:
        """ Reconstruct the approximated matrix A_approx = L @ D @ U """
        logger.debug("mat: Reconstructing matrix using lu object.")
        if self.rank() == 0:
             return cytnx.zeros((self.n_rows, self.n_cols))
        try:
            Aapprox = self.lu.reconstruct()
            if Aapprox.shape() != [self.n_rows, self.n_cols]:
                 logger.warning(f"mat: Reconstructed shape {Aapprox.shape()} differs from expected ({self.n_rows},{self.n_cols}).")
                 return cytnx.zeros((self.n_rows, self.n_cols))
            logger.debug(f"mat: Approximated matrix shape: {Aapprox.shape()}")
            return Aapprox
        except Exception as e:
            logger.error(f"mat: Error during lu.reconstruct: {e}")
            return cytnx.zeros((self.n_rows, self.n_cols))

# Test script
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.debug("Starting debug of CrossData with random matrix.")
    M, N = 5, 4
    np.random.seed(0)
    arr = np.random.rand(M, N) * 10
    arr[1, 2] += 50
    A = from_numpy(arr)
    logger.info("Original matrix A:")
    logger.info(f"Matrix content:\n{A}")

    cross_data = CrossData(M, N)
    logger.info("Adding pivots and tracking approximation error:")
    MAX_RANK_TO_ADD = min(M, N)
    for k in range(MAX_RANK_TO_ADD):
        logger.info(f"--- Iteration {k+1} ---")
        Aapprox = cross_data.mat()
        residual = A - Aapprox

        current_Iset = set(cross_data.lu.Iset) if isinstance(cross_data.lu.Iset, (list, tuple)) else set()
        current_Jset = set(cross_data.lu.Jset) if isinstance(cross_data.lu.Jset, (list, tuple)) else set()

        mask = cytnx.ones(residual.shape(), dtype=residual.dtype(), device=residual.device())

        # --- Fix for Masking using Iteration ---
        if current_Iset:
            iset_indices = list(current_Iset) # Use list directly
            for idx in iset_indices:
                if 0 <= idx < mask.shape()[0]: # Check bounds
                    mask[idx, :] = 0.0 # Assign 0.0 (float)
        if current_Jset:
            jset_indices = list(current_Jset) # Use list directly
            for idx in jset_indices:
                 if 0 <= idx < mask.shape()[1]: # Check bounds
                    mask[:, idx] = 0.0 # Assign 0.0 (float)
        # --- End Fix ---

        masked_residual = residual * mask
        abs_residual = cytnx.linalg.Abs(masked_residual)
        if abs_residual.shape()[0] == 0 or abs_residual.shape()[1] == 0: break
        flat_residual = abs_residual.reshape(-1)
        if flat_residual.shape()[0] == 0: break

        max_val = -1.0
        max_idx = -1
        if flat_residual.shape()[0] > 0:
             temp_max_val = -1.0
             temp_max_idx = -1
             for idx in range(flat_residual.shape()[0]):
                 val = flat_residual[idx].item()
                 if val > temp_max_val:
                      temp_max_val = val
                      temp_max_idx = idx
             max_val = temp_max_val
             max_idx = temp_max_idx

        if max_idx == -1 or max_val < 1e-14:
            logger.info(f"Max residual value {max_val:.2e} below tolerance or not found, stopping.")
            break

        num_cols_res = abs_residual.shape()[1]
        if num_cols_res == 0: break
        i = max_idx // num_cols_res
        j = max_idx % num_cols_res

        logger.info(f"Found pivot at ({i}, {j}) with residual value {residual[i,j].item():.4f} (max abs masked residual: {max_val:.4f})")
        cross_data.addPivot(i, j, A)
        logger.info(f"Rank after adding: {cross_data.rank()}")

    logger.info("--- Final Results ---")
    final_approx_matrix = cross_data.mat()
    final_error_matrix = A - final_approx_matrix
    max_error = 0.0
    if final_error_matrix.shape()[0] > 0 and final_error_matrix.shape()[1] > 0 :
         max_error = final_error_matrix.Abs().Max().item()

    logger.info(f"Final Rank: {cross_data.rank()}")
    logger.info(f"Final maximum absolute error: {max_error:.6e}")
    logger.info("Final approximated matrix:")
    logger.info(f"Matrix content:\n{final_approx_matrix}")
    logger.info("Original matrix:")
    logger.info(f"Matrix content:\n{A}")