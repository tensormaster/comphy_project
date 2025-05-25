# filename: crossdata.py (Further Corrected - Attempt 5: Using manual_vstack)

import logging
import numpy as np
import cytnx
from cytnx import Tensor
# Make sure IMatrix is imported if it's defined in another file
from matrix_interface import IMatrix
from AdaptiveLU import AdaptiveLU
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    shape = t.shape()
    if len(shape) == 1:
        n = shape[0]
        if axis == 0:
            t2 = t.reshape(n, 1)
        else:
            t2 = t.reshape(1, n)
        # logger.debug(f"ensure_2d: Reshaped tensor from {shape} to {t2.shape()}") # Optional: less verbose
        return t2
    return t

def manual_hstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    # ... (implementation as before) ...
    m, n = tensor1.shape()
    p = tensor2.shape()[1]
    if tensor1.shape()[0] != tensor2.shape()[0]:
        raise ValueError("Tensors must have the same number of rows for horizontal stacking.")
    result = cytnx.zeros((m, n + p), dtype=tensor1.dtype(), device=tensor1.device())
    result[:, :n] = tensor1
    result[:, n:n + p] = tensor2
    return result

def manual_vstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    # ... (implementation as before) ...
    m, n = tensor1.shape()
    p = tensor2.shape()[0]
    if tensor1.shape()[1] != tensor2.shape()[1]:
        raise ValueError("Tensors must have the same number of columns for vertical stacking.")
    result = cytnx.zeros((m + p, n), dtype=tensor1.dtype(), device=tensor1.device())
    result[:m, :] = tensor1
    result[m:m + p, :] = tensor2
    return result

class CrossData:
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.C: Optional[Tensor] = None
        self.R: Optional[Tensor] = None
        self.lu = AdaptiveLU(n_rows, n_cols)
        self.cache: Dict[str, Any] = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"Initialized CrossData with {n_rows} rows and {n_cols} columns.")

    def pivotMat(self) -> Optional[Tensor]:
        if self.C is None or not isinstance(self.lu.Iset, (list, tuple)) or not self.lu.Iset:
            logger.debug("pivotMat: C is None or lu.Iset is empty. Returning None.")
            return None

        current_rank = self.rank()
        if current_rank == 0:
            logger.debug("pivotMat: Current rank is 0. Returning None.")
            return None

        try:
            iset_np = np.array(self.lu.Iset, dtype=np.int64)
            if iset_np.size == 0:
                logger.warning("pivotMat: Iset is effectively empty. Returning None.")
                return None

            # Check bounds before proceeding with slicing/taking
            if np.max(iset_np) >= self.C.shape()[0]:
                logger.error(f"pivotMat: Max Iset index {np.max(iset_np)} out of bounds for C shape {self.C.shape()}.")
                return None

            # --- MODIFICATION START (Attempt 5: Using manual_vstack for concatenation) ---
            pivot_matrix_candidate: Optional[Tensor] = None
            if iset_np.size > 0:
                # Get the first row
                first_row = self.C[iset_np[0], :]
                pivot_matrix_candidate = ensure_2d(first_row, axis=1) # Ensure (1, N)

                # Manually vstack subsequent rows
                for i in range(1, iset_np.size):
                    next_row = self.C[iset_np[i], :]
                    pivot_matrix_candidate = manual_vstack(pivot_matrix_candidate, ensure_2d(next_row, axis=1))
            else:
                # If no rows selected, return an empty matrix with correct column dimension
                # This case is already covered by iset_np.size == 0 check above, but for completeness.
                pivot_matrix_candidate = cytnx.zeros((0, self.C.shape()[1]), dtype=self.C.dtype(), device=self.C.device())

            if pivot_matrix_candidate is None:
                logger.error("pivotMat: Failed to create pivot_matrix_candidate due to unexpected None state.")
                return None
            # --- MODIFICATION END ---

            # --- 修正開始：確保 pivot_matrix 是 2D 且形狀正確 ---
            if current_rank == 1:
                if pivot_matrix_candidate.shape() == [1]:
                    pivot_matrix = pivot_matrix_candidate.reshape(1, 1)
                elif pivot_matrix_candidate.shape() == [current_rank, current_rank]:
                    pivot_matrix = pivot_matrix_candidate
                else:
                    logger.warning(f"pivotMat: Rank 1 pivot_matrix_candidate shape is {pivot_matrix_candidate.shape()}, expected [1] or [1,1]. Returning None.")
                    return None
            elif pivot_matrix_candidate.rank() == 2 and \
                 pivot_matrix_candidate.shape()[0] == current_rank and \
                 pivot_matrix_candidate.shape()[1] == current_rank:
                pivot_matrix = pivot_matrix_candidate
            else:
                logger.warning(f"pivotMat: Resulting pivot_matrix_candidate shape {pivot_matrix_candidate.shape()} "
                               f"is not 2D or not rank x rank ({current_rank}x{current_rank}). Returning None.")
                return None
            # --- 修正結束 ---

            logger.debug(f"pivotMat: Successfully created pivot matrix of shape: {pivot_matrix.shape()}")
            return pivot_matrix
        except Exception as e:
            logger.error(f"pivotMat: Error: {e}", exc_info=True)
            return None

    # ... (rest of the CrossData class and test script remains the same) ...
    def rank(self) -> int:
        current_rank = len(self.lu.Iset) if isinstance(self.lu.Iset, (list, tuple)) else 0
        return current_rank

    # ... (leftMat, rightMat, availRows, availCols, firstPivotValue, eval - keep as is for now) ...

    def addPivotRow(self, i: int, A: Union[IMatrix, Tensor]): # A can be IMatrix or Tensor
        """
        Processes row 'i' from matrix A to update self.R and self.lu.
        'i' is an integer index valid for A.
        """
        logger.debug(f"addPivotRow: Request to process row {i} from matrix A (type: {type(A)}).")
        row_tensor_extracted: Optional[Tensor] = None

        if hasattr(A, 'get_row_as_tensor') and callable(A.get_row_as_tensor): # Check if it's our IMatrix
            try:
                row_tensor_extracted = A.get_row_as_tensor(i)
            except IndexError:
                logger.error(f"addPivotRow (IMatrix): Row index {i} out of bounds."); return
            except Exception as e:
                logger.error(f"addPivotRow (IMatrix): Error calling get_row_as_tensor: {e}"); return
        elif isinstance(A, Tensor): # If A is already a cytnx.Tensor
            try:
                if not (0 <= i < A.shape()[0]):
                    logger.error(f"addPivotRow (Tensor): Row index {i} out of bounds for shape {A.shape()}."); return
                row_tensor_extracted = A[i, :] # Get full row
            except Exception as e:
                logger.error(f"addPivotRow (Tensor): Error slicing tensor: {e}"); return
        else:
            logger.error(f"addPivotRow: Unsupported type for A: {type(A)}"); return

        if row_tensor_extracted is None: return

        logger.debug(f"[addPivotRow] Extracted row tensor for index {i}, shape={row_tensor_extracted.shape()}")

        # Ensure it's 2D (1, num_cols_A) for vstacking into R
        # Assuming row_tensor_extracted is 1D
        row_2d = ensure_2d(row_tensor_extracted.clone(), axis=1) # axis=1 for (1, N)

        if self.R is None:
            self.R = row_2d
        else:
            if self.R.shape()[1] != row_2d.shape()[1]:
                logger.error(f"addPivotRow: Mismatch in columns for vstack. R_cols: {self.R.shape()[1]}, row_cols: {row_2d.shape()[1]}"); return
            self.R = manual_vstack(self.R, row_2d)
        logger.debug(f"addPivotRow: Updated R, new shape: {self.R.shape()}.")

        self.lu.add_pivot_row(i, row_tensor_extracted)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

    def addPivotCol(self, j: int, A: Union[IMatrix, Tensor]): # A can be IMatrix or Tensor
        """
        Processes column 'j' from matrix A to update self.C and self.lu.
        'j' is an integer index valid for A.
        """
        logger.debug(f"addPivotCol: Request to process column {j} from matrix A (type: {type(A)}).")
        col_tensor_extracted: Optional[Tensor] = None

        if hasattr(A, 'get_col_as_tensor') and callable(A.get_col_as_tensor): # Check if it's our IMatrix
            try:
                col_tensor_extracted = A.get_col_as_tensor(j)
            except IndexError:
                logger.error(f"addPivotCol (IMatrix): Col index {j} out of bounds."); return
            except Exception as e:
                logger.error(f"addPivotCol (IMatrix): Error calling get_col_as_tensor: {e}"); return
        elif isinstance(A, Tensor): # If A is already a cytnx.Tensor
            try:
                if not (0 <= j < A.shape()[1]):
                    logger.error(f"addPivotCol (Tensor): Col index {j} out of bounds for shape {A.shape()}."); return
                col_tensor_extracted = A[:, j] # Get full col
            except Exception as e:
                logger.error(f"addPivotCol (Tensor): Error slicing tensor: {e}"); return
        else:
            logger.error(f"addPivotCol: Unsupported type for A: {type(A)}"); return

        if col_tensor_extracted is None: return

        logger.debug(f"[addPivotCol] Extracted col tensor for index {j}, shape={col_tensor_extracted.shape()}")

        # Ensure it's 2D (num_rows_A, 1) for hstacking into C
        # Assuming col_tensor_extracted is 1D
        col_2d = ensure_2d(col_tensor_extracted.clone(), axis=0) # axis=0 for (N, 1)

        if self.C is None:
            self.C = col_2d
        else:
            if self.C.shape()[0] != col_2d.shape()[0]:
                logger.error(f"addPivotCol: Mismatch in rows for hstack. C_cols: {self.C.shape()[0]}, col_rows: {col_2d.shape()[0]}"); return
            self.C = manual_hstack(self.C, col_2d)
        logger.debug(f"addPivotCol: Updated C, new shape: {self.C.shape()}.")

        self.lu.add_pivot_col(j, col_tensor_extracted)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

    def addPivot(self, i: int, j: int, A: Union[IMatrix, Tensor]): # A can be IMatrix or Tensor
        logger.debug(f"addPivot: Adding pivot at ({i}, {j}) from A (type: {type(A)}).")
        # These calls will now correctly dispatch based on A's type (once IMatrix has get_... methods)
        self.addPivotRow(i, A)
        self.addPivotCol(j, A)

    def mat(self) -> Optional[Tensor]: # Can return None if rank is 0 or error
        logger.debug("mat: Reconstructing matrix.")
        if self.rank() == 0:
            logger.debug("mat: Rank is 0, returning zero matrix.")
            # Ensure dtype and device are consistent, e.g., taken from lu or init params
            # For simplicity, using default for now if not otherwise available
            default_dtype = cytnx.Type.Double
            default_device = cytnx.Device.cpu
            if self.lu and hasattr(self.lu, 'L') and self.lu.L is not None : # Try to get from LU if possible
                default_dtype = self.lu.L.dtype()
                default_device = self.lu.L.device()
            return cytnx.zeros((self.n_rows, self.n_cols), dtype=default_dtype, device=default_device)
        try:
            Aapprox = self.lu.reconstruct()
            if Aapprox.shape() != [self.n_rows, self.n_cols]:
                logger.warning(f"mat: Reconstructed shape {Aapprox.shape()} differs from CrossData "
                               f"expected ({self.n_rows},{self.n_cols}). This might indicate an issue in AdaptiveLU.reconstruct().")
                # Depending on desired behavior, you might pad or raise an error.
                # For now, returning as is, but this is a potential source of downstream errors.
            return Aapprox
        except Exception as e:
            logger.error(f"mat: Error during lu.reconstruct: {e}", exc_info=True)
            return None # Return None on error to be handled by caller


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

# --- Test script (if __name__ == "__main__") from your file ---
# This test script will now call the modified addPivotRow/Col which can handle
# A being a cytnx.Tensor (as it is in this test script).
# When TensorCI1 calls addPivot with a MatLazyIndex, the hasattr checks will
# trigger the (yet to be implemented in MatLazyIndex) get_row/col_as_tensor methods.
if __name__ == "__main__":
    # (Your test script content from the previous prompt, it should now work with this CrossData)
    # Ensure that matrix_interface.py (with IMatrix) is in the Python path or same directory
    # For the test to run standalone without matrix_interface.IMatrix fully defined yet,
    # you might need a placeholder IMatrix for type hinting if strict checking is on,
    # or rely on the hasattr checks.
    # To make this script runnable standalone before matrix_interface.py is fixed,
    # you could add a dummy IMatrix class here for the type hint if needed:
    # class IMatrix: pass

    logger.setLevel(logging.INFO)
    logger.debug("Starting debug of CrossData with random matrix.")
    M, N = 5, 4
    np.random.seed(0)
    arr = np.random.rand(M, N) * 10
    arr[1, 2] += 50
    A_tensor = cytnx.from_numpy(arr) # A_tensor is a cytnx.Tensor
    logger.info("Original matrix A:")
    logger.info(f"{A_tensor}") # Cytnx tensors are directly printable

    cross_data = CrossData(M, N)

    # --- Diagnostic prints from previous suggestion (can be kept or removed) ---
    logger.info(f"cross_data object type: {type(cross_data)}")
    if hasattr(cross_data, 'rank') and callable(cross_data.rank):
        logger.info(f"cross_data HAS attribute 'rank' and it's callable.")
        try: logger.info(f"Direct call to cross_data.rank(): {cross_data.rank()}")
        except Exception as e_rank_direct_call: logger.error(f"Error calling rank() directly: {e_rank_direct_call}")
    else: logger.error("cross_data DOES NOT HAVE 'rank' or it's not callable.")
    # ---

    logger.info("Adding pivots and tracking approximation error:")
    MAX_RANK_TO_ADD = min(M, N)
    for k in range(MAX_RANK_TO_ADD):
        logger.info(f"--- Iteration {k+1} ---")
        Aapprox = cross_data.mat()
        if Aapprox is None:
            logger.error("Aapprox is None, cannot compute residual. Stopping.")
            break

        residual = A_tensor - Aapprox # Use A_tensor

        current_Iset = set(cross_data.lu.Iset if isinstance(cross_data.lu.Iset, (list, tuple)) else [])
        current_Jset = set(cross_data.lu.Jset if isinstance(cross_data.lu.Jset, (list, tuple)) else [])

        mask = cytnx.ones(residual.shape(), dtype=residual.dtype(), device=residual.device())
        if current_Iset:
            for idx in list(current_Iset):
                if 0 <= idx < mask.shape()[0]: mask[idx, :] = 0.0
        if current_Jset:
            for idx in list(current_Jset):
                if 0 <= idx < mask.shape()[1]: mask[:, idx] = 0.0

        masked_residual = residual * mask
        abs_residual = cytnx.linalg.Abs(masked_residual)

        if not (abs_residual.shape()[0] > 0 and abs_residual.shape()[1] > 0): break
        flat_residual = abs_residual.reshape(-1)
        if not (flat_residual.shape()[0] > 0): break

        max_val, max_idx = -1.0, -1
        # Simplified max finding for Tensor (assuming Cytnx >= 0.9.0 for .Max())
        # For finding argmax, iteration or numpy conversion is still needed for older Cytnx
        # or if a specific element needs to be identified beyond just the max value.
        # The iteration below is fine for finding both value and flat index.
        temp_max_val, temp_max_idx = -1.0, -1
        for idx_flat in range(flat_residual.shape()[0]):
            val_item = flat_residual[idx_flat].item()
            if val_item > temp_max_val:
                temp_max_val, temp_max_idx = val_item, idx_flat
        max_val, max_idx = temp_max_val, temp_max_idx

        if max_idx == -1 or max_val < 1e-14:
            logger.info(f"Max residual {max_val:.2e} too small. Stopping."); break

        num_cols_res = abs_residual.shape()[1]
        if num_cols_res == 0: break
        i, j = max_idx // num_cols_res, max_idx % num_cols_res

        logger.info(f"Found pivot at ({i}, {j}), resid_val={residual[i,j].item():.4f} (masked_abs_max={max_val:.4f})")

        # Pass the original full Tensor A_tensor to addPivot
        cross_data.addPivot(i, j, A_tensor)
        logger.info(f"Rank after adding: {cross_data.rank()}")

    logger.info("--- Final Results ---")
    final_approx_matrix = cross_data.mat()
    if final_approx_matrix is not None:
        final_error_matrix = A_tensor - final_approx_matrix
        max_error = final_error_matrix.Abs().Max().item() if (final_error_matrix.shape()[0] > 0 and final_error_matrix.shape()[1] > 0) else 0.0
        logger.info(f"Final Rank: {cross_data.rank()}")
        logger.info(f"Final max abs error: {max_error:.6e}")
        logger.info(f"Final approximated matrix:\n{final_approx_matrix}")
    else:
        logger.error("Final approximated matrix is None.")
    logger.info(f"Original matrix:\n{A_tensor}")