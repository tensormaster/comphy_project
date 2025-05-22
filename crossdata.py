# filename: crossdata.py (Corrected for NumPy index casting error)

# tensormaster/comphy_project/comphy_project-main/crossdata.py
import logging
import numpy as np
import cytnx
from cytnx import Type, Device, Tensor # Explicit imports
from AdaptiveLU import AdaptiveLU 
from matrix_interface import IMatrix # Import IMatrix for type hinting
from typing import Optional, Union # For type hinting

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_2d(t: Tensor, axis=0) -> Tensor:
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
    m, n = tensor1.shape()
    p = tensor2.shape()[1]
    if tensor1.shape()[0] != tensor2.shape()[0]:
        raise ValueError("Tensors must have the same number of rows for horizontal stacking.")
    # Ensure consistent dtype and device for the result tensor
    res_dtype = tensor1.dtype()
    res_device = tensor1.device()
    if tensor2.dtype() != res_dtype: # Promote dtype if necessary (e.g. float + complex = complex)
        if (res_dtype == Type.Float or res_dtype == Type.Double) and \
           (tensor2.dtype() == Type.ComplexFloat or tensor2.dtype() == Type.ComplexDouble):
            res_dtype = tensor2.dtype()
    
    result = cytnx.zeros((m, n + p), dtype=res_dtype, device=res_device)
    result[:, :n] = tensor1.astype(res_dtype).to(res_device)
    result[:, n:n + p] = tensor2.astype(res_dtype).to(res_device)
    logger.debug(f"manual_hstack: stacked tensors with shapes {tensor1.shape()} and {tensor2.shape()} to {result.shape()}")
    return result

def manual_vstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    m, n = tensor1.shape()
    p = tensor2.shape()[0]
    if tensor1.shape()[1] != tensor2.shape()[1]:
        raise ValueError("Tensors must have the same number of columns for vertical stacking.")
    res_dtype = tensor1.dtype()
    res_device = tensor1.device()
    if tensor2.dtype() != res_dtype:
        if (res_dtype == Type.Float or res_dtype == Type.Double) and \
           (tensor2.dtype() == Type.ComplexFloat or tensor2.dtype() == Type.ComplexDouble):
            res_dtype = tensor2.dtype()

    result = cytnx.zeros((m + p, n), dtype=res_dtype, device=res_device)
    result[:m, :] = tensor1.astype(res_dtype).to(res_device)
    result[m:m + p, :] = tensor2.astype(res_dtype).to(res_device)
    logger.debug(f"manual_vstack: stacked tensors with shapes {tensor1.shape()} and {tensor2.shape()} to {result.shape()}")
    return result

class CrossData:
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.C: Optional[Tensor] = None
        self.R: Optional[Tensor] = None
        self.lu = AdaptiveLU(n_rows, n_cols)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        logger.debug(f"Initialized CrossData with {n_rows} rows and {n_cols} columns.")

    def _determine_target_dtype_device(self, existing_tensor: Optional[Tensor]) -> tuple[int, int]:
        """Helper to determine target dtype and device."""
        if existing_tensor is not None:
            return existing_tensor.dtype(), existing_tensor.device()
        return Type.Double, Device.cpu # Default

    def _add_pivot_row_from_tensor_data(self, original_row_idx: int, row_tensor_1d: Tensor):
        logger.debug(f"_add_pivot_row_from_tensor_data: Adding data for original row {original_row_idx}, shape={row_tensor_1d.shape()}")
        if len(row_tensor_1d.shape()) != 1 or row_tensor_1d.shape()[0] != self.n_cols:
            raise ValueError(f"Provided row_tensor_1d must be 1D with length {self.n_cols}, got shape {row_tensor_1d.shape()}")

        row_2d = row_tensor_1d.reshape(1, self.n_cols)
        
        if self.R is None:
            self.R = row_2d.clone()
            logger.debug(f"_add_pivot_row_from_tensor_data: Initialized R with shape {self.R.shape()}.")
        else:
            # Ensure consistent dtype/device before stacking
            res_dtype, res_device = self._determine_target_dtype_device(self.R)
            if row_2d.dtype() != res_dtype or row_2d.device() != res_device:
                # Check for type promotion (e.g., float + complex -> complex)
                if (res_dtype == Type.Float or res_dtype == Type.Double) and \
                   (row_2d.dtype() == Type.ComplexFloat or row_2d.dtype() == Type.ComplexDouble):
                    res_dtype = row_2d.dtype()
                self.R = self.R.astype(res_dtype).to(res_device) # Ensure self.R is compatible
                row_2d_compat = row_2d.astype(res_dtype).to(res_device)
            else:
                row_2d_compat = row_2d
            self.R = manual_vstack(self.R, row_2d_compat)
            logger.debug(f"_add_pivot_row_from_tensor_data: Updated R, new shape: {self.R.shape()}.")
        
        self.lu.add_pivot_row(original_row_idx, row_tensor_1d) # AdaptiveLU expects the 1D tensor

    def _add_pivot_col_from_tensor_data(self, original_col_idx: int, col_tensor_1d: Tensor):
        logger.debug(f"_add_pivot_col_from_tensor_data: Adding data for original col {original_col_idx}, shape={col_tensor_1d.shape()}")
        if len(col_tensor_1d.shape()) != 1 or col_tensor_1d.shape()[0] != self.n_rows:
            raise ValueError(f"Provided col_tensor_1d must be 1D with length {self.n_rows}, got shape {col_tensor_1d.shape()}")

        col_2d = col_tensor_1d.reshape(self.n_rows, 1)

        if self.C is None:
            self.C = col_2d.clone()
            logger.debug(f"_add_pivot_col_from_tensor_data: Initialized C with shape {self.C.shape()}.")
        else:
            res_dtype, res_device = self._determine_target_dtype_device(self.C)
            if col_2d.dtype() != res_dtype or col_2d.device() != res_device:
                if (res_dtype == Type.Float or res_dtype == Type.Double) and \
                   (col_2d.dtype() == Type.ComplexFloat or col_2d.dtype() == Type.ComplexDouble):
                    res_dtype = col_2d.dtype()
                self.C = self.C.astype(res_dtype).to(res_device)
                col_2d_compat = col_2d.astype(res_dtype).to(res_device)
            else:
                col_2d_compat = col_2d
            self.C = manual_hstack(self.C, col_2d_compat)
            logger.debug(f"_add_pivot_col_from_tensor_data: Updated C, new shape: {self.C.shape()}.")

        self.lu.add_pivot_col(original_col_idx, col_tensor_1d)

    # tensormaster/comphy_project/comphy_project-main/crossdata.py
# Ensure you have 'import numpy as np' at the top of the file.

# In class CrossData:
# ... other methods ...

    def addPivot(self, i: int, j: int, A_input: Union[Tensor, IMatrix]):
        """ Update the cross data by adding a new pivot at position (i, j).
            A_input can be a cytnx.Tensor or an IMatrix object.
        """
        logger.debug(f"addPivot: Adding pivot ({i}, {j}) from input type {type(A_input)}.")
        
        row_tensor_1d: Optional[Tensor] = None
        col_tensor_1d: Optional[Tensor] = None

        if isinstance(A_input, IMatrix):
            if not (0 <= i < A_input.n_rows and 0 <= j < A_input.n_cols):
                logger.error(f"addPivot: Pivot indices ({i},{j}) out of bounds for IMatrix "
                             f"dimensions ({A_input.n_rows}, {A_input.n_cols}).")
                return

            row_values_list = A_input.submat(rows=[i], cols=list(range(A_input.n_cols)))
            res_dtype_r, res_device_r = self._determine_target_dtype_device(self.R)
            # Corrected conversion from list:
            np_row_values = np.array(row_values_list) 
            row_tensor_1d = cytnx.from_numpy(np_row_values).to(res_device_r).astype(res_dtype_r)


            col_values_list = A_input.submat(rows=list(range(A_input.n_rows)), cols=[j])
            res_dtype_c, res_device_c = self._determine_target_dtype_device(self.C)
            # Corrected conversion from list:
            np_col_values = np.array(col_values_list)
            col_tensor_1d = cytnx.from_numpy(np_col_values).to(res_device_c).astype(res_dtype_c)


        elif isinstance(A_input, Tensor):
            A_tensor_2d = ensure_2d(A_input, axis=0) 
            if not (0 <= i < A_tensor_2d.shape()[0] and 0 <= j < A_tensor_2d.shape()[1]):
                logger.error(f"addPivot: Pivot indices ({i},{j}) out of bounds for Tensor "
                             f"shape {A_tensor_2d.shape()}.")
                return
            row_tensor_1d = A_tensor_2d[i, :].clone()
            col_tensor_1d = A_tensor_2d[:, j].clone()
        else:
            raise TypeError(f"Unsupported type for A_input in addPivot: {type(A_input)}")

        if row_tensor_1d is not None:
            # Ensure row_tensor_1d is 1D before passing
            if len(row_tensor_1d.shape()) != 1:
                 logger.warning(f"addPivot: row_tensor_1d for _add_pivot_row_from_tensor_data is not 1D, shape: {row_tensor_1d.shape()}. Reshaping.")
                 # This case should ideally not happen if logic above is correct
                 # For safety, attempt to flatten or handle, though source of error should be fixed.
                 # If it was (1, N), reshape to (N,). If (N,1), also to (N,).
                 # Assuming it should be a flat list of column entries for that row.
                 if row_tensor_1d.shape()[0] == 1 and row_tensor_1d.shape()[1] > 1: # was (1,N)
                     row_tensor_1d = row_tensor_1d.reshape(row_tensor_1d.shape()[1])
                 elif row_tensor_1d.shape()[1] == 1 and row_tensor_1d.shape()[0] > 1: # was (N,1)
                     row_tensor_1d = row_tensor_1d.reshape(row_tensor_1d.shape()[0])
                 # If still not 1D, _add_pivot_row_from_tensor_data will raise ValueError

            self._add_pivot_row_from_tensor_data(i, row_tensor_1d)
        if col_tensor_1d is not None:
            # Ensure col_tensor_1d is 1D
            if len(col_tensor_1d.shape()) != 1:
                logger.warning(f"addPivot: col_tensor_1d for _add_pivot_col_from_tensor_data is not 1D, shape: {col_tensor_1d.shape()}. Reshaping.")
                if col_tensor_1d.shape()[0] == 1 and col_tensor_1d.shape()[1] > 1: 
                     col_tensor_1d = col_tensor_1d.reshape(col_tensor_1d.shape()[1])
                elif col_tensor_1d.shape()[1] == 1 and col_tensor_1d.shape()[0] > 1:
                     col_tensor_1d = col_tensor_1d.reshape(col_tensor_1d.shape()[0])

            self._add_pivot_col_from_tensor_data(j, col_tensor_1d)
        
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}

# ... rest of the class ...

    def pivotMat(self) -> Optional[Tensor]:
        if self.C is None or not self.lu.Iset or self.rank() == 0 : # Check rank instead of Jset
            logger.debug("pivotMat: No pivot set found or C is None, or rank is 0. Returning None.")
            return None
        
        # Ensure Iset indices are valid for the current state of C
        iset_indices_np = np.array(self.lu.Iset[:self.rank()], dtype=np.int64) # Use current rank
        
        # Check bounds against C, which now grows with pivots
        if np.any(iset_indices_np >= self.C.shape()[0]):
            logger.error(f"pivotMat: Iset indices {iset_indices_np} (up to rank {self.rank()}) "
                         f"out of bounds for C shape {self.C.shape()}")
            return None
        
        # C stores columns. We need C[Iset, Jset_implicit_in_order]
        # If C has shape (n_rows, current_rank), and Iset maps to rows of original A.
        # pivotMat should be A[Iset, Jset].
        # C's columns are the Jset columns in order. C's rows are all n_rows of original A.
        # So, C[Iset_indices, :] gives the rows from original A, selected by Iset, across all chosen Jset columns.
        # This is A[Iset, Jset_ordered].
        try:
            # We need to select rows from C using Iset_indices, and all columns up to the current rank.
            # self.C has shape (self.n_rows, self.rank())
            if self.C.shape()[1] != self.rank():
                logger.error(f"pivotMat: C's column count {self.C.shape()[1]} does not match rank {self.rank()}.")
                return None

            pivot_matrix = self.C[iset_indices_np, :] 
            logger.debug(f"pivotMat: Pivot matrix shape: {pivot_matrix.shape()} (expected ({self.rank()}, {self.rank()}))")
            if pivot_matrix.shape()[0] != self.rank() or pivot_matrix.shape()[1] != self.rank():
                logger.warning(f"pivotMat: Resulting pivot matrix shape {pivot_matrix.shape()} is not square to rank {self.rank()}. This might be an issue.")

            return pivot_matrix
        except Exception as e:
             logger.error(f"pivotMat: Unexpected error: {e}", exc_info=True)
             return None

    def leftMat(self) -> Optional[Tensor]:
        k = self.rank()
        if k == 0: return None
        if self.cache.get('LD') is None:
            try:
                if self.lu.D is None or self.lu.L is None: return None
                if self.lu.D.shape()[0] < k or self.lu.L.shape()[1] < k: # Check L columns
                    logger.warning(f"leftMat: Not enough data in D ({self.lu.D.shape()}) or L ({self.lu.L.shape()}) for rank {k}")
                    return None
                D_diag = cytnx.linalg.Diag(self.lu.D[:k])
                L_k = self.lu.L[:, :k] # L has shape (n_rows, rank)
                if L_k.shape()[1] != k or D_diag.shape()[0] != k or D_diag.shape()[1] != k:
                    logger.warning(f"leftMat: Shape mismatch for L_k ({L_k.shape()}) or D_diag ({D_diag.shape()}) with rank {k}")
                    return None
                self.cache['LD'] = L_k @ D_diag
                logger.debug("leftMat: Computed new left matrix (L * D).")
            except Exception as e:
                 logger.error(f"leftMat: Error computing L*D: {e}", exc_info=True)
                 self.cache['LD'] = None
                 return None
        if self.cache.get('LD') is not None: logger.debug(f"leftMat: Shape: {self.cache['LD'].shape()}")
        return self.cache.get('LD')

    def rightMat(self) -> Optional[Tensor]:
        k = self.rank()
        if k == 0 or self.lu.U is None: return None
        if self.lu.U.shape()[0] < k: # Check U rows
            logger.warning(f"rightMat: Not enough data in U ({self.lu.U.shape()}) for rank {k}")
            return None
        U_k = self.lu.U[:k, :] # U has shape (rank, n_cols)
        if U_k.shape()[0] != k:
             logger.warning(f"rightMat: Shape mismatch for U_k ({U_k.shape()}) with rank {k}")
             return None
        logger.debug(f"rightMat: Shape: {U_k.shape()}")
        return U_k
    
    def availRows(self) -> list:
        if self.cache.get('I_avail') is None:
            current_Iset = set(self.lu.Iset)
            self.cache['I_avail'] = [i for i in range(self.n_rows) if i not in current_Iset]
        return self.cache.get('I_avail', [])

    def availCols(self) -> list:
        if self.cache.get('J_avail') is None:
            current_Jset = set(self.lu.Jset)
            self.cache['J_avail'] = [j for j in range(self.n_cols) if j not in current_Jset]
        return self.cache.get('J_avail', [])

    def rank(self) -> int:
        return self.lu.rank # Use rank from AdaptiveLU

    def firstPivotValue(self) -> float:
        if self.C is None or not self.lu.Iset or self.rank() == 0:
            logger.debug("firstPivotValue: No valid pivot set, C is None, or rank is 0. Returning 1.0")
            return 1.0
        try:
             first_pivot_row_idx = self.lu.Iset[0]
             # C's columns correspond to Jset in order. First col of C is Jset[0].
             first_pivot_col_idx_in_C = 0 # The first column added to C corresponds to self.lu.Jset[0]
             if first_pivot_row_idx >= self.C.shape()[0] or first_pivot_col_idx_in_C >= self.C.shape()[1]:
                  logger.error(f"firstPivotValue: First pivot index ({first_pivot_row_idx}, col_in_C {first_pivot_col_idx_in_C}) "
                               f"out of bounds for C shape {self.C.shape()}.")
                  return 1.0
             pivot_value = self.C[first_pivot_row_idx, first_pivot_col_idx_in_C].item()
             logger.debug(f"firstPivotValue: First pivot value is {pivot_value}")
             return pivot_value
        except Exception as e:
             logger.error(f"firstPivotValue: Error getting pivot value: {e}", exc_info=True)
             return 1.0

    def eval(self, i: int, j: int) -> float:
        if self.rank() == 0: return 0.0
        LD = self.leftMat() # Shape (n_rows, rank)
        U = self.rightMat()  # Shape (rank, n_cols)
        if LD is None or U is None: return 0.0
        try:
             if not (0 <= i < LD.shape()[0] and 0 <= j < U.shape()[1]):
                  logger.warning(f"eval: Index ({i},{j}) out of bounds for LD ({LD.shape()}) or U ({U.shape()}).")
                  return 0.0
             
             LD_row_i = LD[i, :] # 1D Tensor of shape (rank,)
             U_col_j = U[:, j]   # 1D Tensor of shape (rank,)

             if LD_row_i.shape()[0] != U_col_j.shape()[0] or LD_row_i.shape()[0] != self.rank():
                  logger.warning(f"eval: Shape mismatch for dot product. LD_row_i: {LD_row_i.shape()}, U_col_j: {U_col_j.shape()}, rank: {self.rank()}")
                  return 0.0
             
             val = cytnx.linalg.Dot(LD_row_i, U_col_j).item()
             return val
        except Exception as e:
             logger.error(f"eval: Error during evaluation at ({i},{j}): {e}", exc_info=True)
             return 0.0

    def mat(self) -> Tensor:
        logger.debug("mat: Reconstructing matrix using L, D, U from lu object.")
        k = self.rank()
        if k == 0:
             return cytnx.zeros((self.n_rows, self.n_cols), dtype=Type.Double, device=Device.cpu) # Default dtype/device
        try:
            # Get L, D_diag, U from lu component, sliced to current rank k
            L_k = self.lu.L[:, :k]
            D_values_k = self.lu.D[:k] # This is 1D
            U_k = self.lu.U[:k, :]

            if L_k.shape() != [self.n_rows, k] or D_values_k.shape() != [k] or U_k.shape() != [k, self.n_cols]:
                logger.error(f"mat: Shape mismatch for L_k({L_k.shape()}), D_values_k({D_values_k.shape()}), or U_k({U_k.shape()}) "
                             f"with rank {k}, n_rows {self.n_rows}, n_cols {self.n_cols}.")
                return cytnx.zeros((self.n_rows, self.n_cols), dtype=Type.Double, device=Device.cpu)

            D_diag_k = cytnx.linalg.Diag(D_values_k)
            
            Aapprox = L_k @ D_diag_k @ U_k
            if Aapprox.shape() != [self.n_rows, self.n_cols]:
                 logger.warning(f"mat: Reconstructed shape {Aapprox.shape()} differs from expected ({self.n_rows},{self.n_cols}).")
                 # This case should ideally not happen if L_k, D_diag_k, U_k are correct
                 return cytnx.zeros((self.n_rows, self.n_cols), dtype=Aapprox.dtype(), device=Aapprox.device())
            logger.debug(f"mat: Approximated matrix shape: {Aapprox.shape()}")
            return Aapprox
        except Exception as e:
            logger.error(f"mat: Error during lu.reconstruct logic: {e}", exc_info=True)
            return cytnx.zeros((self.n_rows, self.n_cols), dtype=Type.Double, device=Device.cpu)

if __name__ == "__main__":
    logger.setLevel(logging.INFO) # Set to INFO for tests, DEBUG for development
    logger.info("Starting test of CrossData with random matrix.") # Changed from debug to info
    M, N = 5, 4
    np.random.seed(0) # For reproducibility
    arr = np.random.rand(M, N) * 10
    # Make one element significantly larger to guide initial pivot choice if not masked
    arr[M//2, N//2] += 50 
    A = cytnx.from_numpy(arr).astype(Type.Double) # Ensure double type
    logger.info("Original matrix A:")
    logger.info(f"\n{A}")

    cross_data = CrossData(M, N)
    logger.info("Adding pivots and tracking approximation error:")
    MAX_RANK_TO_ADD = min(M, N)

    for k_iter in range(MAX_RANK_TO_ADD):
        logger.info(f"--- Iteration {k_iter + 1} ---")
        Aapprox = cross_data.mat()
        residual = A - Aapprox

        current_Iset = set(cross_data.lu.Iset)
        current_Jset = set(cross_data.lu.Jset)

        # Create a mask of available entries
        mask = cytnx.ones(residual.shape(), dtype=Type.Bool, device=residual.device())
        
        # Mask rows already in Iset
        if current_Iset:
            for r_idx in current_Iset:
                if 0 <= r_idx < mask.shape()[0]:
                    mask[r_idx, :] = False
        
        # Mask columns already in Jset
        if current_Jset:
            for c_idx in current_Jset:
                if 0 <= c_idx < mask.shape()[1]:
                    mask[:, c_idx] = False
        
        # Apply mask to absolute residual
        # Element-wise product for boolean mask might require casting or careful handling.
        # A simple way: iterate and find max only among True mask entries.
        abs_residual = cytnx.linalg.Abs(residual)
        
        max_val = -1.0
        pivot_i, pivot_j = -1, -1

        for r in range(abs_residual.shape()[0]):
            for c in range(abs_residual.shape()[1]):
                if mask[r, c].item(): # Check if this element is available
                    val = abs_residual[r, c].item()
                    if val > max_val:
                        max_val = val
                        pivot_i, pivot_j = r, c
        
        if pivot_i == -1 or max_val < 1e-12: # Tolerance for stopping
            logger.info(f"Max residual value {max_val:.2e} (at ({pivot_i},{pivot_j})) below tolerance or no valid pivot found. Stopping.")
            break

        logger.info(f"Found pivot at ({pivot_i}, {pivot_j}) with residual value {residual[pivot_i,pivot_j].item():.4f} "
                    f"(max abs masked residual: {max_val:.4f})")
        cross_data.addPivot(pivot_i, pivot_j, A) # Pass the original matrix A
        logger.info(f"Rank after adding: {cross_data.rank()}")
        logger.info(f"  Iset: {cross_data.lu.Iset}")
        logger.info(f"  Jset: {cross_data.lu.Jset}")
        # logger.info(f"  D: {cross_data.lu.D[:cross_data.rank()].numpy() if cross_data.rank() > 0 else []}")


    logger.info("--- Final Results ---")
    final_approx_matrix = cross_data.mat()
    final_error_matrix = A - final_approx_matrix
    max_abs_error_val = 0.0

    if np.prod(final_error_matrix.shape()) > 0:# Check if tensor is not empty
        # Calculate max absolute error only if the tensor has elements
        if final_error_matrix.shape()[0] > 0 and final_error_matrix.shape()[1] > 0:
             max_abs_error_val = cytnx.linalg.Abs(final_error_matrix).Max().item()


    logger.info(f"Final Rank: {cross_data.rank()}")
    logger.info(f"Final maximum absolute error: {max_abs_error_val:.6e}")
    logger.info(f"Final approximated matrix (Rank {cross_data.rank()}):")
    logger.info(f"\n{final_approx_matrix}")
    logger.info("Original matrix:")
    logger.info(f"\n{A}")

    # Verification that the test runs to completion
    assert cross_data.rank() <= MAX_RANK_TO_ADD, "Rank exceeded max possible."
    # A more meaningful assertion would depend on the expected error for a given rank.
    # For now, if it runs and the error is somewhat small, it's a pass for the script structure.
    logger.info(f"Test script completed. Final error was {max_abs_error_val:.2e}.")
    if MAX_RANK_TO_ADD > 0 : # Only check error if pivots were added
         assert max_abs_error_val < 1e-10, f"Final error {max_abs_error_val} is too high for a full rank approximation."
    
    logger.info("CrossData test script passed.")