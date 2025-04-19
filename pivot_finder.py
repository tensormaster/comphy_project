# filename: pivot_finder.py

import random
import numpy as np
import cytnx
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

# Assuming crossdata.py and matrix_interface.py are in the same directory
# or accessible via PYTHONPATH
from crossdata import CrossData #
from matrix_interface import IMatrix #

@dataclass
class PivotData:
    """Holds data for a potential pivot."""
    i: int = -1
    j: int = -1
    error: float = -1.0

@dataclass
class PivotFinderParam:
    """Parameters for PivotFinder."""
    full_piv: bool = False # Use full search instead of rook search
    n_rook_start: int = 1 # Number of random starting columns for rook search
    n_rook_iter: int = 5 # Max iterations per rook search start
    # Optional function: return True if (i, j) can be a pivot, False otherwise
    f_bool: Optional[Callable[[int, int], bool]] = None
    # Optional tensors for weighted error calculation
    weight_row: Optional[cytnx.Tensor] = None
    weight_col: Optional[cytnx.Tensor] = None

class PivotFinder:
    """
    Finds the next pivot for Cross Interpolation, mimicking pivot_finder.h.
    Tries to maximize the local error |A[i, j] - ci.eval(i, j)| * weights.
    """
    def __init__(self, param: PivotFinderParam = PivotFinderParam()):
        self.p = param #

    def _local_error(self, i: int, j: int, Aij: float, Aij_approx: float) -> float:
        """
        Computes the local error, applying weights if available.
        """
        err = abs(Aij_approx - Aij) #
        # Apply weights similar to environment error mode
        if self.p.weight_row is not None and self.p.weight_col is not None:
             # Assuming weights are 1D tensors (vectors)
             w_row = self.p.weight_row[i].item() if i < self.p.weight_row.shape()[0] else 1.0
             w_col = self.p.weight_col[j].item() if j < self.p.weight_col.shape()[0] else 1.0
             err *= abs(w_row * w_col) #
        return err

    def _filter(self, I0: List[int], J0: List[int]) -> List[Tuple[int, int]]:
        """
        Keeps the indices that satisfy the condition f_bool (if any).
        """
        ids = []
        if not self.p.f_bool: # No filter function provided
             ids = [(i, j) for i in I0 for j in J0] #
        else:
            for i in I0:
                for j in J0:
                    if self.p.f_bool(i, j): #
                        ids.append((i, j))
        return ids

    def _find_in(self, A: IMatrix, ci: CrossData, I0: List[int], J0: List[int]) -> PivotData:
        """
        Find the pivot maximizing localError within the submatrix A(I0, J0).
        """
        ids = self._filter(I0, J0) #
        if not ids:
            return PivotData() # Return default if no valid pivots

        # Evaluate true and approximate values
        # A.eval is expected to return a list of values for the list of (i,j) tuples
        data_fu = A.eval(ids) #
        # ci.eval evaluates one point at a time
        data_ci = [ci.eval(i, j) for i, j in ids] #


        pivot = PivotData() # Initialize best pivot found so far

        for c, (i, j) in enumerate(ids):
            err = self._local_error(i, j, data_fu[c], data_ci[c])
            if err > pivot.error: # Check if this is the best error so far
                pivot = PivotData(i=i, j=j, error=err)

        return pivot

    def _avail_cols_sample(self, ci: CrossData) -> List[int]:
        """
        Get a list of available columns, applying f_bool if needed.
        """
        avail_c = ci.availCols() #
        if not self.p.f_bool:
            return avail_c #

        # Filter available columns based on f_bool applied across any available row
        J0_filtered = set()
        avail_r = ci.availRows() #
        if not avail_r: # Need at least one row to check columns
             return []
        # Check each available column against available rows
        for j in avail_c:
             # Check if the column j is valid with at least one available row i
             if any(self.p.f_bool(i, j) for i in avail_r):
                  J0_filtered.add(j)
        return list(J0_filtered) #


    def __call__(self, A: IMatrix, ci: CrossData) -> PivotData:
        """
        Propose a pivot trying to maximize the local error of the cross interpolation ci
        with respect to the true matrix A.
        """
        I0 = ci.availRows() #
        J0 = ci.availCols() #

        pivot_best = PivotData() #
        if not I0 or not J0:
            return pivot_best # Cannot find pivot if no rows or cols available

        if self.p.full_piv:
            # Perform search over the entire available submatrix
            pivot_best = self._find_in(A, ci, I0, J0)
        else:
            # Perform alternating (rook) search
            Jsample = self._avail_cols_sample(ci) #
            if not Jsample:
                return pivot_best # Cannot start rook search without sample columns

            for _ in range(self.p.n_rook_start): #
                # Start with a random available column
                j_start = random.choice(Jsample)
                pivot = PivotData(j=j_start) # Initial pivot only has column

                for k in range(self.p.n_rook_iter): #
                    # Search for best row in the current column j
                    pivot_col_search = self._find_in(A, ci, I0, [pivot.j])
                    if pivot_col_search.i == pivot.i and k > 0: # Rook established if row didn't change
                         pivot = pivot_col_search # Update error just in case
                         break
                    pivot = pivot_col_search # Update current pivot (now has i, j, error)
                    if pivot.error == -1.0: # No valid pivot found in this column
                         break

                    # Search for best column in the current row i
                    pivot_row_search = self._find_in(A, ci, [pivot.i], J0)
                    if pivot_row_search.j == pivot.j: # Rook established if column didn't change
                         pivot = pivot_row_search # Update error just in case
                         break
                    pivot = pivot_row_search # Update current pivot
                    if pivot.error == -1.0: # No valid pivot found in this row
                         break

                # Update the best pivot found across different starts
                if pivot.error > pivot_best.error:
                    pivot_best = pivot

        return pivot_best

# Example Usage (requires AdaptiveLU.py from crossdata.py dependencies)
if __name__ == "__main__":
    from matrix_interface import MatDense
    from AdaptiveLU import AdaptiveLU # Make sure AdaptiveLU.py is available
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Example Matrix A
    M, N = 5, 4
    np_A = np.arange(M * N, dtype=float).reshape(M, N)
    np_A[2,2] += 100 # Make one element large
    A_ct = cytnx.from_numpy(np_A)
    A_mat = MatDense(A_ct) # Use the dense matrix interface
    logger.info(f"Original Matrix A:\n{A_ct}")

    # Initialize CrossData
    cross_data = CrossData(M, N) #
    # Need to initialize AdaptiveLU within CrossData if not done automatically
    cross_data.lu = AdaptiveLU(M, N)

    # --- Find First Pivot ---
    # For the very first pivot, the approximation is zero everywhere
    # So we just find the element with the largest absolute value in A
    abs_A = cytnx.linalg.Abs(A_ct)
    max_val = -1.0
    start_i, start_j = -1, -1
    for r in range(M):
        for c in range(N):
            val = abs_A[r, c].item()
            if val > max_val:
                max_val = val
                start_i, start_j = r, c

    logger.info(f"Found starting pivot at ({start_i}, {start_j}) with value {A_ct[start_i, start_j].item()}")
    cross_data.addPivot(start_i, start_j, A_ct) #

    # --- Find Subsequent Pivots using PivotFinder ---
    params = PivotFinderParam(full_piv=False) # Use rook search
    finder = PivotFinder(params)

    MAX_RANK = min(M, N)
    for k in range(1, MAX_RANK):
        logger.info(f"\n--- Iteration {k} ---")
        pivot_found = finder(A_mat, cross_data) #

        if pivot_found.error < 1e-12 or pivot_found.i == -1:
            logger.info("Stopping: Error is too small or no valid pivot found.")
            break

        logger.info(f"Found pivot at ({pivot_found.i}, {pivot_found.j}) with error {pivot_found.error:.4e}")
        cross_data.addPivot(pivot_found.i, pivot_found.j, A_ct) #
        logger.info(f"Current Rank: {cross_data.rank()}") #
        logger.info(f"Current Iset: {cross_data.lu.Iset}")
        logger.info(f"Current Jset: {cross_data.lu.Jset}")


    # --- Final Result ---
    A_approx = cross_data.mat() # Reconstruct the approximation
    logger.info(f"\nFinal Approximation (Rank {cross_data.rank()}):\n{A_approx}")
    error_tensor = A_ct - A_approx
    max_abs_error = cytnx.linalg.Abs(error_tensor).Max().item()
    logger.info(f"Max Absolute Error: {max_abs_error:.4e}")