# filename: tensor_ci.py

import cytnx
import numpy as np
import logging
import math
import copy
from typing import List, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field

# Assuming these modules are accessible
from IndexSet import IndexSet
from matrix_interface import IMatrix, IMatrixIndex, MatDenseIndex, MatLazyIndex
from crossdata import CrossData # Assumes modified addPivotRow/Col accepting vectors
from pivot_finder import PivotFinder, PivotFinderParam, PivotData
from AdaptiveLU import AdaptiveLU # Assuming AdaptiveLU is used by CrossData
from comphy_project.tensorfuc import CytnxTensorFunction
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define MultiIndex type alias for clarity (often tuples of ints)
MultiIndex = Tuple[Any, ...] # Use Any to match IndexSet's flexibility

@dataclass
class TensorCIParam:
    """Parameters for the TensorCI algorithm."""
    n_iter: int = 0
    reltol: float = 1e-12
    pivot1: Optional[List[int]] = None # First pivot multi-index (full length)
    pivot_finder_param: PivotFinderParam = field(default_factory=PivotFinderParam)
    weights: Optional[List[cytnx.Tensor]] = None # For environment learning
    cond: Optional[Callable[[MultiIndex], bool]] = None
    pi_matrix_dense: bool = True # Use MatDenseIndex for Pi matrices
    use_function_cache: bool = True # Use function cache for f_raw

# Helper function to combine multi-indices
def combine_multi_indices(set1: IndexSet, set2: IndexSet) -> IndexSet:
    """ Combine two IndexSets containing multi-indices. """
    combined_list = []
    list1 = set1.from_int()
    list2 = set2.from_int()

    # Ensure indices are tuples, handle potential non-iterables in IndexSet
    def to_tuple(item):
        try: return tuple(item)
        except TypeError: return (item,)

    list1_tuples = [to_tuple(idx) for idx in list1]
    list2_tuples = [to_tuple(idx) for idx in list2]

    if not list1_tuples: return IndexSet(list2_tuples)
    if not list2_tuples: return IndexSet(list1_tuples)

    for idx1 in list1_tuples:
        for idx2 in list2_tuples:
            combined_list.append(idx1 + idx2) # Concatenate tuples
    return IndexSet(combined_list)


class TensorCI:
    """
    Builds a Tensor Train using 2-site Cross Interpolation.
    Mimics TensorCI1 from tensor_ci.h.
    """
    def __init__(self, f: Callable[[MultiIndex], float], local_dims: List[int], param: TensorCIParam = TensorCIParam()):
        self.f_raw = f
        self.f = CytnxTensorFunction(f, use_cache=param.use_function_cache)
        self.param = param
        self.local_dims = local_dims
        self.L = len(local_dims) # Number of sites

        if self.param.pivot1 is None:
            self.param.pivot1 = [0] * self.L
        if len(self.param.pivot1) != self.L:
             raise ValueError("Length of pivot1 must match number of local dimensions")

        # --- Initialize Data Structures ---
        self.Iset: List[IndexSet] = [IndexSet() for _ in range(self.L + 1)] # Pad for easier indexing Iset[0] to Iset[L]
        self.Jset: List[IndexSet] = [IndexSet() for _ in range(self.L + 1)] # Pad Jset[0] to Jset[L]
        self.localSet: List[IndexSet] = [IndexSet([(i,) for i in range(d)]) for d in self.local_dims] # Local indices as tuples

        self.T3: List[Optional[cytnx.Tensor]] = [None] * self.L # T-tensors T(i,s,j)
        self.P: List[Optional[cytnx.Tensor]] = [None] * (self.L) # Pivot matrices P(i,j) for bonds 0 to L-1

        # Pi matrix representations and their CI states (for bonds 0 to L-2)
        self.Pi_mat: List[Optional[IMatrixIndex]] = [None] * (self.L - 1)
        self.cross: List[Optional[CrossData]] = [None] * (self.L - 1)

        self.pivot_error: List[float] = [] # Max error per iteration
        self.pivot_error_last_iter: List[float] = [1.0] * (self.L - 1)
        self.cIter: int = 0 # Iteration counter

                # --- Environment Initialization (if weights provided) ---
        self.L_env: Optional[List[Optional[cytnx.Tensor]]] = None
        self.R_env: Optional[List[Optional[cytnx.Tensor]]] = None
        if self.param.weights:
            if len(self.param.weights) != self.L:
                 logger.error("Length of weights must match number of sites. Disabling environment learning.")
            else:
                logger.info("Initializing environment vectors (L_env, R_env)...")
                self.L_env: List[Optional[cytnx.Tensor]] = [None] * (self.L + 1)
                self.R_env: List[Optional[cytnx.Tensor]] = [None] * (self.L + 1)

                # --- Calculate initial L_env ---
                # L_env[0] is scalar 1 (represented as a rank-0 tensor or tensor with shape [1])
                L_env_0_np = np.array([1.0], dtype=np.float64)
                self.L_env[0] = cytnx.from_numpy(np.array([1.0], dtype=np.float64)).to(cytnx.Device.cpu)
                self.R_env[self.L] = cytnx.from_numpy(np.array([1.0], dtype=np.float64)).to(cytnx.Device.cpu)
                for p in range(self.L):
                    Mp = self._get_TP1_at(p) if p < self.L -1 else self.T3[p] # Use TP1 or T3 for M_p
                    wp = self.param.weights[p] # Weight for site p (expected shape [sp])
                    L_prev = self.L_env[p]

                    if Mp is None or L_prev is None or wp is None:
                        logger.error(f"Cannot calculate L_env[{p+1}], required components missing.")
                        # Invalidate remaining L_env
                        for k in range(p + 1, self.L + 1): self.L_env[k] = None
                        break

                    try:
                        # Contract L_prev [r_prev] -- M_p [r_prev, s_p, r_p] -- w_p [s_p] -> L_curr [r_p]
                        # 1. Contract L_prev with first index of M_p
                        #    reshape L_prev to [1, r_prev]
                        #    reshape M_p to [r_prev, s_p * r_p]
                        #    tmp = L_prev @ M_p -> [1, s_p * r_p] -> reshape [s_p, r_p]
                        r_prev, s_p, r_p = Mp.shape()
                        L_prev_resh = L_prev.reshape(1, r_prev)
                        Mp_resh = Mp.reshape(r_prev, s_p * r_p)
                        tmp = L_prev_resh @ Mp_resh # [1, s_p * r_p]
                        tmp_resh = tmp.reshape(s_p, r_p)

                        # 2. Contract tmp_resh with wp (element-wise assumed)
                        #    wp shape [s_p], needs reshape to [s_p, 1] maybe?
                        #    Result L_curr should be shape [r_p]
                        wp_resh = wp.reshape(s_p, 1)
                        # Element-wise product? tmp_resh * wp_resh -> [s_p, r_p]
                        # Then sum over s_p axis -> [r_p]
                        # Use tensordot: contract axis 0 of tmp_resh with axis 0 of wp
                        L_curr = cytnx.linalg.Tensordot(tmp_resh, wp, axes=([0],[0])) # Result shape [r_p]
                        self.L_env[p+1] = L_curr.contiguous() # Store contiguous copy
                        logger.debug(f"  L_env[{p+1}] shape: {self.L_env[p+1].shape()}")

                    except Exception as e:
                        logger.error(f"Error calculating L_env[{p+1}]: {e}")
                        for k in range(p + 1, self.L + 1): self.L_env[k] = None
                        break

                # --- Calculate initial R_env ---
                # R_env[L] is scalar 1
                R_env_L_np = np.array([1.0], dtype=np.float64)
                self.R_env[self.L] = cytnx.from_numpy(R_env_L_np).to(cytnx.Device.cpu) # FIXED
                for p in range(self.L - 1, -1, -1): # Sweep from L-1 down to 0
                    Mp = self._get_P1T_at(p) if p > 0 else self.T3[p] # Use P1T or T3 for M_p
                    wp = self.param.weights[p] # Weight for site p
                    R_next = self.R_env[p+1]

                    if Mp is None or R_next is None or wp is None:
                        logger.error(f"Cannot calculate R_env[{p}], required components missing.")
                        for k in range(p, -1, -1): self.R_env[k] = None
                        break

                    try:
                        # Contract w_p [s_p] -- M_p [r_p, s_p, r_next] -- R_next [r_next] -> R_curr [r_p]
                        # 1. Contract M_p with R_next
                        #    reshape M_p to [r_p * s_p, r_next]
                        #    reshape R_next to [r_next, 1]
                        #    tmp = M_p @ R_next -> [r_p * s_p, 1] -> reshape [r_p, s_p]
                        r_p, s_p, r_next = Mp.shape()
                        Mp_resh = Mp.reshape(r_p * s_p, r_next)
                        R_next_resh = R_next.reshape(r_next, 1)
                        tmp = Mp_resh @ R_next_resh # [r_p * s_p, 1]
                        tmp_resh = tmp.reshape(r_p, s_p)

                        # 2. Contract tmp_resh with wp
                        #    wp shape [s_p], needs reshape [1, s_p]?
                        #    Result R_curr shape [r_p]
                        # Use tensordot: contract axis 1 of tmp_resh with axis 0 of wp
                        R_curr = cytnx.linalg.Tensordot(tmp_resh, wp, axes=([1],[0])) # Result shape [r_p]
                        self.R_env[p] = R_curr.contiguous()
                        logger.debug(f"  R_env[{p}] shape: {self.R_env[p].shape()}")

                    except Exception as e:
                        logger.error(f"Error calculating R_env[{p}]: {e}")
                        for k in range(p, -1, -1): self.R_env[k] = None
                        break
                logger.info("Environment vector initialization complete.")

        # --- Rank-1 Initialization ---
        logger.info("Starting Rank-1 Initialization...")
        pivot1_tuple = tuple(self.param.pivot1) # Ensure tuple for function call
        try:
            pivot1_val = self.f_raw(pivot1_tuple)
        except Exception as e:
            logger.error(f"Error calling initial pivot function f({pivot1_tuple}): {e}")
            raise
        if abs(pivot1_val) < 1e-15:
             raise ValueError(f"f(pivot1={pivot1_tuple}) is zero ({pivot1_val}). Provide a better first pivot.")
        self.pivot_error.append(abs(pivot1_val))

        # Initialize Iset/Jset with the first pivot's indices (as tuples)
        self.Iset[0] = IndexSet([()]) # Bond 0: Empty tuple index
        for p in range(self.L): # Iset[1] to Iset[L]
            self.Iset[p+1] = IndexSet([pivot1_tuple[:p+1]])
        for p in range(self.L): # Jset[0] to Jset[L-1]
            self.Jset[p] = IndexSet([pivot1_tuple[p+1:]])
        self.Jset[self.L] = IndexSet([()]) # Bond L: Empty tuple index

        # Build Pi matrices and CrossData for each bond (0 to L-2)
        for p in range(self.L - 1):
            self.Pi_mat[p] = self._build_pi_at(p)
            if self.Pi_mat[p].n_rows == 0 or self.Pi_mat[p].n_cols == 0:
                 logger.warning(f"Pi matrix at bond {p} has zero dimensions. Skipping CrossData init.")
                 continue # Cannot initialize CrossData for zero-dim matrix

            self.cross[p] = CrossData(self.Pi_mat[p].n_rows, self.Pi_mat[p].n_cols)
            # AdaptiveLU should be initialized inside CrossData or here if needed externally
            # self.cross[p].lu = AdaptiveLU(self.Pi_mat[p].n_rows, self.Pi_mat[p].n_cols)

            # Add the first pivot (index (0, 0) relative to the rank-1 Pi matrix)
            pivot_i_idx = 0 # Corresponds to Iset[p] + localSet[p]
            pivot_j_idx = 0 # Corresponds to localSet[p+1] + Jset[p+1]

            try:
                # Extract the row/column vector for the modified addPivotRow/Col
                pi_row_vals = self.Pi_mat[p].submat([pivot_i_idx], list(range(self.Pi_mat[p].n_cols)))
                pi_col_vals = self.Pi_mat[p].submat(list(range(self.Pi_mat[p].n_rows)), [pivot_j_idx])
                # Convert list to cytnx Tensor
                pi_row_np = np.array(pi_row_vals, dtype=np.float64)
                pi_col_np = np.array(pi_col_vals, dtype=np.float64)
                pi_row_vec = cytnx.from_numpy(pi_row_np) # FIXED
                pi_col_vec = cytnx.from_numpy(pi_col_np) # FIXED

                # Call the MODIFIED CrossData methods accepting vectors
                self.cross[p].addPivotRow(pivot_i_idx, pi_row_vec)
                self.cross[p].addPivotCol(pivot_j_idx, pi_col_vec)

                # Extract initial T3 and P (rank 1)
                self._update_T3P_from_cross(p)
            except Exception as e:
                logger.error(f"Error adding initial pivot for bond {p}: {e}")
                # Handle error appropriately, maybe skip this bond or raise

        logger.info("Rank-1 Initialization Complete.")

        # Perform initial iterations if requested
        self.iterate(self.param.n_iter)


    def len(self) -> int:
        """Return the number of sites."""
        return self.L

    def _build_pi_at(self, p: int) -> IMatrixIndex:
        """Build the IMatrixIndex representation for Pi_p."""
        logger.debug(f"_build_pi_at(p={p})")
        # Combine Iset[p] and localSet[p] for rows
        # Combine localSet[p+1] and Jset[p+1] for columns
        I_combined = combine_multi_indices(self.Iset[p], self.localSet[p])
        J_combined = combine_multi_indices(self.localSet[p+1], self.Jset[p+1])
        logger.debug(f"  Pi_{p}: Iset size = {len(I_combined)}, Jset size = {len(J_combined)}")

        # Function to evaluate Pi_p(combined_i, combined_j)
        def pi_func(idx_i: MultiIndex, idx_j: MultiIndex) -> float:
            try:
                # Ensure inputs are tuples
                idx_i = tuple(idx_i) if isinstance(idx_i, (list, tuple)) else (idx_i,)
                idx_j = tuple(idx_j) if isinstance(idx_j, (list, tuple)) else (idx_j,)

                # Split combined indices back based on lengths
                len_i_p = len(self.Iset[p].from_int()[0]) if len(self.Iset[p]) > 0 else 0
                len_s_p = 1 # localSet[p] indices are single elements (tuples)
                len_s_p1 = 1 # localSet[p+1] indices are single elements (tuples)

                # Check if combined indices have expected length
                if len(idx_i) != len_i_p + len_s_p:
                     raise ValueError(f"Row index {idx_i} length mismatch (expected {len_i_p + len_s_p})")
                if len(idx_j) != len_s_p1 + len(self.Jset[p+1].from_int()[0]) if len(self.Jset[p+1]) > 0 else len_s_p1:
                     raise ValueError(f"Col index {idx_j} length mismatch")

                i_part = idx_i[:len_i_p]
                s_p = idx_i[len_i_p:]
                s_p1 = idx_j[:len_s_p1]
                j_part = idx_j[len_s_p1:]

                full_idx = i_part + s_p + s_p1 + j_part
                # logger.debug(f"pi_func: Evaluating f({full_idx}) from ({idx_i}, {idx_j})")
                return self.f_raw(full_idx)
            except Exception as e:
                 logger.error(f"Error in pi_func for bond {p} with indices ({idx_i}, {idx_j}): {e}")
                 # Decide on error handling: return 0.0, NaN, or raise? Raising is safer.
                 raise

        # Create the appropriate matrix interface instance
        return IMatrix(pi_func, I_combined.from_int(), J_combined.from_int(), self.param.pi_matrix_dense)


    def _build_pivot_finder_at(self, p: int) -> PivotFinder:
        """Configure PivotFinder for bond p."""
        pf_param = copy.deepcopy(self.param.pivot_finder_param)

        # --- Environment Weights (Keep existing logic here) ---
        if self.param.weights and self.tt_env is not None:
            # ... (Existing code to calculate pf_param.weight_row/col) ...
            L_p = self.L_env[p]
            R_p1 = self.R_env[p+1]
            w_p = self.param.weights[p]
            w_p1 = self.param.weights[p+1]
            pi_rows = self.Pi_mat[p].n_rows
            pi_cols = self.Pi_mat[p].n_cols
            if L_p is not None and R_p1 is not None and w_p is not None and w_p1 is not None:
                try:
                    s_p = self.local_dims[p]
                    s_p1 = self.local_dims[p+1]
                    r_p = L_p.shape()[0]
                    r_p1 = R_p1.shape()[0]
                    weight_row_mat = L_p.reshape(r_p, 1) * w_p.reshape(1, s_p)
                    weight_row_vec = weight_row_mat.reshape(-1)
                    weight_col_mat = w_p1.reshape(s_p1, 1) * R_p1.reshape(1, r_p1)
                    weight_col_vec = weight_col_mat.reshape(-1)
                    if weight_row_vec.shape()[0] == pi_rows and weight_col_vec.shape()[0] == pi_cols:
                        pf_param.weight_row = cytnx.linalg.Abs(weight_row_vec)
                        pf_param.weight_col = cytnx.linalg.Abs(weight_col_vec)
                        logger.debug(f"  Added environment weights for PivotFinder at bond {p}")
                    else: logger.warning(f"  Weight dimension mismatch at bond {p}. Skipping weights.")
                except Exception as e: logger.error(f"  Error calculating environment weights for bond {p}: {e}")


        # --- Pivot Condition Check (Implementation Added) ---
        if self.param.cond:
            pi_mat_p = self.Pi_mat[p] # <<< Get reference to Pi Matrix for index mapping
            if pi_mat_p is None:
                 logger.warning(f"Cannot apply pivot condition at bond {p}: Pi_mat is None.")
            else:
                # Define helper function to map Pi indices -> global index -> call cond
                def pi_cond(pi_i_idx: int, pi_j_idx: int) -> bool:
                    try:
                        idx_i = pi_mat_p.Iset.from_int()[pi_i_idx]
                        idx_j = pi_mat_p.Jset.from_int()[pi_j_idx]
                        # Reconstruct full_idx (ensure idx_i/j are tuples)
                        idx_i = tuple(idx_i) if isinstance(idx_i, (list, tuple)) else (idx_i,)
                        idx_j = tuple(idx_j) if isinstance(idx_j, (list, tuple)) else (idx_j,)
                        len_i_p = len(self.Iset[p].from_int()[0]) if len(self.Iset[p]) > 0 else 0
                        i_part = idx_i[:len_i_p]
                        s_p = idx_i[len_i_p:] # Should be len 1
                        s_p1 = idx_j[:1]      # Should be len 1
                        j_part = idx_j[1:]
                        full_idx = i_part + s_p + s_p1 + j_part
                        return self.param.cond(full_idx) # <<< Call user's condition function
                    except Exception as e:
                         logger.error(f"Error during pivot condition check at bond {p} for Pi indices ({pi_i_idx}, {pi_j_idx}): {e}")
                         return False # Disallow pivot on error
                # Assign the checker function to the PivotFinder parameter
                pf_param.f_bool = pi_cond # <<< Assign condition check function
                logger.debug(f"  Added pivot condition check function for PivotFinder at bond {p}")

        return PivotFinder(pf_param)

    def iterate(self, n_iter: int = 1):
        """Perform sweeps to add pivots."""
        logger.info(f"Starting {n_iter} iteration(s)...")
        for i in range(n_iter):
            self.cIter += 1
            current_max_error = 0.0
            if self.cIter == 1 and self.param.n_iter == 0 : # Only skip if n_iter was explicitly 0
                 logger.info("Skipping initial iteration as n_iter=0.")
                 continue

            # Determine sweep direction (bonds 0 to L-2)
            sweep_bonds = range(self.L - 1) if self.cIter % 2 != 0 else range(self.L - 2, -1, -1) # Start forward

            logger.info(f"Iteration {self.cIter} {'(Forward)' if self.cIter % 2 != 0 else '(Backward)'} Sweep...")
            errors_this_sweep = []
            for p in sweep_bonds:
                logger.debug(f"Processing bond {p}")
                error_p = self.add_pivot_at(p)
                if error_p is not None:
                    self.pivot_error_last_iter[p] = error_p
                    errors_this_sweep.append(error_p)

            current_max_error = max(errors_this_sweep) if errors_this_sweep else 0.0
            self.pivot_error.append(current_max_error)
            logger.info(f"Iteration {self.cIter} finished. Max error this iter: {current_max_error:.4e}")

            # Check for convergence based on relative tolerance
            initial_error = self.pivot_error[0] if self.pivot_error else 1.0
            if initial_error == 0: initial_error = 1.0 # Avoid division by zero
            if current_max_error < self.param.reltol * initial_error:
                 logger.info(f"Convergence tolerance ({self.param.reltol:.1e}) reached.")
                 break

    def add_pivot_at(self, p: int) -> Optional[float]:
        """Try to add a pivot at bond p. Returns the error of the added pivot or None."""
        if self.Pi_mat[p] is None or self.cross[p] is None:
             logger.warning(f"Skipping bond {p}: Pi matrix or CrossData missing.")
             return None

        finder = self._build_pivot_finder_at(p)
        pivot_data = finder(self.Pi_mat[p], self.cross[p])

        # Check tolerance relative to initial error
        initial_error = self.pivot_error[0] if self.pivot_error else 1.0
        if initial_error == 0: initial_error = 1.0
        tolerance = self.param.reltol * initial_error

        if pivot_data.error < tolerance or pivot_data.i == -1:
             logger.debug(f"Pivot error {pivot_data.error:.2e} at bond {p} below tolerance {tolerance:.2e} or invalid.")
             return None

        logger.info(f"Adding pivot at bond {p}: Pi-index ({pivot_data.i}, {pivot_data.j}), Error: {pivot_data.error:.4e}")

        try:
            # Get the actual row/column vectors from Pi_mat BEFORE updating CrossData
            pi_row_vals = self.Pi_mat[p].submat([pivot_data.i], list(range(self.Pi_mat[p].n_cols)))
            pi_col_vals = self.Pi_mat[p].submat(list(range(self.Pi_mat[p].n_rows)), [pivot_data.j])
            pi_row_np = np.array(pi_row_vals, dtype=np.float64)
            pi_col_np = np.array(pi_col_vals, dtype=np.float64)
            pi_row_vec = cytnx.from_numpy(pi_row_np)
            pi_col_vec = cytnx.from_numpy(pi_col_np)

            # Ensure correct shapes for addPivotRow/Col
            pi_row_vec = pi_row_vec.reshape(1, -1)
            pi_col_vec = pi_col_vec.reshape(-1, 1)

            # Add pivot row info (updates cross[p].R, lu, Iset[p+1], T3[p+1], P[p])
            self._add_pivot_row_at(p, pivot_data.i, pi_row_vec)
            # Add pivot col info (updates cross[p].C, lu, Jset[p], T3[p], P[p])
            self._add_pivot_col_at(p, pivot_data.j, pi_col_vec)

            return pivot_data.error
        except Exception as e:
            logger.error(f"Error processing pivot addition at bond {p}: {e}")
            return None # Indicate failure
    
    # Inside class TensorCI:

    def _update_env_at(self, p: int):
        """ Update L_env and R_env after pivot addition at bond p, propagating changes. """
        # <<< This method needs the full propagation logic >>>
        if not self.param.weights or self.tt_env is None: # Check tt_env renamed to L_env/R_env
             return

        logger.debug(f"_update_env_at(p={p}) - Propagating environment updates.")

        # --- Update L environment from site p outwards ---
        logger.debug(f"  Updating L environment from bond {p} >>")
        valid_L = True
        # Loop from the affected bond p up to L-1 (to update L_env[p+1]...L_env[L])
        for k in range(p, self.L):
            L_prev = self.L_env[k]
            # Use effective MPS tensor M_k
            Mk = self._get_TP1_at(k) if k < self.L - 1 else self.T3[k] # Use TP1 for internal, T3 for last
            wk = self.param.weights[k]

            if L_prev is None or Mk is None or wk is None:
                logger.warning(f"  Cannot update L_env[{k+1}], required component missing. Invalidating subsequent L_env.")
                for m in range(k + 1, self.L + 1): self.L_env[m] = None
                valid_L = False
                break # Stop propagating L update

            try:
                r_prev, s_k, r_k = Mk.shape()
                if L_prev.shape()[0] != r_prev: raise ValueError("L/M shape mismatch")
                if wk.shape()[0] != s_k: raise ValueError("w/M shape mismatch")

                # Contract: L_curr [r_k] = L_prev [r_prev] -- M_k [r_prev, s_k, r_k] -- w_k [s_k]
                L_prev_resh = L_prev.reshape(1, r_prev)
                Mk_resh = Mk.reshape(r_prev, s_k * r_k)
                tmp = L_prev_resh @ Mk_resh
                tmp_resh = tmp.reshape(s_k, r_k)
                L_curr = cytnx.linalg.Tensordot(tmp_resh, wk, axes=([0],[0]))
                self.L_env[k+1] = L_curr.contiguous()
                logger.debug(f"    Updated L_env[{k+1}] shape: {self.L_env[k+1].shape()}")
            except Exception as e:
                logger.error(f"Error updating L_env[{k+1}]: {e}")
                for m in range(k + 1, self.L + 1): self.L_env[m] = None
                valid_L = False
                break
        if valid_L: logger.debug("  Finished L environment update sweep.")


        # --- Update R environment from site p+1 backwards ---
        logger.debug(f"  Updating R environment << from bond {p+1} ") # Start update from R[p+1]
        valid_R = True
        # Loop from site p+1 down to 0 (to update R_env[p], R_env[p-1], ..., R_env[0])
        for k in range(p + 1, -1, -1):
            R_next = self.R_env[k+1]
            # Use effective MPS tensor M_k
            Mk = self._get_P1T_at(k) if k > 0 else self.T3[k] # Use P1T for internal, T3 for first
            wk = self.param.weights[k]

            if R_next is None or Mk is None or wk is None:
                logger.warning(f"  Cannot update R_env[{k}], required component missing. Invalidating preceding R_env.")
                for m in range(k, -1, -1): self.R_env[m] = None
                valid_R = False
                break # Stop propagating R update

            try:
                r_k, s_k, r_next = Mk.shape()
                if R_next.shape()[0] != r_next: raise ValueError("M/R shape mismatch")
                if wk.shape()[0] != s_k: raise ValueError("M/w shape mismatch")

                # Contract: R_curr [r_k] = w_k [s_k] -- M_k [r_k, s_k, r_next] -- R_next [r_next]
                Mk_resh = Mk.reshape(r_k * s_k, r_next)
                R_next_resh = R_next.reshape(r_next, 1)
                tmp = Mk_resh @ R_next_resh
                tmp_resh = tmp.reshape(r_k, s_k)
                R_curr = cytnx.linalg.Tensordot(tmp_resh, wk, axes=([1],[0]))
                self.R_env[k] = R_curr.contiguous()
                logger.debug(f"    Updated R_env[{k}] shape: {self.R_env[k].shape()}")
            except Exception as e:
                logger.error(f"Error updating R_env[{k}]: {e}")
                for m in range(k, -1, -1): self.R_env[m] = None
                valid_R = False
                break
        if valid_R: logger.debug("  Finished R environment update sweep.")

    def _get_Pinv(self, p: int) -> Optional[cytnx.Tensor]:
        """ Helper to calculate pseudo-inverse of P[p] """
        P_p = self.P[p]
        bond_rank = P_p.shape()[0] if P_p is not None else 0
        if P_p is None or bond_rank == 0:
            logger.debug(f"P[{p}] is None or empty, cannot invert.")
            return None
        try:
            if P_p.shape()[0] != P_p.shape()[1]:
                logger.warning(f"P[{p}] is not square ({P_p.shape()}), using pseudoinverse.")

            # Use SVD based pseudoinverse for stability
            # Add checks for S being non-empty
            U, S, Vt = cytnx.linalg.Svd(P_p)
            if S.shape()[0] == 0:
                 logger.warning(f"SVD of P[{p}] resulted in empty S, cannot invert.")
                 return None

            tol = 1e-14 * S[0].item() # Use tolerance based on largest singular value
            S_inv_vals = [1.0 / s.item() if s.item() > tol else 0.0 for s in S]
            S_inv_diag = cytnx.linalg.Diag(cytnx.Tensor(S_inv_vals, device=P_p.device()).astype(P_p.dtype()))

            # Ensure dimensions match for multiplication
            if Vt.shape()[1] != S_inv_diag.shape()[0] or S_inv_diag.shape()[1] != U.shape()[0]:
                 logger.error(f"Shape mismatch during Pinv[{p}] calculation: Vt({Vt.shape()}), S_inv({S_inv_diag.shape()}), U({U.shape()})")
                 return None

            Pinv_p = Vt.Conj().Transpose() @ S_inv_diag @ U.Conj().Transpose()
            logger.debug(f"Calculated P_inv[{p}] shape {Pinv_p.shape()}")
            return Pinv_p
        except Exception as e:
            logger.error(f"Error calculating inverse for P[{p}]: {e}")
            return None

    def _get_TP1_at(self, p: int) -> Optional[cytnx.Tensor]:
        """ Computes T3[p] @ P[p]^-1 """
        T3_p = self.T3[p]
        Pinv_p = self._get_Pinv(p)

        if T3_p is None or Pinv_p is None:
            logger.warning(f"Cannot compute TP1 at {p}: T3 or Pinv is None.")
            return None

        try:
            rp, sp, rp1 = T3_p.shape()
            if Pinv_p.shape() != [rp1, rp1]:
                 logger.error(f"Shape mismatch for TP1 at {p}: T3 shape {T3_p.shape()}, Pinv shape {Pinv_p.shape()}")
                 return None
            # Reshape T3[p]: [rp, sp, rp1] -> [rp * sp, rp1]
            T3p_mat = T3_p.reshape(rp * sp, rp1)
            Mp_mat = T3p_mat @ Pinv_p # [rp * sp, rp1]
            Mp = Mp_mat.reshape(rp, sp, rp1)
            logger.debug(f"Computed TP1[{p}] shape {Mp.shape()}")
            return Mp
        except Exception as e:
            logger.error(f"Error computing TP1 at {p}: {e}")
            return None

    def _get_P1T_at(self, p: int) -> Optional[cytnx.Tensor]:
        """ Computes P[p-1]^-1 @ T3[p] """
        T3_p = self.T3[p]
        # Get Pinv for the *previous* bond
        if p == 0: # Cannot compute P^-1 T for the first site
             logger.warning("Cannot compute P1T at site 0.")
             return T3_p # Return T3 itself, as Pinv[-1] is identity (conceptually)
        Pinv_prev = self._get_Pinv(p - 1)

        if T3_p is None or Pinv_prev is None:
            logger.warning(f"Cannot compute P1T at {p}: T3 or Pinv_prev is None.")
            return None

        try:
            rp, sp, rp1 = T3_p.shape()
            if Pinv_prev.shape() != [rp, rp]:
                 logger.error(f"Shape mismatch for P1T at {p}: Pinv_prev shape {Pinv_prev.shape()}, T3 shape {T3_p.shape()}")
                 return None
            # Reshape T3[p]: [rp, sp, rp1] -> [rp, sp * rp1]
            T3p_mat = T3_p.reshape(rp, sp * rp1)
            Mp_mat = Pinv_prev @ T3p_mat # [rp, sp * rp1]
            Mp = Mp_mat.reshape(rp, sp, rp1)
            logger.debug(f"Computed P1T[{p}] shape {Mp.shape()}")
            return Mp
        except Exception as e:
            logger.error(f"Error computing P1T at {p}: {e}")
            return None


    def _add_pivot_row_at(self, p: int, pivot_i: int, pi_row_vec: cytnx.Tensor):
        """ Update state after adding row pivot_i from Pi_mat[p]. """
        logger.debug(f"_add_pivot_row_at(p={p}, pivot_i={pivot_i})")
        try:
            # 1. Get the multi-index corresponding to Pi matrix row index pivot_i
            pi_i_multi_idx = self.Pi_mat[p].Iset.from_int()[pivot_i]
            logger.debug(f"  Multi-index for Pi row {pivot_i}: {pi_i_multi_idx}")

            # 2. Add this multi-index to the global Iset for the *next* bond (p+1)
            self.Iset[p+1].push_back(pi_i_multi_idx)
            logger.debug(f"  Updated Iset[{p+1}] size: {len(self.Iset[p+1])}")

            # 3. Add the row vector to CrossData (using MODIFIED signature)
            self.cross[p].addPivotRow(pivot_i, pi_row_vec)

            # 4. Update T3[p+1] and P[p] based on the new cross[p] state
            self._update_T3P_from_cross(p)

            # 5. Update rows of Pi matrix at bond p+1 (if exists)
            self._update_pi_rows_at(p + 1)
        except Exception as e:
            logger.error(f"Error in _add_pivot_row_at(p={p}, pivot_i={pivot_i}): {e}")
            # Decide on error handling: re-raise, log and continue, etc.


    def _add_pivot_col_at(self, p: int, pivot_j: int, pi_col_vec: cytnx.Tensor):
        """ Update state after adding col pivot_j from Pi_mat[p]. """
        logger.debug(f"_add_pivot_col_at(p={p}, pivot_j={pivot_j})")
        try:
            # 1. Get the multi-index corresponding to Pi matrix col index pivot_j
            pi_j_multi_idx = self.Pi_mat[p].Jset.from_int()[pivot_j]
            logger.debug(f"  Multi-index for Pi col {pivot_j}: {pi_j_multi_idx}")

            # 2. Add this multi-index to the global Jset for *this* bond (p)
            self.Jset[p].push_back(pi_j_multi_idx)
            logger.debug(f"  Updated Jset[{p}] size: {len(self.Jset[p])}")

            # 3. Add the column vector to CrossData (using MODIFIED signature)
            self.cross[p].addPivotCol(pivot_j, pi_col_vec)

            # 4. Update T3[p] and P[p] based on the new cross[p] state
            self._update_T3P_from_cross(p) # P[p] gets updated by both row and col adds

            # 5. Update columns of Pi matrix at bond p-1 (if exists)
            self._update_pi_cols_at(p - 1)
        except Exception as e:
            logger.error(f"Error in _add_pivot_col_at(p={p}, pivot_j={pivot_j}): {e}")


    def _update_T3P_from_cross(self, p: int):
        """ Extract/update T3[p], T3[p+1], P[p] from cross[p] """
        logger.debug(f"_update_T3P_from_cross({p})")
        if self.cross[p] is None: return
        rank = self.cross[p].rank()
        if rank == 0: return

        # Update T3[p] from cross[p].C (Shape: [n_row(pi), rank])
        C_mat = self.cross[p].C
        if C_mat is not None:
             try:
                  dim_Ip = len(self.Iset[p])
                  dim_Sp = len(self.localSet[p])
                  # Expected rank for T3[p] is rank of bond p-1 (cols of T3[p])
                  # But C_mat has rank cols. Need rank of bond p (dim_Jp)
                  dim_Jp = len(self.Jset[p]) if p > 0 else 1 # Rank of bond p
                  if C_mat.shape()[0] == dim_Ip * dim_Sp and C_mat.shape()[1] == rank:
                      # T3[p] should have shape [dim_Ip, dim_Sp, dim_Jp (rank of bond p)]
                      # C_mat has shape [dim_Ip*dim_Sp, rank (rank of bond p)]
                      self.T3[p] = C_mat.reshape(dim_Ip, dim_Sp, rank) # C_mat is A[I_pi, J_p]
                      logger.debug(f"  Updated T3[{p}] shape: {self.T3[p].shape()}")
                  else: logger.error(f"  Dim mismatch for T3[{p}]: C rows {C_mat.shape()[0]} vs {dim_Ip * dim_Sp} or C cols {C_mat.shape()[1]} vs {rank}")
             except Exception as e: logger.error(f"  Error reshaping C for T3[{p}]: {e}")

        # Update T3[p+1] from cross[p].R (Shape: [rank, n_col(pi)])
        R_mat = self.cross[p].R
        if R_mat is not None:
             try:
                  dim_Sp1 = len(self.localSet[p+1])
                  dim_Jp1 = len(self.Jset[p+1])
                  if R_mat.shape()[1] == dim_Sp1 * dim_Jp1 and R_mat.shape()[0] == rank:
                       # T3[p+1] has shape [rank(bond p), dim_Sp1, dim_Jp1]
                       self.T3[p+1] = R_mat.reshape(rank, dim_Sp1, dim_Jp1) # R_mat is A[I_{p+1}, J_pi]
                       logger.debug(f"  Updated T3[{p+1}] shape: {self.T3[p+1].shape()}")
                  else: logger.error(f"  Dim mismatch for T3[{p+1}]: R cols {R_mat.shape()[1]} vs {dim_Sp1 * dim_Jp1} or R rows {R_mat.shape()[0]} vs {rank}")
             except Exception as e: logger.error(f"  Error reshaping R for T3[{p+1}]: {e}")

        # Update P[p] from cross[p].pivotMat (Shape: [rank, rank])
        piv_mat = self.cross[p].pivotMat()
        if piv_mat is not None:
            if piv_mat.shape() == [rank, rank]:
                self.P[p] = piv_mat.clone()
                logger.debug(f"  Updated P[{p}] shape: {self.P[p].shape()}")
            else: logger.warning(f"  Pivot matrix P[{p}] shape {piv_mat.shape()} unexpected (expected ({rank},{rank})). Skipping update.")


    def _update_pi_rows_at(self, p: int):
        """ Update Pi_mat[p] and cross[p] when Iset[p] changes. """
        if p < 0 or p >= self.L - 1: return
        logger.debug(f"_update_pi_rows_at(p={p}) -- Rebuilding Pi and CrossData")
        # Rebuild Pi matrix with new Iset[p]
        self.Pi_mat[p] = self._build_pi_at(p)
        # Re-initialize CrossData (loses LU state, less efficient but simpler)
        self.cross[p] = CrossData(self.Pi_mat[p].n_rows, self.Pi_mat[p].n_cols)
        self.cross[p].lu = AdaptiveLU(self.Pi_mat[p].n_rows, self.Pi_mat[p].n_cols)
        # Re-add all current pivots for bond p (Iset[p+1], Jset[p])
        Iset_p1_indices = self.Iset[p+1].from_int()
        Jset_p_indices = self.Jset[p].from_int()
        if len(Iset_p1_indices) != len(Jset_p_indices):
             logger.error(f"Rank mismatch at bond {p} during row update.")
             return
        for rank_idx in range(len(Iset_p1_indices)):
             idx_i = Iset_p1_indices[rank_idx]
             idx_j = Jset_p_indices[rank_idx]
             try:
                  pi_i = self.Pi_mat[p].Iset.pos(idx_i)
                  pi_j = self.Pi_mat[p].Jset.pos(idx_j)
                  # Extract row/col vectors using new Pi_mat
                  pi_row_vals = self.Pi_mat[p].submat([pi_i], list(range(self.Pi_mat[p].n_cols)))
                  pi_col_vals = self.Pi_mat[p].submat(list(range(self.Pi_mat[p].n_rows)), [pi_j])
                  pi_row_vec = cytnx.from_numpy(np.array(pi_row_vals, dtype=np.float64)).reshape(1,-1)
                  pi_col_vec = cytnx.from_numpy(np.array(pi_col_vals, dtype=np.float64)).reshape(-1,1)
                  # Add pivot using the new indices and vectors
                  self.cross[p].addPivotRow(pi_i, pi_row_vec)
                  self.cross[p].addPivotCol(pi_j, pi_col_vec)
             except ValueError: logger.error(f"Could not find pivot multi-index ({idx_i}, {idx_j}) in new Pi_mat[{p}] during row update.")
             except Exception as e: logger.error(f"Error re-adding pivot ({idx_i}, {idx_j}) at bond {p} during row update: {e}")


    def _update_pi_cols_at(self, p: int):
        """ Update Pi_mat[p] and cross[p] when Jset[p+1] changes. """
        if p < 0 or p >= self.L - 1: return
        logger.debug(f"_update_pi_cols_at(p={p}) -- Rebuilding Pi and CrossData")
        # Rebuild Pi matrix with new Jset[p+1]
        self.Pi_mat[p] = self._build_pi_at(p)
        # Re-initialize CrossData
        self.cross[p] = CrossData(self.Pi_mat[p].n_rows, self.Pi_mat[p].n_cols)
        self.cross[p].lu = AdaptiveLU(self.Pi_mat[p].n_rows, self.Pi_mat[p].n_cols)
        # Re-add all current pivots for bond p (Iset[p+1], Jset[p])
        Iset_p1_indices = self.Iset[p+1].from_int()
        Jset_p_indices = self.Jset[p].from_int()
        if len(Iset_p1_indices) != len(Jset_p_indices):
             logger.error(f"Rank mismatch at bond {p} during column update.")
             return
        for rank_idx in range(len(Iset_p1_indices)):
            idx_i = Iset_p1_indices[rank_idx]
            idx_j = Jset_p_indices[rank_idx]
            try:
                pi_i = self.Pi_mat[p].Iset.pos(idx_i)
                pi_j = self.Pi_mat[p].Jset.pos(idx_j)
                # Extract row/col vectors using new Pi_mat
                pi_row_vals = self.Pi_mat[p].submat([pi_i], list(range(self.Pi_mat[p].n_cols)))
                pi_col_vals = self.Pi_mat[p].submat(list(range(self.Pi_mat[p].n_rows)), [pi_j])
                pi_row_vec = cytnx.from_numpy(np.array(pi_row_vals, dtype=np.float64)).reshape(1,-1) # FIXED
                pi_col_vec = cytnx.from_numpy(np.array(pi_col_vals, dtype=np.float64)).reshape(-1,1) # FIXED
                # Add pivot using the new indices and vectors
                self.cross[p].addPivotRow(pi_i, pi_row_vec)
                self.cross[p].addPivotCol(pi_j, pi_col_vec)
            except ValueError: logger.error(f"Could not find pivot multi-index ({idx_i}, {idx_j}) in new Pi_mat[{p}] during col update.")
            except Exception as e: logger.error(f"Error re-adding pivot ({idx_i}, {idx_j}) at bond {p} during col update: {e}")


    def get_tensor_train(self, center: int = -1) -> List[cytnx.Tensor]:
        """ Construct the Tensor Train / MPS representation. """
        if center < 0: center += self.L
        if not (0 <= center < self.L):
             raise ValueError("Invalid center index")
        logger.info(f"Constructing Tensor Train with center at {center}")

        mps: List[Optional[cytnx.Tensor]] = [None] * self.L

        # Calculate P^-1 matrices (pseudoinverse for stability)
        P_inv: List[Optional[cytnx.Tensor]] = [None] * self.L # Store Pinv for bonds 0 to L-1
        for p in range(self.L - 1): # Iterate through bonds 0 to L-2
             bond_rank = self.P[p].shape()[0] if self.P[p] is not None else 0
             logger.debug(f"Processing P[{p}] with shape {self.P[p].shape() if self.P[p] is not None else 'None'} (rank={bond_rank})")
             if self.P[p] is not None and bond_rank > 0:
                  try:
                       # Check if matrix is square and well-conditioned before SVD
                       if self.P[p].shape()[0] != self.P[p].shape()[1]:
                           logger.warning(f"P[{p}] is not square ({self.P[p].shape()}), using pseudoinverse.")
                       U, S, Vt = cytnx.linalg.Svd(self.P[p])
                       # Threshold small singular values
                       tol = 1e-14 * S[0].item() if S.shape()[0] > 0 else 1e-14
                       S_inv_vals = [1.0 / s.item() if s.item() > tol else 0.0 for s in S]
                       S_inv_diag = cytnx.linalg.Diag(cytnx.Tensor(S_inv_vals))
                       # Handle non-square case if necessary: pinv = V @ S_inv.T @ U.T
                       # For square: pinv = V @ S_inv @ U.T
                       P_inv[p] = Vt.Conj().Transpose() @ S_inv_diag @ U.Conj().Transpose()
                       logger.debug(f"Calculated P_inv[{p}] shape {P_inv[p].shape()}")
                  except Exception as e:
                       logger.error(f"Error calculating inverse for P[{p}]: {e}")
                       P_inv[p] = None
             else:
                  logger.debug(f"P[{p}] is None or empty, skipping inverse.")

        # Construct MPS tensors M_p
        for p in range(self.L):
            T3_p = self.T3[p]
            if T3_p is None:
                logger.warning(f"T3[{p}] is None, cannot construct MPS tensor.")
                continue

            rp, sp, rp1 = T3_p.shape() # Ranks: r_p = rank(bond p-1), r_{p+1} = rank(bond p)

            if p == center:
                mps[p] = T3_p.clone()
                logger.debug(f"Set mps[{p}] (center) shape {mps[p].shape()}")
            elif p < center:
                # M[p] = T3[p] @ P[p]^-1 (Result shape: rp, sp, rp1_new=rp1)
                Pinv_p = P_inv[p] # Use P_inv for bond p
                if Pinv_p is not None and Pinv_p.shape() == [rp1, rp1]:
                     try:
                          # Reshape T3[p]: [rp, sp, rp1] -> [rp * sp, rp1]
                          T3p_mat = T3_p.reshape(rp * sp, rp1)
                          Mp_mat = T3p_mat @ Pinv_p # [rp * sp, rp1]
                          mps[p] = Mp_mat.reshape(rp, sp, rp1)
                          logger.debug(f"Constructed mps[{p}] (< center) shape {mps[p].shape()}")
                     except Exception as e: logger.error(f"Error constructing mps[{p}] (< center): {e}")
                else: logger.warning(f"Skipping mps[{p}] (< center): P_inv[{p}] invalid (shape {Pinv_p.shape() if Pinv_p else 'None'})")
            else: # p > center
                # M[p] = P[p-1]^-1 @ T3[p] (Result shape: rp_new=rp, sp, rp1)
                Pinv_prev = P_inv[p-1] # Use P_inv for bond p-1
                if Pinv_prev is not None and Pinv_prev.shape() == [rp, rp]:
                     try:
                          # Reshape T3[p]: [rp, sp, rp1] -> [rp, sp * rp1]
                          T3p_mat = T3_p.reshape(rp, sp * rp1)
                          Mp_mat = Pinv_prev @ T3p_mat # [rp, sp * rp1]
                          mps[p] = Mp_mat.reshape(rp, sp, rp1)
                          logger.debug(f"Constructed mps[{p}] (> center) shape {mps[p].shape()}")
                     except Exception as e: logger.error(f"Error constructing mps[{p}] (> center): {e}")
                else: logger.warning(f"Skipping mps[{p}] (> center): P_inv[{p-1}] invalid (shape {Pinv_prev.shape() if Pinv_prev else 'None'})")

        # Return only successfully created tensors
        final_mps = [m for m in mps if m is not None]
        if len(final_mps) != self.L:
            logger.warning(f"Tensor train construction incomplete: {len(final_mps)}/{self.L} tensors created.")
        return final_mps
if __name__ == "__main__":
    import time

    # --- Simple Test Case: Product function ---
    # f(x0, x1, x2) = (x0+1)*(x1+1)*(x2+1)

    # Set logging level to INFO to reduce output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Apply level to the logger used in the class

    L = 3 # Number of sites
    D = 2 # Local dimension (0 or 1) -> small domain
    local_dims = [D] * L

    FUNC_CALL_COUNT = 0
    def simple_product_func(idx: MultiIndex) -> float:
        """ Calculates (x0+1)*(x1+1)*(x2+1) """
        global FUNC_CALL_COUNT
        FUNC_CALL_COUNT += 1
        # Ensure idx is tuple of ints
        try:
            idx_int = tuple(int(i[0]) if isinstance(i, tuple) else int(i) for i in idx)
            if len(idx_int) != L: raise ValueError("Index length mismatch")
        except Exception as e:
            logger.error(f"Invalid index format: {idx}. Error: {e}")
            return 0.0
        val = 1.0
        for i in idx_int:
            val *= (i + 1.0)
        return float(val)

    logger.info(f"Testing Simple Product Function: L={L}, D={D}")

    # --- TCI Parameters ---
    # <<< Set pi_matrix_dense to False for efficiency! >>>
    params = TensorCIParam(
        n_iter = 5, # Fewer iterations needed for simple function
        reltol = 1e-10,
        pivot1 = [0] * L, # Start with (0,0,0)
        pivot_finder_param = PivotFinderParam(full_piv=False, n_rook_iter=3),
        pi_matrix_dense = False # <<< Use Lazy Evaluation!
    )

    # --- Run TCI ---
    logger.info("\n--- Running TCI ---")
    start_time = time.time()
    try:
        tci = TensorCI(simple_product_func, local_dims, params)
        mps_tensors = tci.get_tensor_train(center=L//2)
    except Exception as e:
        logger.error(f"TCI Run failed: {e}", exc_info=True)
        mps_tensors = []
    end_time = time.time()
    logger.info(f"TCI Run completed in {end_time - start_time:.3f} seconds.")
    logger.info(f"Function call count: {FUNC_CALL_COUNT}")
    logger.info(f"Final Max Error per Iteration: {tci.pivot_error}")

    # --- Basic Verification ---
    logger.info("\n--- Verification ---")
    final_ranks = []
    if mps_tensors and len(mps_tensors) == L:
        final_ranks = [mps_tensors[p].shape()[2] for p in range(L - 1)]
        logger.info(f"Final Bond dimensions: {final_ranks}")

        # Check one value, e.g., f(1,1,1) = (1+1)*(1+1)*(1+1) = 8
        test_idx = tuple([D - 1] * L) # e.g., (1, 1, 1) if D=2
        true_val = simple_product_func(test_idx)
        logger.info(f"Checking index {test_idx}, True value = {true_val}")

        try:
            # Contract MPS
            vec = cytnx.ones(1, dtype=mps_tensors[0].dtype(), device=mps_tensors[0].device())
            for p in range(L):
                M = mps_tensors[p]
                idx_p = test_idx[p]
                if not (0 <= idx_p < M.shape()[1]):
                     raise IndexError(f"Physical index {idx_p} out of bounds for M[{p}] shape {M.shape()}")
                # Contract env [r_p] with M[:, idx_p, :] [r_p, r_{p+1}]
                vec = vec.reshape(1, -1) @ M[:, idx_p, :]
                vec = vec.reshape(-1)
            approx_val = vec.item()
            logger.info(f"MPS approximated value = {approx_val:.8f}")
            if abs(true_val) > 1e-14:
                rel_err = abs(true_val - approx_val) / abs(true_val)
                logger.info(f"Relative Error = {rel_err:.4e}")
            else:
                 logger.info(f"Absolute Error = {abs(true_val - approx_val):.4e}")

        except Exception as e:
             logger.error(f"Error during MPS contraction for verification: {e}", exc_info=True)
    else:
        logger.error("MPS construction failed or incomplete. Cannot verify.")
