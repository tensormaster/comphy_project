# In tensor_ci.py

import cytnx
from cytnx import UniTensor, Tensor, Bond, BD_IN, BD_OUT, linalg, Type, Device
from typing import List, Callable, Optional, Any, Tuple
import itertools
import logging
import numpy as np # For initial pivot1 handling
import dataclasses

# Assuming other classes are importable
from tensor_train import TensorTrain
from IndexSet import IndexSet # Your Python IndexSet
from crossdata import CrossData # Your Python CrossData (uses your AdaptiveLU)
from matrix_interface import IMatrix, MatLazyIndex # Your Python matrix_interface
from pivot_finder import PivotFinder, PivotFinderParam, PivotData # Your Python pivot_finder
from tensorfuc import TensorFunction # Your Python TensorFunction
from AdaptiveLU import AdaptiveLU # Your Python AdaptiveLU

# Import helper functions if they are in a separate file
from tensor_utils import cube_as_matrix1, cube_as_matrix2, mat_AB1, mat_A1B

logger = logging.getLogger(__name__)
MultiIndex = Tuple[int, ...] # Define MultiIndex if not already globally available

@dataclasses.dataclass
class TensorCI1Param: # Keep your existing definition
    nIter: int = 0
    reltol: float = 1e-12
    pivot1: Optional[List[int]] = None # This is List[int] in C++, usually one full multi-index
    fullPiv: bool = False
    nRookIter: int = 5
    weight: Optional[List[List[float]]] = None
    cond: Optional[Callable[[List[int]], bool]] = None # cond takes a full multi-index
    useCachedFunction: bool = True # Corresponds to TensorFunction's use_cache

    def to_pivot_finder_param(self, 
                              tensor_ci_instance: 'TensorCI1', 
                              current_bond_p: int) -> PivotFinderParam:
        adapted_f_bool = None
        if self.cond is not None:
            def f_bool_for_finder(r_idx_int: int, c_idx_int: int) -> bool:
                # This closure needs access to the Iset_lists and Jset_lists
                # of the *Pi matrix* for bond p, not self.Iset_lists of TensorCI1 directly.
                # Pi_mat[p].Iset (from buildPiAt) and Pi_mat[p].Jset are needed.
                # Let's assume Pi_mat[p] (a MatLazyIndex) has .Iset and .Jset attributes
                # that store the full list of multi-indices for its rows and columns.
                pi_matrix_view = tensor_ci_instance.mat_lazy_Pi_at[current_bond_p]
                
                if r_idx_int >= len(pi_matrix_view.Iset_list_internal) or \
                   c_idx_int >= len(pi_matrix_view.Jset_list_internal):
                    return False 

                multi_idx_I_for_Pi_row = pi_matrix_view.Iset_list_internal[r_idx_int]
                multi_idx_J_for_Pi_col = pi_matrix_view.Jset_list_internal[c_idx_int]
                
                full_multi_idx_for_fc = multi_idx_I_for_Pi_row + multi_idx_J_for_Pi_col
                return self.cond(list(full_multi_idx_for_fc))
            adapted_f_bool = f_bool_for_finder
        
        # Weight handling can be added later if tt_sum logic is implemented
        return PivotFinderParam(
            full_piv=self.fullPiv,
            n_rook_iter=self.nRookIter,
            f_bool=adapted_f_bool 
        )

class TensorCI1:
    def __init__(self,
                 fc: TensorFunction,
                 phys_dims: List[int], 
                 param: Optional[TensorCI1Param] = None,
                 dtype: int = cytnx.Type.Double,
                 device: int = cytnx.Device.cpu):

        self.fc = fc
        self.phys_dims = phys_dims
        self.D = len(phys_dims) 
        self.param = param if param is not None else TensorCI1Param()
        self.dtype = dtype
        self.device = device
        self.dMax = 0 # Placeholder, dMax is used by compressLU or TT construction, not directly in CI param

        if self.param and hasattr(self.param, 'dMax_tt') and self.param.dMax_tt > 0: # Example if dMax was in param
            self.dMax = self.param.dMax_tt
        # Or, if dMax is a global setting for the TCI process controlled elsewhere:
        # self.dMax = some_global_dmax_setting (passed to __init__ or set via a method)

        if self.D < 1:
            raise ValueError("Number of sites (len(phys_dims)) must be at least 1.")
        
        # 1. Handle initial pivot (param.pivot1)
        self.initial_pivot_multi_index: Tuple[int, ...]
        if self.param.pivot1 is None or not self.param.pivot1:
            self.initial_pivot_multi_index = tuple([0] * self.D)
        else:
            if len(self.param.pivot1) != self.D:
                raise ValueError(f"Length of pivot1 must match number of sites D.")
            self.initial_pivot_multi_index = tuple(self.param.pivot1)

        initial_pivot_value = self.fc(self.initial_pivot_multi_index)
        if abs(initial_pivot_value) < 1e-15:
            raise ValueError(f"Value of f at initial pivot {self.initial_pivot_multi_index} is ~zero. Provide a better pivot.")

        # 2. Initialize Iset_lists_param, Jset_lists_param for defining PiAt matrices
        # These correspond to C++ Iset[p] and Jset[p+1] (for PiAt(p)) and are based on initial_pivot_multi_index.
        # For PiAt(p): uses Iset[p] (sites 0..p-1) and Jset[p+1] (sites p+2..D-1)
        self.param_Iset_for_Pi: List[List[MultiIndex]] = [[] for _ in range(self.D)] 
        self.param_Jset_for_Pi: List[List[MultiIndex]] = [[] for _ in range(self.D)] 

        for p_site in range(self.D): # p_site is the site index
            self.param_Iset_for_Pi[p_site] = [self.initial_pivot_multi_index[:p_site]] # For PiAt(p_site), left part is sites 0..p_site-1
            if p_site + 1 < self.D: # For PiAt(p_site), right part uses Jset[p_site+1] in C++ (sites (p_site+1)+1 .. D-1)
                 self.param_Jset_for_Pi[p_site+1] = [self.initial_pivot_multi_index[p_site+2:]]
            # param_Jset_for_Pi[0] is not used for PiAt. param_Jset_for_Pi[D-1] will be [()]

        # Initialize member lists
        num_bonds = self.D - 1 if self.D > 0 else 0
        self.mat_lazy_Pi_at: List[MatLazyIndex] = [None] * num_bonds
        self.P_cross_data: List[CrossData] = [None] * num_bonds
        self.tt = TensorTrain() 
        self.tt.M = [None] * self.D 
        self.P_pivot_matrices: List[Tensor] = [None] * self.D 
        self.pivFinder: List[PivotFinder] = [None] * num_bonds
        self.errorDecay: List[float] = []
        self.done: bool = False


        # 3. Create Pi matrices, CrossData objects, and add the first pivot for each bond
        for p_bond in range(num_bonds): # p_bond from 0 to D-2
            # Row multi-indices for Pi_mat[p_bond]: kron(param_Iset_for_Pi[p_bond], phys_dims[p_bond])
            pi_rows_multi_indices: List[MultiIndex] = []
            for i_idx_multi in self.param_Iset_for_Pi[p_bond]: 
                for phys_val_site_p_bond in range(self.phys_dims[p_bond]):
                    pi_rows_multi_indices.append(i_idx_multi + (phys_val_site_p_bond,))
            
            # Col multi-indices for Pi_mat[p_bond]: kron(phys_dims[p_bond+1], param_Jset_for_Pi[p_bond+1])
            pi_cols_multi_indices: List[MultiIndex] = []
            # Jset for bond p_bond corresponds to sites p_bond+2 onwards, used with site p_bond+1
            # param_Jset_for_Pi[p_bond+1] contains [initial_pivot_multi_index[p_bond+2:]]
            jset_to_use_for_pi_cols = self.param_Jset_for_Pi[p_bond+1] if p_bond+1 < self.D else [()]

            for phys_val_site_p_bond_plus_1 in range(self.phys_dims[p_bond+1]):
                for j_idx_multi in jset_to_use_for_pi_cols:
                    pi_cols_multi_indices.append( (phys_val_site_p_bond_plus_1,) + j_idx_multi )
            
            def fc_for_Pi_p_factory(p_bnd_captured): 
                def fc_for_Pi_p(pi_row_mi: MultiIndex, pi_col_mi: MultiIndex) -> float:
                    full_mi = pi_row_mi + pi_col_mi
                    return self.fc(full_mi)
                return fc_for_Pi_p

            self.mat_lazy_Pi_at[p_bond] = MatLazyIndex(fc_for_Pi_p_factory(p_bond), 
                                                       pi_rows_multi_indices, 
                                                       pi_cols_multi_indices)
            # Store for PivotFinderParam.cond adaptation
            self.mat_lazy_Pi_at[p_bond].Iset_list_internal = pi_rows_multi_indices 
            self.mat_lazy_Pi_at[p_bond].Jset_list_internal = pi_cols_multi_indices

            self.P_cross_data[p_bond] = CrossData(
                self.mat_lazy_Pi_at[p_bond].n_rows, 
                self.mat_lazy_Pi_at[p_bond].n_cols
            )
            self.P_cross_data[p_bond].lu = AdaptiveLU(
                self.mat_lazy_Pi_at[p_bond].n_rows,
                self.mat_lazy_Pi_at[p_bond].n_cols,
                tol=self.param.reltol, 
                rankmax=self.dMax if self.dMax > 0 else 0 # Use dMax for AdaptiveLU of CrossData
            )
            
            # Determine the pivot indices in Pi_mat[p_bond] corresponding to initial_pivot_multi_index
            # Pi_mat[p_bond] row_mi: initial_pivot_multi_index[0...p_bond]
            # Pi_mat[p_bond] col_mi: initial_pivot_multi_index[p_bond+1...D-1]
            row_pivot_mi_for_Pi = self.initial_pivot_multi_index[:p_bond+1]
            col_pivot_mi_for_Pi = self.initial_pivot_multi_index[p_bond+1:]
            try:
                pivot_i_in_Pi_p = pi_rows_multi_indices.index(row_pivot_mi_for_Pi)
                pivot_j_in_Pi_p = pi_cols_multi_indices.index(col_pivot_mi_for_Pi)
            except ValueError:
                raise ValueError(f"Initial pivot for bond {p_bond} not constructible in Pi_mat index sets. Logic error.")

            self.P_cross_data[p_bond].addPivot(pivot_i_in_Pi_p, pivot_j_in_Pi_p, self.mat_lazy_Pi_at[p_bond])

        # 4. Initialize self.tt.M (T3 cores) and self.P_pivot_matrices
        # Based on C++: T3[p] from cross[p].C, T3[p+1] from cross[p].R
        # This means M[p] uses C from P_cross_data[p] AND R from P_cross_data[p-1]

        for p_site in range(self.D): # p_site is the site index for M[p_site]
            # Determine chi_left and chi_right for M[p_site]
            # chi_left is rank of bond p_site-1 (from P_cross_data[p_site-1].rank())
            # chi_right is rank of bond p_site (from P_cross_data[p_site].rank())
            # All initial ranks are 1.
            
            chi_left = 1 if p_site == 0 else self.P_cross_data[p_site-1].rank()
            chi_right = 1 if p_site == self.D - 1 else self.P_cross_data[p_site].rank()
            d_current = self.phys_dims[p_site]

            core_data: Tensor
            if p_site == 0: # First core M[0], data from C of P_cross_data[0]
                C0 = self.P_cross_data[0].lu.C # Shape (d_0, 1) because pi_rows_multi_indices for bond 0 had length d0, rank is 1
                if C0 is None: raise RuntimeError("C0 is None in init")
                core_data = C0.reshape(1, d_current, chi_right) 
            else: # Middle or last core M[p_site], data from R of P_cross_data[p_site-1]
                  # R_{p_site-1} has shape (chi_left, d_{p_site} * chi_right_of_R)
                  # where chi_right_of_R is related to the column space of Pi_{p_site-1}
                  # For initial rank 1, R_{p_site-1} from cross[p_site-1] has shape (1, d_{p_site}) if Jset was trivial for next site.
                Rp_prev = self.P_cross_data[p_site-1].lu.R # R from bond p_site-1
                if Rp_prev is None: raise RuntimeError(f"R{p_site-1} is None in init")
                # Rp_prev shape (chi_left, num_cols_Pi_{p-1}).
                # num_cols_Pi_{p-1} = d_current * (chi_right if not last site else 1 for trivial Jset)
                core_data = Rp_prev.reshape(chi_left, d_current, chi_right)

            b_left = Bond(chi_left, BD_IN)
            b_phys = Bond(d_current, BD_OUT)
            b_right = Bond(chi_right, BD_OUT)
            
            lbl_left = "L_bound" if p_site == 0 else f"link{p_site-1}"
            lbl_phys = f"p{p_site}"
            lbl_right = "R_bound" if p_site == self.D - 1 else f"link{p_site}"
            
            self.tt.M[p_site] = UniTensor([b_left, b_phys, b_right], labels=[lbl_left, lbl_phys, lbl_right], rowrank=1, dtype=self.dtype, device=self.device)
            self.tt.M[p_site].put_block(core_data.astype(self.dtype).to(self.device))

            if p_site < num_bonds : # For P_pivot_matrices up to D-2
                piv_mat = self.P_cross_data[p_site].pivotMat()
                self.P_pivot_matrices[p_site] = piv_mat if piv_mat is not None else cytnx.eye(1, dtype=self.dtype, device=self.device)
        
        # Last pivot matrix P[D-1]
        if self.D > 0: # P_pivot_matrices has size D
            self.P_pivot_matrices[self.D-1] = cytnx.eye(1, dtype=self.dtype, device=self.device)
        
        # Initialize PivotFinders
        for p_bond in range(num_bonds):
            finder_param_p = self.param.to_pivot_finder_param(self, p_bond) 
            self.pivFinder[p_bond] = PivotFinder(param=finder_param_p)

        if self.param.nIter > 0 and num_bonds > 0 :
            self.IterateN(self.param.nIter)
            
    def _update_tt_and_P_matrices_after_pivot(self, p_bond: int):
        """
        Helper function to update tt.M[p_bond], tt.M[p_bond+1] (if applicable)
        and P_pivot_matrices[p_bond] after P_cross_data[p_bond].lu has been updated
        with a new pivot.
        This mirrors the C++ logic where T3[p] views cross[p].C and T3[p+1] views cross[p].R.
        """
        C_p = self.P_cross_data[p_bond].lu.C
        R_p = self.P_cross_data[p_bond].lu.R
        current_rank_p = self.P_cross_data[p_bond].rank()

        if C_p is None: # R_p can be None if C_p is None and rank is 0.
            logger.warning(f"Bond {p_bond}: C_p matrix in CrossData is None. Cannot update TT cores M[{p_bond}] comprehensively.")
            # Potentially set M[p_bond] to a small identity or zero tensor of correct shape if rank is known
            # For now, if C_p is None, it implies rank is likely 0 or an error occurred in AdaptiveLU.
            # We might need to handle this by creating zero tensors for TT cores if that's the desired behavior.
            # For this draft, we'll assume C_p and R_p are valid if rank > 0.
            if current_rank_p == 0: # If rank is truly zero
                 chi_left_Mp = 1 if p_bond == 0 else self.tt.M[p_bond].bonds()[0].dim()
                 chi_right_Mp = 1 # Rank is 0 for this bond
                 data_Mp = cytnx.zeros((chi_left_Mp, self.phys_dims[p_bond], chi_right_Mp), dtype=self.dtype, device=self.device)
                 self.tt.M[p_bond].put_block(data_Mp)
                 
                 if p_bond + 1 < self.D:
                    chi_left_Mp1 = chi_right_Mp # = 1
                    chi_right_Mp1 = 1 if (p_bond + 1) == (self.D - 1) else self.tt.M[p_bond+1].bonds()[2].dim()
                    data_Mp1 = cytnx.zeros((chi_left_Mp1, self.phys_dims[p_bond+1], chi_right_Mp1), dtype=self.dtype, device=self.device)
                    self.tt.M[p_bond+1].put_block(data_Mp1)
            # Fall through to update P_pivot_matrices
            
        else: # C_p is not None, rank_p > 0
            # --- Update self.tt.M[p_bond] (corresponds to T3[p] in C++) ---
            # Data from C_matrix of P_cross_data[p_bond]
            # Expected shape: (chi_left, d_current, chi_right_current_bond)
            chi_left_Mp = self.tt.M[p_bond].bonds()[0].dim() # Existing left bond dimension
            d_current_Mp = self.phys_dims[p_bond]
            # New right bond dimension for M[p_bond] is the new rank of P_cross_data[p_bond]
            new_chi_right_Mp = current_rank_p 

            # C_p has shape (num_rows_Pi_p, rank_p). 
            # num_rows_Pi_p should be chi_left_Mp * d_current_Mp
            if C_p.shape()[0] != chi_left_Mp * d_current_Mp:
                logger.error(f"Update M[{p_bond}]: C_p rows {C_p.shape()[0]} mismatch. Expected {chi_left_Mp * d_current_Mp}. "
                             f"(chi_left={chi_left_Mp}, d={d_current_Mp})")
                # This indicates a potential issue in how dimensions are tracked or reshaped.
                # For now, proceed with reshape if total elements match, otherwise error.
                if C_p.shape()[0] * C_p.shape()[1] != chi_left_Mp * d_current_Mp * new_chi_right_Mp:
                     raise ValueError("Fatal dimension mismatch for C_p when updating M[p_bond].")

            core_data_Mp = C_p.reshape(chi_left_Mp, d_current_Mp, new_chi_right_Mp)
            
            # Update existing M[p_bond] UniTensor
            self.tt.M[p_bond].bonds()[2].assign(Bond(new_chi_right_Mp, BD_OUT)) # Update right bond
            self.tt.M[p_bond].set_labels([self.tt.M[p_bond].labels()[0], self.tt.M[p_bond].labels()[1], f"link{p_bond}"]) # Update label
            self.tt.M[p_bond].put_block(core_data_Mp.astype(self.dtype).to(self.device))

            # --- Update self.tt.M[p_bond+1] (corresponds to T3[p+1] in C++) ---
            # Data from R_matrix of P_cross_data[p_bond]
            if p_bond + 1 < self.D:
                if R_p is None:
                    raise RuntimeError(f"R_p is None for bond {p_bond} when trying to update M[{p_bond+1}]")

                chi_left_Mp1 = new_chi_right_Mp # New left bond for M[p+1] is new_chi_right_Mp
                d_current_Mp1 = self.phys_dims[p_bond+1]
                # Right bond of M[p+1] is not changed by update of bond p_bond,
                # unless it's the very last bond, then it's 1.
                chi_right_Mp1 = self.tt.M[p_bond+1].bonds()[2].dim() 

                # R_p has shape (rank_p, num_cols_Pi_p).
                # num_cols_Pi_p should be d_current_Mp1 * chi_right_Mp1
                if R_p.shape()[1] != d_current_Mp1 * chi_right_Mp1:
                    logger.error(f"Update M[{p_bond+1}]: R_p cols {R_p.shape()[1]} mismatch. Expected {d_current_Mp1 * chi_right_Mp1}. "
                                 f"(d={d_current_Mp1}, chi_right={chi_right_Mp1})")
                    if R_p.shape()[0] * R_p.shape()[1] != chi_left_Mp1 * d_current_Mp1 * chi_right_Mp1:
                        raise ValueError("Fatal dimension mismatch for R_p when updating M[p_bond+1].")

                core_data_Mp1 = R_p.reshape(chi_left_Mp1, d_current_Mp1, chi_right_Mp1)

                # Update existing M[p_bond+1] UniTensor
                self.tt.M[p_bond+1].bonds()[0].assign(Bond(chi_left_Mp1, BD_IN)) # Update left bond
                self.tt.M[p_bond+1].set_labels([f"link{p_bond}", self.tt.M[p_bond+1].labels()[1], self.tt.M[p_bond+1].labels()[2]]) # Update label
                self.tt.M[p_bond+1].put_block(core_data_Mp1.astype(self.dtype).to(self.device))

        # Update Pivot Matrix P_k (square matrix from CrossData)
        pivot_mat_p = self.P_cross_data[p_bond].pivotMat()
        self.P_pivot_matrices[p_bond] = pivot_mat_p if pivot_mat_p is not None \
                                       else cytnx.eye(current_rank_p if current_rank_p > 0 else 1, 
                                                      dtype=self.dtype, device=self.device)


    def IterateN(self, n_iterations: int, update_M_data: bool = True): # Renamed from Iterate to IterateN
        """
        Performs n_iterations of the TCI procedure (half-sweeps or full sweeps).
        C++ iterate(n) does n half-sweeps.
        """
        logger.info(f"Starting {n_iterations} TCI iteration steps (half-sweeps).")
        num_bonds = self.D - 1
        if num_bonds < 0 : # D=0 or D=1 (no bonds)
            logger.warning("No bonds to iterate over.")
            return 0.0

        max_error_overall = 0.0
        
        for iter_count in range(n_iterations):
            max_error_this_half_sweep = 0.0
            # Example: Simple left-to-right half-sweep for adding pivots
            # C++ iterate() seems to do half-sweeps, alternating directions if called multiple times.
            # For now, let's do a fixed direction half-sweep.
            # The C++ TensorCI1::iterate(int nSweep) does nSweep/2 full sweeps (L->R then R->L).
            # Let's make IterateN do n_iterations of full sweeps for simplicity here.
            
            current_sweep = iter_count + 1
            logger.info(f"--- TCI Full Sweep: {current_sweep}/{n_iterations} ---")

            # Left-to-right sweep (updates Jset[p], T3[p] from C[p], T3[p+1] from R[p])
            logger.debug(f"Sweep {current_sweep}: Left-to-Right")
            for p_bond in range(num_bonds): # 0 to D-2
                logger.debug(f"  Processing bond L->R: {p_bond}")
                # This corresponds to C++ addPivotColAt(p_bond, new_pivot_col_idx_in_Pi)
                # 1. Find pivot
                pi_matrix_view = self.mat_lazy_Pi_at[p_bond]
                pivot_finder_p = self.pivFinder[p_bond]
                # PivotFinder needs to know available rows/cols in Pi_matrix_view
                # based on P_cross_data[p_bond].lu.Iset/Jset for Pi_matrix_view.
                # This is complex as PivotFinder takes IMatrix and CrossData.
                # The IMatrix here is pi_matrix_view, CrossData is P_cross_data[p_bond].
                pivot_result: PivotData = pivot_finder_p(pi_matrix_view, self.P_cross_data[p_bond])

                if pivot_result.i == -1 or pivot_result.j == -1 or abs(pivot_result.error) < self.param.reltol: # No good pivot or error too small
                    logger.debug(f"  Bond {p_bond}: No suitable new pivot found or error too small ({pivot_result.error:.2e}).")
                    continue
                
                max_error_this_half_sweep = max(max_error_this_half_sweep, abs(pivot_result.error))
                self.errorDecay.append(abs(pivot_result.error))

                # 2. Add pivot (this updates P_cross_data[p_bond].lu.C, .R, .Iset, .Jset, .rank)
                # addPivot takes integer indices for pi_matrix_view
                self.P_cross_data[p_bond].addPivot(pivot_result.i, pivot_result.j, pi_matrix_view)
                logger.debug(f"  Bond {p_bond}: Added pivot ({pivot_result.i}, {pivot_result.j}), error={pivot_result.error:.2e}, new rank={self.P_cross_data[p_bond].rank()}")


                # 3. Update self.tt.M and self.P_pivot_matrices
                if update_M_data:
                    self._update_tt_and_P_matrices_after_pivot(p_bond)

                # 4. TODO: updatePiColsAt(p_bond-1) if p_bond > 0
                # This step updates the Pi matrix and CrossData of the *previous* bond
                # to reflect the change in basis/rank of the current bond P_cross_data[p_bond].
                # This is complex as it requires changing MatLazyIndex's Jset_list_internal
                # and then potentially re-doing parts of CrossData[p_bond-1]'s CI.
                # if p_bond > 0: self._update_Pi_and_CrossData_cols_at(p_bond - 1, self.P_cross_data[p_bond])


            # Right-to-left sweep (updates Iset[p+1], T3[p+1] from C[p], T3[p] from R[p])
            logger.debug(f"Sweep {current_sweep}: Right-to-Left")
            for p_bond in range(num_bonds - 1, -1, -1): # D-2 down to 0
                logger.debug(f"  Processing bond R->L: {p_bond}")
                # This corresponds to C++ addPivotRowAt(p_bond, new_pivot_row_idx_in_Pi)
                pi_matrix_view = self.mat_lazy_Pi_at[p_bond]
                pivot_finder_p = self.pivFinder[p_bond]
                pivot_result: PivotData = pivot_finder_p(pi_matrix_view, self.P_cross_data[p_bond])

                if pivot_result.i == -1 or pivot_result.j == -1 or abs(pivot_result.error) < self.param.reltol:
                    logger.debug(f"  Bond {p_bond}: No suitable new pivot found or error too small ({pivot_result.error:.2e}).")
                    continue
                
                max_error_this_half_sweep = max(max_error_this_half_sweep, abs(pivot_result.error))
                # Error decay might double count if pivot is same, but tracks max error per search
                # self.errorDecay.append(abs(pivot_result.error)) 

                self.P_cross_data[p_bond].addPivot(pivot_result.i, pivot_result.j, pi_matrix_view)
                logger.debug(f"  Bond {p_bond}: Added pivot ({pivot_result.i}, {pivot_result.j}), error={pivot_result.error:.2e}, new rank={self.P_cross_data[p_bond].rank()}")


                if update_M_data:
                    self._update_tt_and_P_matrices_after_pivot(p_bond)
                
                # 4. TODO: updatePiRowsAt(p_bond+1) if p_bond < num_bonds -1
                # if p_bond < num_bonds - 1: self._update_Pi_and_CrossData_rows_at(p_bond + 1, self.P_cross_data[p_bond])

            max_error_overall = max_error_this_half_sweep # Or keep track of overall max error differently
            if max_error_this_half_sweep < self.param.reltol : # Global stopping for all bonds
                logger.info(f"Converged after {current_sweep} full sweeps. Max error in sweep: {max_error_this_half_sweep:.2e}")
                self.done = True
                break
        
        return max_error_overall if self.errorDecay else 0.0
    def _update_tt_M_and_P_pivot_matrix_after_crossdata_update(self, p_bond: int):
        """
        Updates self.tt.M[p_bond], self.tt.M[p_bond+1] (data and bonds),
        and self.P_pivot_matrices[p_bond] after self.P_cross_data[p_bond].lu
        has been updated with a new pivot.

        This reflects the C++ logic where T3[p] views cross[p].C and T3[p+1] views cross[p].R.
        """
        current_cross_data = self.P_cross_data[p_bond]
        if current_cross_data.lu.C is None or current_cross_data.lu.R is None:
            logger.warning(f"Bond {p_bond}: C or R matrix in CrossData.lu is None. TT core update might be incomplete.")
            # If C or R is None, rank is likely 0.
            # We still need to ensure TT cores have correct (trivial) bond dimensions.
            # For M[p_bond]:
            chi_left_Mp = self.tt.M[p_bond].bonds()[0].dim()
            d_Mp = self.phys_dims[p_bond]
            new_chi_right_Mp = current_cross_data.rank() # Will be 0 if C is None

            self.tt.M[p_bond].bonds()[2].assign(Bond(new_chi_right_Mp, BD_OUT))
            # Ensure labels are consistent, especially the connecting "link" label
            self.tt.M[p_bond].set_labels([
                self.tt.M[p_bond].labels()[0], 
                self.tt.M[p_bond].labels()[1], 
                f"link{p_bond}"
            ])
            # Create and put a zero block if data is missing
            zero_block_Mp = cytnx.zeros((chi_left_Mp, d_Mp, new_chi_right_Mp), dtype=self.dtype, device=self.device)
            self.tt.M[p_bond].put_block(zero_block_Mp)

            if p_bond + 1 < self.D:
                # For M[p_bond+1]:
                chi_left_Mp1 = new_chi_right_Mp # Left bond matches new right bond of M[p_bond]
                d_Mp1 = self.phys_dims[p_bond+1]
                chi_right_Mp1 = self.tt.M[p_bond+1].bonds()[2].dim() # Right bond of M[p+1] remains for now

                self.tt.M[p_bond+1].bonds()[0].assign(Bond(chi_left_Mp1, BD_IN))
                self.tt.M[p_bond+1].set_labels([
                    f"link{p_bond}", 
                    self.tt.M[p_bond+1].labels()[1], 
                    self.tt.M[p_bond+1].labels()[2]
                ])
                zero_block_Mp1 = cytnx.zeros((chi_left_Mp1, d_Mp1, chi_right_Mp1), dtype=self.dtype, device=self.device)
                self.tt.M[p_bond+1].put_block(zero_block_Mp1)
            
            # Update pivot matrix
            self.P_pivot_matrices[p_bond] = cytnx.eye(new_chi_right_Mp if new_chi_right_Mp > 0 else 1, 
                                                      dtype=self.dtype, device=self.device)
            return

        # Proceed if C_p and R_p are available
        C_p = current_cross_data.lu.C
        R_p = current_cross_data.lu.R
        new_rank_p = current_cross_data.rank() # New rank of bond p

        # --- Update self.tt.M[p_bond] (corresponds to T3[p] in C++) ---
        # Data from C_matrix of P_cross_data[p_bond]
        # M[p_bond] current shape: (chi_left_old, d_p, chi_right_old)
        chi_left_Mp = self.tt.M[p_bond].bonds()[0].dim()
        d_p = self.phys_dims[p_bond]
        
        # C_p is (num_rows_Pi_p, new_rank_p). 
        # num_rows_Pi_p = chi_left_Mp * d_p according to C++ T3[p] = Cube(C.mem, Isize, localsize, Jsize)
        # where Isize is chi_left_Mp, localsize is d_p, Jsize is new_rank_p.
        if C_p.shape()[0] != chi_left_Mp * d_p or C_p.shape()[1] != new_rank_p:
            logger.error(f"Update M[{p_bond}]: C_p shape {C_p.shape()} mismatch. "
                         f"Expected ({chi_left_Mp * d_p}, {new_rank_p}). "
                         f"(chi_left={chi_left_Mp}, d={d_p}, new_rank_p={new_rank_p})")
            # Fallback or error handling if dimensions are severely mismatched
            # This might indicate an issue in AdaptiveLU's C matrix construction or dim tracking
            # For now, we will try to reshape but it might fail if numel doesn't match
            if C_p.shape()[0] * C_p.shape()[1] != chi_left_Mp * d_p * new_rank_p:
                 raise ValueError(f"Fatal dimension mismatch for C_p when updating M[{p_bond}]. Cannot reshape.")
        
        core_data_Mp = C_p.reshape(chi_left_Mp, d_p, new_rank_p)
        
        self.tt.M[p_bond].bonds()[2].assign(Bond(new_rank_p, BD_OUT)) # Update right bond dim
        self.tt.M[p_bond].set_labels([self.tt.M[p_bond].labels()[0], self.tt.M[p_bond].labels()[1], f"link{p_bond}"])
        self.tt.M[p_bond].put_block(core_data_Mp.astype(self.dtype).to(self.device))

        # --- Update self.tt.M[p_bond+1] (corresponds to T3[p+1] in C++) ---
        # Data from R_matrix of P_cross_data[p_bond]
        if p_bond + 1 < self.D:
            new_chi_left_Mp1 = new_rank_p # New left bond for M[p+1]
            d_Mp1 = self.phys_dims[p_bond+1]
            chi_right_Mp1_old = self.tt.M[p_bond+1].bonds()[2].dim() # Existing right bond dim

            # R_p has shape (new_rank_p, num_cols_Pi_p).
            # num_cols_Pi_p = d_Mp1 * chi_right_Mp1_old (if Jset for Pi_p was trivial for far right sites)
            # More generally, num_cols_Pi_p = len(self.mat_lazy_Pi_at[p_bond].Jset_list_internal)
            if R_p.shape()[0] != new_chi_left_Mp1 or \
               R_p.shape()[1] != len(self.mat_lazy_Pi_at[p_bond].Jset_list_internal) :
                logger.error(f"Update M[{p_bond+1}]: R_p shape {R_p.shape()} mismatch. "
                             f"Expected ({new_chi_left_Mp1}, {len(self.mat_lazy_Pi_at[p_bond].Jset_list_internal)}).")
                if R_p.shape()[0] * R_p.shape()[1] != new_chi_left_Mp1 * d_Mp1 * chi_right_Mp1_old:
                     raise ValueError(f"Fatal dimension mismatch for R_p when updating M[{p_bond+1}]. Cannot reshape.")
            
            # The reshape must result in (new_chi_left_Mp1, d_Mp1, chi_right_Mp1_old)
            # This requires R_p.shape()[1] == d_Mp1 * chi_right_Mp1_old.
            # This implies that the column space of Pi_mat[p_bond] (which R_p's columns span)
            # must correctly factorize into phys_dims[p_bond+1] and the right virtual bond of M[p_bond+1].
            # This is automatically true if Jset_list_internal for Pi_mat[p_bond] was kron(phys_dims[p+1], J_param_part)
            
            core_data_Mp1 = R_p.reshape(new_chi_left_Mp1, d_Mp1, chi_right_Mp1_old)

            self.tt.M[p_bond+1].bonds()[0].assign(Bond(new_chi_left_Mp1, BD_IN)) # Update left bond
            self.tt.M[p_bond+1].set_labels([f"link{p_bond}", self.tt.M[p_bond+1].labels()[1], self.tt.M[p_bond+1].labels()[2]])
            self.tt.M[p_bond+1].put_block(core_data_Mp1.astype(self.dtype).to(self.device))

        # --- Update Pivot Matrix self.P_pivot_matrices[p_bond] ---
        pivot_mat_p = current_cross_data.pivotMat() # From CrossData instance
        if pivot_mat_p is None:
            logger.warning(f"pivotMat() for bond {p_bond} returned None. Using identity.")
            self.P_pivot_matrices[p_bond] = cytnx.eye(new_rank_p if new_rank_p > 0 else 1, 
                                                      dtype=self.dtype, device=self.device)
        else:
            if pivot_mat_p.shape() != [new_rank_p, new_rank_p]:
                logger.warning(f"Pivot matrix P[{p_bond}] shape {pivot_mat_p.shape()} does not match new rank ({new_rank_p},{new_rank_p}).")
                # Fallback or resize? For now, use as is if not None.
            self.P_pivot_matrices[p_bond] = pivot_mat_p.astype(self.dtype).to(self.device)


    def _update_adjacent_Pi_and_CrossData_cols(self, p_bond_to_update_Pi: int, p_bond_source_of_Jset_change: int):
        """
        Updates Pi_mat[p_bond_to_update_Pi] and P_cross_data[p_bond_to_update_Pi]
        because Jset for its columns (derived from site p_bond_to_update_Pi+1, which uses
        Iset from bond p_bond_source_of_Jset_change) has changed.
        Corresponds to C++ updatePiColsAt(p_prev) after Jset[p_curr] changed at cross[p_curr].
        p_bond_to_update_Pi = p_curr - 1
        p_bond_source_of_Jset_change = p_curr
        """
        if p_bond_to_update_Pi < 0:
            return

        logger.debug(f"Updating Pi_mat[{p_bond_to_update_Pi}] (cols) due to changes at bond {p_bond_source_of_Jset_change}")

        # The columns of Pi_mat[p_bond_to_update_Pi] are kron(phys_dims[p_bond_to_update_Pi+1], J_param_for_Pi_cols)
        # J_param_for_Pi_cols was self.param_Jset_for_Pi_col_part[p_bond_to_update_Pi+1]
        # This Jset corresponds to sites (p_bond_to_update_Pi+1)+1 ... D-1.
        # This part does NOT directly depend on the *rank* (Iset/Jset) of P_cross_data[p_bond_source_of_Jset_change].
        # The C++ logic seems more intricate:
        # Pi.setCols(xfac::kron(localSet[p+1], Jset[p+1]));
        # cross[p].setCols(cube_as_matrix1(T3[p+1]), Q);
        # Here T3[p+1] (our M[p+1]) itself has changed rank due to cross[p]'s update.
        # This means the *basis* for Jset[p+1] (relative to cross[p]) has changed.

        # This needs careful thought. The `setCols` in C++ IMatrix takes a new set of MultiIndex.
        # If the *rank* of bond `p_bond_source_of_Jset_change` changes, it means the basis
        # for the multi-indices `self.Iset_lists_ci[p_bond_source_of_Jset_change+1]` changes.
        # This `Iset_lists_ci` defines the *rows* of `Pi_mat[p_bond_source_of_Jset_change]`.
        # This is subtle. For now, this is a placeholder. The primary effect of rank change
        # is on the dimensions of TT cores, handled by _update_tt_M...
        # If this method is about re-defining the Pi matrix itself, it's more involved.
        pass


    def _update_adjacent_Pi_and_CrossData_rows(self, p_bond_to_update_Pi: int, p_bond_source_of_Iset_change: int):
        """
        Updates Pi_mat[p_bond_to_update_Pi] and P_cross_data[p_bond_to_update_Pi]
        because Iset for its rows has changed.
        Corresponds to C++ updatePiRowsAt(p_next) after Iset[p_curr+1] changed at cross[p_curr].
        p_bond_to_update_Pi = p_curr + 1
        p_bond_source_of_Iset_change = p_curr
        """
        if p_bond_to_update_Pi >= len(self.mat_lazy_Pi_at) or p_bond_to_update_Pi < 0 :
            return
        logger.debug(f"Updating Pi_mat[{p_bond_to_update_Pi}] (rows) due to changes at bond {p_bond_source_of_Iset_change}")
        # Similar complexity to _update_adjacent_Pi_and_CrossData_cols
        pass


    def iterate_one_full_sweep(self, update_M_data: bool = True) -> float:
        max_err_sweep = 0.0
        num_bonds = self.D - 1
        if num_bonds < 0 : return 0.0

        # Left-to-right sweep
        logger.debug("Iterate: Left-to-Right Sweep")
        for p_bond in range(num_bonds): 
            logger.debug(f"  L->R Processing bond {p_bond}")
            pi_matrix_view = self.mat_lazy_Pi_at[p_bond]
            current_cross_data_p = self.P_cross_data[p_bond]
            pivot_finder_p = self.pivFinder[p_bond]
            
            pivot_result: PivotData = pivot_finder_p(pi_matrix_view, current_cross_data_p)

            if pivot_result.i != -1 and pivot_result.j != -1 and \
               abs(pivot_result.error) >= self.param.reltol * (max(self.errorDecay) if self.errorDecay else 1.0):
                
                max_err_sweep = max(max_err_sweep, abs(pivot_result.error))
                self.errorDecay.append(abs(pivot_result.error))

                self.P_cross_data[p_bond].addPivot(pivot_result.i, pivot_result.j, pi_matrix_view)
                logger.debug(f"    Bond {p_bond} (L->R): Added pivot ({pivot_result.i}, {pivot_result.j}), "
                             f"err={pivot_result.error:.2e}, new rank={self.P_cross_data[p_bond].rank()}")

                if update_M_data:
                    self._update_tt_M_and_P_pivot_matrix_after_crossdata_update(p_bond)
                
                # Update Pi_mat[p_bond-1].cols and cross[p_bond-1].cols
                # Jset of P_cross_data[p_bond] is related to Iset of Pi_mat[p_bond] (its rows)
                # This is tricky. The C++ uses T3[p+1] which is M[p_bond+1] to update cross[p_bond].setCols
                # The change at bond p_bond affects the *basis/rank* of the link between site p_bond and p_bond+1.
                # This new basis for "link p_bond" is what Pi_mat[p_bond-1] needs for its column definition
                # (as its columns involve site p_bond).
                if p_bond > 0:
                     self._update_adjacent_Pi_and_CrossData_cols(p_bond_to_update_Pi=p_bond - 1, 
                                                                 p_bond_source_of_Jset_change=p_bond)
            else:
                 logger.debug(f"    Bond {p_bond} (L->R): No suitable new pivot or error {pivot_result.error:.2e} too small.")
        
        # Right-to-left sweep
        logger.debug("Iterate: Right-to-Left Sweep")
        for p_bond in range(num_bonds - 1, -1, -1): 
            logger.debug(f"  R->L Processing bond {p_bond}")
            pi_matrix_view = self.mat_lazy_Pi_at[p_bond]
            current_cross_data_p = self.P_cross_data[p_bond]
            pivot_finder_p = self.pivFinder[p_bond]
            
            pivot_result: PivotData = pivot_finder_p(pi_matrix_view, current_cross_data_p)

            if pivot_result.i != -1 and pivot_result.j != -1 and \
               abs(pivot_result.error) >= self.param.reltol * (max(self.errorDecay) if self.errorDecay else 1.0):
                
                max_err_sweep = max(max_err_sweep, abs(pivot_result.error))
                self.errorDecay.append(abs(pivot_result.error))

                self.P_cross_data[p_bond].addPivot(pivot_result.i, pivot_result.j, pi_matrix_view)
                logger.debug(f"    Bond {p_bond} (R->L): Added pivot ({pivot_result.i}, {pivot_result.j}), "
                             f"err={pivot_result.error:.2e}, new rank={self.P_cross_data[p_bond].rank()}")

                if update_M_data:
                    self._update_tt_M_and_P_pivot_matrix_after_crossdata_update(p_bond)
                
                # Update Pi_mat[p_bond+1].rows and cross[p_bond+1].rows
                # Iset of P_cross_data[p_bond] is related to Jset of Pi_mat[p_bond] (its cols)
                # The change at bond p_bond affects the basis for "link p_bond".
                # This new basis for "link p_bond" is what Pi_mat[p_bond+1] needs for its row definition.
                if p_bond < num_bonds - 1:
                     self._update_adjacent_Pi_and_CrossData_rows(p_bond_to_update_Pi=p_bond + 1,
                                                                 p_bond_source_of_Iset_change=p_bond)
            else:
                logger.debug(f"    Bond {p_bond} (R->L): No suitable new pivot or error {pivot_result.error:.2e} too small.")

        if max_err_sweep < self.param.reltol * (max(self.errorDecay) if self.errorDecay else 1.0):
            logger.info(f"TCI converged after sweep. Max error in sweep: {max_err_sweep:.2e}")
            self.done = True
        
        return max_err_sweep
    

    def get_canonical_tt(self, center: int) -> TensorTrain:
        # ... (Implementation will use mat_AB1, mat_A1B, cube_as_matrix1/2) ...
        # This method should create a DEEP COPY of self.tt and self.P_pivot_matrices
        # and then operate on the copy.
        if not self.tt.M or not all(core is not None for core in self.tt.M):
            raise RuntimeError("TensorTrain self.tt.M is not fully initialized before calling get_canonical_tt.")
        if not all(p_mat is not None for p_mat in self.P_pivot_matrices):
            raise RuntimeError("Pivot matrices self.P_pivot_matrices not fully initialized.")

        # Create a deep copy of the TensorTrain to be canonicalized
        # Your TensorTrain class would need a clone() or deepcopy mechanism
        tt_copy = self.tt.clone() # Assuming TensorTrain has a clone method
        
        # P_pivot_matrices_copy = [p.clone() for p in self.P_pivot_matrices] # If they are Tensors
        P_pivot_matrices_copy = []
        for p_mat in self.P_pivot_matrices:
            if isinstance(p_mat, Tensor): P_pivot_matrices_copy.append(p_mat.clone())
            else: P_pivot_matrices_copy.append(p_mat) # Should all be Tensors


        # Left of center: A_p <- A_p P_p^-1
        for p in range(center):
            Ap_matrix = cube_as_matrix2(tt_copy.M[p]) # (chi_L*d, chi_R)
            Pp_inv = self.P_pivot_matrices[p] # This is P_p, mat_AB1 needs B=P_p
            
            # Result of A @ B^-1
            # Ap_matrix is ( (L*d) x R_dim_of_A )
            # Pp_inv is ( R_dim_of_A x R_dim_of_A )
            if Ap_matrix.shape()[1] != Pp_inv.shape()[0] : # Check if R_dim_of_A matches P_p rows
                 raise ValueError(f"Dim mismatch for A P^-1 at site {p}: A_cols={Ap_matrix.shape()[1]}, P_rows={Pp_inv.shape()[0]}")

            new_Ap_matrix_data = mat_AB1(Ap_matrix, Pp_inv) # mat_AB1 computes Ap_matrix @ Inv(Pp_inv)
            
            # Reshape and update core
            chi_L, d, chi_R_old = tt_copy.M[p].shape()
            # The right bond dim chi_R does not change in this operation, P_p acts on right bond space
            new_core_data = new_Ap_matrix_data.reshape(chi_L, d, chi_R_old) 
            tt_copy.M[p].put_block(new_core_data)

            # Absorb P_p into next core M[p+1]: M_{p+1} <- P_p M_{p+1}
            if p + 1 < self.D:
                Mp1_matrix = cube_as_matrix1(tt_copy.M[p+1]) # (chi_L_of_Mp1, (d*chi_R_of_Mp1) )
                                                             # chi_L_of_Mp1 must match Pp_inv.shape()[1] (cols)
                if Pp_inv.shape()[1] != Mp1_matrix.shape()[0]:
                    raise ValueError(f"Dim mismatch for P M at site {p+1}: P_cols={Pp_inv.shape()[1]}, M_rows={Mp1_matrix.shape()[0]}")
                
                new_Mp1_matrix_data = Pp_inv @ Mp1_matrix
                chi_L_new_Mp1, d_Mp1, chi_R_Mp1 = tt_copy.M[p+1].shape() # chi_L_new_Mp1 is new from P
                new_core_data_p1 = new_Mp1_matrix_data.reshape(Pp_inv.shape()[0], d_Mp1, chi_R_Mp1) # Left bond of M[p+1] changes
                
                # Update bonds of M[p+1] if left dim changed
                tt_copy.M[p+1].bonds()[0].assign(Bond(Pp_inv.shape()[0], BD_IN))
                tt_copy.M[p+1].set_labels([f"link{p}", tt_copy.M[p+1].labels()[1], tt_copy.M[p+1].labels()[2]])
                tt_copy.M[p+1].put_block(new_core_data_p1)


        # Right of center: A_p <- P_{p-1}^-1 A_p
        for p in range(self.D - 1, center, -1): # Loop D-1 down to center+1
            Ap_matrix = cube_as_matrix1(tt_copy.M[p]) # (chi_L, d*chi_R)
            Pp_prev_inv = self.P_pivot_matrices[p-1] # This is P_{p-1}
            
            # Result of P_prev^-1 @ A
            if Pp_prev_inv.shape()[1] != Ap_matrix.shape()[0]:
                 raise ValueError(f"Dim mismatch for P^-1 A at site {p}: P_cols={Pp_prev_inv.shape()[1]}, A_rows={Ap_matrix.shape()[0]}")

            new_Ap_matrix_data = mat_A1B(Pp_prev_inv, Ap_matrix)
            
            chi_L_old, d, chi_R = tt_copy.M[p].shape()
            # Left bond dim chi_L changes, P_{p-1} acts on left bond space
            new_core_data = new_Ap_matrix_data.reshape(Pp_prev_inv.shape()[0], d, chi_R) 
            
            tt_copy.M[p].bonds()[0].assign(Bond(Pp_prev_inv.shape()[0], BD_IN))
            tt_copy.M[p].set_labels([f"link{p-1}", tt_copy.M[p].labels()[1], tt_copy.M[p].labels()[2]])
            tt_copy.M[p].put_block(new_core_data)

            # Absorb P_{p-1} into M[p-1]: M_{p-1} <- M_{p-1} P_{p-1}
            if p - 1 >= 0:
                Mp_prev_matrix = cube_as_matrix2(tt_copy.M[p-1]) # (chi_L*d, chi_R_of_Mp-1)
                                                                 # chi_R_of_Mp-1 must match Pp_prev_inv.shape()[0] (rows)
                if Mp_prev_matrix.shape()[1] != Pp_prev_inv.shape()[0]:
                     raise ValueError(f"Dim mismatch for M P at site {p-1}: M_cols={Mp_prev_matrix.shape()[1]}, P_rows={Pp_prev_inv.shape()[0]}")

                new_Mp_prev_matrix_data = Mp_prev_matrix @ Pp_prev_inv
                
                chi_L_Mp_prev, d_Mp_prev, chi_R_old_Mp_prev = tt_copy.M[p-1].shape()
                new_core_data_pm1 = new_Mp_prev_matrix_data.reshape(chi_L_Mp_prev, d_Mp_prev, Pp_prev_inv.shape()[1]) # Right bond of M[p-1] changes

                tt_copy.M[p-1].bonds()[2].assign(Bond(Pp_prev_inv.shape()[1], BD_OUT))
                tt_copy.M[p-1].set_labels([tt_copy.M[p-1].labels()[0], tt_copy.M[p-1].labels()[1], f"link{p-1}"])
                tt_copy.M[p-1].put_block(new_core_data_pm1)
        
        return tt_copy

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- Starting TensorCI1 Update Logic Test ---")

    # --- 1. Setup Mock Objects and Parameters ---
    test_phys_dims = [2, 3, 2] # D = 3 sites
    test_D = len(test_phys_dims)
    test_dtype = cytnx.Type.Double
    test_device = cytnx.Device.cpu

    # Mock TensorFunction
    def mock_fc_func(indices: Tuple[int, ...]) -> float:
        # A simple function for testing, e.g., product of indices + sum
        val = 1.0
        s = 0
        for i, idx in enumerate(indices):
            val *= (idx + 1 + i*0.1) # Make it somewhat non-trivial
            s += idx
        return val + s
    
    mock_tensor_function = TensorFunction(func=mock_fc_func)

    # TensorCI1Param - use a simple one for now
    test_param = TensorCI1Param(pivot1=[0] * test_D, nIter=0) # Start with all-zero pivot

    # --- 2. Instantiate TensorCI1 (this will call __init__) ---
    # __init__ should populate initial P_cross_data, P_pivot_matrices, and tt.M (rank 1)
    try:
        tci_instance = TensorCI1(
            fc=mock_tensor_function,
            phys_dims=test_phys_dims,
            param=test_param,
            dtype=test_dtype,
            device=test_device
        )
        logger.info("__init__ completed.")
        logger.info(f"Initial P_pivot_matrices ranks: {[p.shape if p is not None else None for p in tci_instance.P_pivot_matrices]}")
        for i, core in enumerate(tci_instance.tt.M):
            if core is not None:
                logger.info(f"Initial tt.M[{i}] shape: {core.shape()}, labels: {core.labels()}")
            else:
                logger.warning(f"Initial tt.M[{i}] is None.")

    except Exception as e:
        logger.error(f"Error during TensorCI1 initialization: {e}", exc_info=True)
        # Stop test if init fails
        exit()

    # --- 3. Test _update_tt_M_and_P_matrices_after_crossdata_update ---
    # We need to simulate P_cross_data[p_bond] being updated by adding a pivot.
    
    if test_D > 1:
        p_bond_to_test = 0 # Test update for the first bond
        logger.info(f"\n--- Testing update for bond {p_bond_to_test} ---")

        # Get the CrossData object for this bond
        cross_data_p = tci_instance.P_cross_data[p_bond_to_test]
        pi_matrix_p = tci_instance.mat_lazy_Pi_at[p_bond_to_test]

        # Simulate adding a new pivot to cross_data_p
        # This requires knowing a valid (new_pivot_i, new_pivot_j) in pi_matrix_p
        # For simplicity, let's assume we found a pivot that increases rank to 2.
        # You would normally get this from PivotFinder.
        # Here, we'll manually call addPivot with a known "next" pivot.
        # Note: The actual pivot indices depend on the content of fc and Pi matrix structure.
        # This is a simplified manual pivot addition.
        
        # Example: Assume Pi matrix for bond 0 is at least 2x2 after initial pivot (it won't be yet)
        # To test properly, we need to find a *second* distinct pivot.
        # Let's assume PivotFinder found a new pivot (e.g., row 1, col 1 of Pi_0, if available)
        # The current rank is 1. C is (d0, 1), R is (1, d1*d2_eff)
        
        # Simulate finding a second pivot (this is very manual for testing)
        # In a real scenario, PivotFinder would give these from the *available* indices of Pi_matrix.
        # And the indices are for the *full* Pi matrix.
        # Let's assume initial pivot was (0,0) in Pi_0.
        # If Pi_0 has more rows/cols, pick another, e.g. (1,0) or (0,1) if valid.
        new_pivot_i_in_Pi = 0 # Placeholder - depends on Pi_matrix_p structure
        new_pivot_j_in_Pi = 0 # Placeholder
        
        # Find a non-pivot index:
        if pi_matrix_p.n_rows > 1 and ([0] != cross_data_p.lu.Iset): # if initial pivot was not row 0
             new_pivot_i_in_Pi = 0
        elif pi_matrix_p.n_rows > 1:
             new_pivot_i_in_Pi = 1
        
        if pi_matrix_p.n_cols > 1 and ([0] != cross_data_p.lu.Jset):
             new_pivot_j_in_Pi = 0
        elif pi_matrix_p.n_cols > 1:
             new_pivot_j_in_Pi = 1

        if not (new_pivot_i_in_Pi < pi_matrix_p.n_rows and new_pivot_j_in_Pi < pi_matrix_p.n_cols):
            logger.warning(f"Cannot find a distinct second pivot for bond {p_bond_to_test} in this simple test setup. Skipping update test.")
        else:
            logger.info(f"Simulating adding pivot ({new_pivot_i_in_Pi}, {new_pivot_j_in_Pi}) to CrossData for bond {p_bond_to_test}")
            try:
                cross_data_p.addPivot(new_pivot_i_in_Pi, new_pivot_j_in_Pi, pi_matrix_p)
                new_rank = cross_data_p.rank()
                logger.info(f"CrossData for bond {p_bond_to_test} updated. New rank: {new_rank}")
                if cross_data_p.lu.C is not None:
                    logger.debug(f"  Updated lu.C shape: {cross_data_p.lu.C.shape()}")
                if cross_data_p.lu.R is not None:
                    logger.debug(f"  Updated lu.R shape: {cross_data_p.lu.R.shape()}")

                # Now call the function to update TT cores
                logger.info(f"Calling _update_tt_M_and_P_matrices_after_crossdata_update for bond {p_bond_to_test}")
                tci_instance._update_tt_M_and_P_matrices_after_crossdata_update(p_bond_to_test)

                # Assertions:
                logger.info(f"  M[{p_bond_to_test}] new shape: {tci_instance.tt.M[p_bond_to_test].shape()}, "
                            f"right bond dim: {tci_instance.tt.M[p_bond_to_test].bonds()[2].dim()}")
                assert tci_instance.tt.M[p_bond_to_test].bonds()[2].dim() == new_rank

                if p_bond_to_test + 1 < test_D:
                    logger.info(f"  M[{p_bond_to_test+1}] new shape: {tci_instance.tt.M[p_bond_to_test+1].shape()}, "
                                f"left bond dim: {tci_instance.tt.M[p_bond_to_test+1].bonds()[0].dim()}")
                    assert tci_instance.tt.M[p_bond_to_test+1].bonds()[0].dim() == new_rank
                
                logger.info(f"  P_pivot_matrices[{p_bond_to_test}] new shape: {tci_instance.P_pivot_matrices[p_bond_to_test].shape()}")
                assert tci_instance.P_pivot_matrices[p_bond_to_test].shape() == [new_rank, new_rank]
                logger.info(f"Update logic test for bond {p_bond_to_test} seems plausible based on shapes.")

            except Exception as e:
                logger.error(f"Error during update test for bond {p_bond_to_test}: {e}", exc_info=True)

    logger.info("--- TensorCI1 Update Logic Test Finished ---")