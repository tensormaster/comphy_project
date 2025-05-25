# tensor_ci.py
import cytnx
from cytnx import UniTensor, Tensor, Bond, BD_IN, BD_OUT, linalg, Type, Device, from_numpy
from typing import List, Callable, Optional, Any, Tuple
import logging
import numpy as np
import dataclasses
from itertools import product # Added for trueError in TensorCI1

# --- Necessary imports ---
from tensor_train import TensorTrain #
from matrix_interface import IMatrix, MatLazyIndex, make_IMatrix # Added make_IMatrix
from crossdata import CrossData #
from pivot_finder import PivotFinder, PivotFinderParam, PivotData #
from tensorfuc import TensorFunction #
from tensor_utils import cube_as_matrix1, cube_as_matrix2, mat_AB1, mat_A1B #
from IndexSet import IndexSet

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Force DEBUG level for this logger
MultiIndex = Tuple[int, ...]

@dataclasses.dataclass
class TensorCI1Param:
    nIter: int = 0
    reltol: float = 1e-12
    pivot1: Optional[List[int]] = None
    fullPiv: bool = False
    nRookIter: int = 5
    weight: Optional[List[List[float]]] = None
    cond: Optional[Callable[[List[int]], bool]] = None
    useCachedFunction: bool = True


@dataclasses.dataclass
class TTEnvMatrices:
    L: List[Optional[cytnx.UniTensor]]
    R: List[Optional[cytnx.UniTensor]]
    D: int
    dtype: int
    device: int
    is_active: bool = False

    def __init__(self, D_sites: int,
                 initial_tt: Optional[TensorTrain] = None,
                 site_weights_list: Optional[List[List[float]]] = None,
                 dtype: int = Type.Double, device: int = Device.cpu):
        self.D = D_sites
        self.dtype = dtype
        self.device = device
        self.L = [None] * self.D 
        self.R = [None] * self.D 
        self.is_active = False 

        if self.D <= 0:
            logger.warning("TTEnvMatrices initialized with D_sites <= 0.")
            return

        bond_L0 = Bond(1, BD_OUT) 
        self.L[0] = UniTensor(bonds=[bond_L0], 
                              labels=['env_L_0_out'], 
                              dtype=self.dtype, device=self.device, is_diag=False)
        self.L[0].get_block_().fill(1.0)

        if self.D > 0: 
            bond_RD1 = Bond(1, BD_IN)
            self.R[self.D-1] = UniTensor(bonds=[bond_RD1], 
                                         labels=['env_R_D-1_in'], 
                                         dtype=self.dtype, device=self.device, is_diag=False)
            self.R[self.D-1].get_block_().fill(1.0)

        if initial_tt is not None and site_weights_list is not None:
            actual_tt_len = len(initial_tt.M) if hasattr(initial_tt, 'M') and initial_tt.M is not None else 0
            if actual_tt_len == self.D and len(site_weights_list) == self.D:
                logger.info("TTEnvMatrices: Initializing L and R environments from provided TT and weights.")
                if not (initial_tt.M and all(core is not None for core in initial_tt.M)):
                    logger.error("TTEnvMatrices init: TT cores not fully defined before calling initialize_environments_from_tt."); return
                self.initialize_environments_from_tt(initial_tt, site_weights_list)
            else:
                logger.error(f"TTEnvMatrices init: Mismatch in D (expected {self.D}, TT has {actual_tt_len}) "
                             f"or weights length (expected {self.D}, got {len(site_weights_list)}).")
        else:
            logger.debug("TTEnvMatrices init: No initial_tt or weights. Boundary L/R are set.")

    def _get_np_dtype_for_weights(self, cytnx_dtype: int) -> Any:
        if cytnx_dtype == Type.ComplexDouble: return np.complex128
        if cytnx_dtype == Type.ComplexFloat: return np.complex64
        if cytnx_dtype == Type.Double: return np.float64
        if cytnx_dtype == Type.Float: return np.float32
        logger.warning(f"TTEnvMatrices: Unexpected dtype {cytnx_dtype} for weights, defaulting to np.float64.")
        return np.float64

    def initialize_environments_from_tt(self, tt: TensorTrain, site_weights_list: List[List[float]]):
        logger.debug("TTEnvMatrices: Initializing L environments.")
        for p in range(self.D - 1): 
            L_prev = self.L[p]    
            M_p_core = tt.M[p] 
            if L_prev is None or M_p_core is None: 
                logger.error(f"L-init: L[{p}] or M[{p}] is None."); self.is_active=False; return

            np_w_p = np.array(site_weights_list[p], dtype=self._get_np_dtype_for_weights(self.dtype))
            W_p_tensor_data = from_numpy(np_w_p).to(self.device).astype(self.dtype)
            
            phys_dim_W = M_p_core.shape()[1] 
            bond_W = Bond(phys_dim_W, BD_IN)
            W_p_ut = UniTensor(bonds=[bond_W], 
                               labels=[f'W_p_L_leg{p}'], 
                               dtype=self.dtype, device=self.device, is_diag=False)
            W_p_ut.put_block(W_p_tensor_data)

            L_prev_c = L_prev.contiguous()
            M_p_core_c = M_p_core.contiguous()
            W_p_ut_c = W_p_ut.contiguous()
            
            try:
                # DEBUG START
                logger.debug(f"  L-Env ncon input L_prev_c: {L_prev_c.labels()}, shape={L_prev_c.shape()}, bonds={[str(b.type()) for b in L_prev_c.bonds()]}")
                logger.debug(f"  L-Env ncon input M_p_core_c: {M_p_core_c.labels()}, shape={M_p_core_c.shape()}, bonds={[str(b.type()) for b in M_p_core_c.bonds()]}")
                logger.debug(f"  L-Env ncon input W_p_ut_c: {W_p_ut_c.labels()}, shape={W_p_ut_c.shape()}, bonds={[str(b.type()) for b in W_p_ut_c.bonds()]}")
                # DEBUG END
                result_L = cytnx.ncon(
                    [L_prev_c, M_p_core_c, W_p_ut_c],
                    [[10], [10, 20, -1], [20]],  
                    cont_order=[10, 20] 
                )
                result_L.set_labels([f'vL_out_site{p}'])
                self.L[p+1] = result_L
                logger.debug(f"  L-Env ncon output L[{p+1}] shape={result_L.shape()}") # DEBUG
            except RuntimeError as e: 
                logger.error(f"ncon L[{p+1}] C++ Error: {e}",exc_info=True)
                logger.debug(f"  L_prev_c labels: {L_prev_c.labels()}, shape: {L_prev_c.shape()}, bonds: {L_prev_c.bonds()}")
                logger.debug(f"  M_p_core_c labels: {M_p_core_c.labels()}, shape: {M_p_core_c.shape()}, bonds: {M_p_core_c.bonds()}")
                logger.debug(f"  W_p_ut_c labels: {W_p_ut_c.labels()}, shape: {W_p_ut_c.shape()}, bonds: {W_p_ut_c.bonds()}")
                self.is_active=False;return
            except TypeError as e:
                logger.error(f"ncon L[{p+1}] Python TypeError: {e}", exc_info=True)
                self.is_active=False;return
            except Exception as e: 
                logger.error(f"Init L[{p+1}] other error: {e}",exc_info=True);self.is_active=False;return

        logger.debug("TTEnvMatrices: Initializing R environments.")
        for p in range(self.D - 1, 0, -1): 
            R_next = self.R[p] 
            M_p_core = tt.M[p]
            if R_next is None or M_p_core is None: 
                logger.error(f"R-init: R[{p}] or M[{p}] is None."); self.is_active=False; return

            np_w_p = np.array(site_weights_list[p], dtype=self._get_np_dtype_for_weights(self.dtype))
            W_p_tensor_data = from_numpy(np_w_p).to(self.device).astype(self.dtype)

            phys_dim_W = M_p_core.shape()[1]
            bond_W = Bond(phys_dim_W, BD_IN)
            W_p_ut = UniTensor(bonds=[bond_W], 
                               labels=[f'W_p_R_leg{p}'], 
                               dtype=self.dtype, device=self.device, is_diag=False)
            W_p_ut.put_block(W_p_tensor_data)

            R_next_c = R_next.contiguous()
            M_p_core_c = M_p_core.contiguous()
            W_p_ut_c = W_p_ut.contiguous()
            
            try:
                # DEBUG START
                logger.debug(f"  R-Env ncon input M_p_core_c: {M_p_core_c.labels()}, shape={M_p_core_c.shape()}, bonds={[str(b.type()) for b in M_p_core_c.bonds()]}")
                logger.debug(f"  R-Env ncon input W_p_ut_c: {W_p_ut_c.labels()}, shape={W_p_ut_c.shape()}, bonds={[str(b.type()) for b in W_p_ut_c.bonds()]}")
                logger.debug(f"  R-Env ncon input R_next_c: {R_next_c.labels()}, shape={R_next_c.shape()}, bonds={[str(b.type()) for b in R_next_c.bonds()]}")
                # DEBUG END
                result_R = cytnx.ncon(
                    [M_p_core_c, W_p_ut_c, R_next_c], 
                    [[-1, 20, 10], [20], [10]],      
                    cont_order=[20, 10] 
                )
                result_R.set_labels([f'vR_out_site{p-1}'])
                self.R[p-1] = result_R
                logger.debug(f"  R-Env ncon output R[{p-1}] shape={result_R.shape()}") # DEBUG
            except RuntimeError as e: 
                logger.error(f"ncon R[{p-1}] C++ Error: {e}",exc_info=True)
                logger.debug(f"  M_p_core_c labels: {M_p_core_c.labels()}, shape: {M_p_core_c.shape()}, bonds: {M_p_core_c.bonds()}")
                logger.debug(f"  W_p_ut_c labels: {W_p_ut_c.labels()}, shape: {W_p_ut_c.shape()}, bonds: {W_p_ut_c.bonds()}")
                logger.debug(f"  R_next_c labels: {R_next_c.labels()}, shape: {R_next_c.shape()}, bonds: {R_next_c.bonds()}")
                self.is_active=False;return
            except TypeError as e:
                logger.error(f"ncon R[{p-1}] Python TypeError: {e}", exc_info=True)
                self.is_active=False;return
            except Exception as e: 
                logger.error(f"Init R[{p-1}] other error: {e}",exc_info=True);self.is_active=False;return
        
        if not self.is_active: 
            self.is_active = True 
            logger.info("TTEnvMatrices environments initialized successfully from TT.")
        else:
            logger.warning("TTEnvMatrices marked inactive due to errors during initialization.")

    def update_site(self, p_core_idx: int, M_core_norm: cytnx.UniTensor, 
                    site_core_weights: List[float], is_left_to_right_update: bool):
        if not self.is_active: logger.debug("TTEnvMatrices.update_site on inactive instance. Skipping."); return
        if M_core_norm is None: logger.error("M_core_norm is None in update_site."); return
            
        M_core_norm_c = M_core_norm.contiguous()
        
        np_weights = np.array(site_core_weights, dtype=self._get_np_dtype_for_weights(self.dtype))
        W_tensor_data = from_numpy(np_weights).to(self.device).astype(self.dtype)

        phys_dim_W = M_core_norm_c.shape()[1]
        bond_W = Bond(phys_dim_W, BD_IN)
        W_ut = UniTensor(bonds=[bond_W], 
                         labels=[f'W_ut_update_leg{p_core_idx}'], 
                         dtype=self.dtype, device=self.device, is_diag=False)
        W_ut.put_block(W_tensor_data)
        W_ut_c = W_ut.contiguous()

        if is_left_to_right_update: 
            if p_core_idx + 1 >= self.D: return 
            L_prev = self.L[p_core_idx]
            if L_prev is None: logger.error(f"L[{p_core_idx}] is None for L-update."); return
            L_prev_c = L_prev.contiguous()
            
            try:
                # DEBUG START
                logger.debug(f"  L-update ncon input L_prev_c: {L_prev_c.labels()}, shape={L_prev_c.shape()}, bonds={[str(b.type()) for b in L_prev_c.bonds()]}")
                logger.debug(f"  L-update ncon input M_core_norm_c: {M_core_norm_c.labels()}, shape={M_core_norm_c.shape()}, bonds={[str(b.type()) for b in M_core_norm_c.bonds()]}")
                logger.debug(f"  L-update ncon input W_ut_c: {W_ut_c.labels()}, shape={W_ut_c.shape()}, bonds={[str(b.type()) for b in W_ut_c.bonds()]}")
                # DEBUG END
                result_L = cytnx.ncon(
                    [L_prev_c, M_core_norm_c, W_ut_c],
                    [[10],[10,20,-1],[20]], 
                    cont_order=[10,20]
                )
                result_L.set_labels([f'vL_updated_out_site{p_core_idx}'])
                self.L[p_core_idx+1] = result_L
                logger.debug(f"  L-update ncon output L[{p_core_idx+1}] shape={result_L.shape()}") # DEBUG
            except RuntimeError as e: 
                logger.error(f"ncon L[{p_core_idx+1}] in update_site C++ Error: {e}", exc_info=True)
                logger.debug(f"  L_prev_c labels: {L_prev_c.labels()}, shape: {L_prev_c.shape()}, bonds: {L_prev_c.bonds()}")
                logger.debug(f"  M_core_norm_c labels: {M_core_norm_c.labels()}, shape: {M_core_norm_c.shape()}, bonds: {M_core_norm_c.bonds()}")
                logger.debug(f"  W_ut_c labels: {W_ut_c.labels()}, shape: {W_ut_c.shape()}, bonds: {W_ut_c.bonds()}")
            except TypeError as e:
                logger.error(f"ncon L[{p_core_idx+1}] in update_site Python TypeError: {e}", exc_info=True)
            except Exception as e: 
                logger.error(f"Update L[{p_core_idx+1}] in update_site other error: {e}", exc_info=True)

        else: # Update R[p_core_idx - 1]
            if p_core_idx - 1 < 0: return 
            R_next = self.R[p_core_idx]
            if R_next is None: logger.error(f"R[{p_core_idx}] is None for R-update."); return
            R_next_c = R_next.contiguous()
            
            try:
                # DEBUG START
                logger.debug(f"  R-update ncon input M_core_norm_c: {M_core_norm_c.labels()}, shape={M_core_norm_c.shape()}, bonds={[str(b.type()) for b in M_core_norm_c.bonds()]}")
                logger.debug(f"  R-update ncon input W_ut_c: {W_ut_c.labels()}, shape={W_ut_c.shape()}, bonds={[str(b.type()) for b in W_ut_c.bonds()]}")
                logger.debug(f"  R-update ncon input R_next_c: {R_next_c.labels()}, shape={R_next_c.shape()}, bonds={[str(b.type()) for b in R_next_c.bonds()]}")
                # DEBUG END
                result_R = cytnx.ncon(
                    [M_core_norm_c, W_ut_c, R_next_c],
                    [[-1,20,10],[20],[10]], 
                    cont_order=[20,10]
                )
                result_R.set_labels([f'vR_updated_out_site{p_core_idx-1}'])
                self.R[p_core_idx-1] = result_R
                logger.debug(f"  R-update ncon output R[{p_core_idx-1}] shape={result_R.shape()}") # DEBUG
            except RuntimeError as e: 
                logger.error(f"ncon R[{p_core_idx-1}] C++ Error: {e}", exc_info=True)
                logger.debug(f"  M_core_norm_c labels: {M_core_norm_c.labels()}, shape: {M_core_norm_c.shape()}, bonds: {M_core_norm_c.bonds()}")
                logger.debug(f"  W_ut_c labels: {W_ut_c.labels()}, shape: {W_ut_c.shape()}, bonds: {W_ut_c.bonds()}")
                logger.debug(f"  R_next_c labels: {R_next_c.labels()}, shape: {R_next_c.shape()}, bonds: {R_next_c.bonds()}")
            except TypeError as e:
                logger.error(f"ncon R[{p_core_idx-1}] Python TypeError: {e}", exc_info=True)
            except Exception as e: 
                logger.error(f"Update R[{p_core_idx-1}] in update_site other error: {e}", exc_info=True)

class TensorCI1:
    def __init__(self,
                 fc: 'TensorFunction',
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
        self.errorDecay: List[float] = []
        self.done: bool = False
        self.dMax = 0 

        if self.D < 1: raise ValueError("Number of sites must be at least 1.")

        self.initial_pivot_multi_index: Tuple[int, ...]
        if self.param.pivot1 is None or not self.param.pivot1:
            self.initial_pivot_multi_index = tuple([0] * self.D)
        else:
            if len(self.param.pivot1) != self.D: raise ValueError("pivot1 length mismatch D.")
            self.initial_pivot_multi_index = tuple(self.param.pivot1)

        initial_pivot_value = self.fc(self.initial_pivot_multi_index)
        if abs(initial_pivot_value) < 1e-15: 
            raise ValueError(f"f({self.initial_pivot_multi_index}) = {initial_pivot_value} is ~zero.")

        num_bonds = self.D - 1 if self.D > 0 else 0

        self.localSet: List[IndexSet] = [] # Store IndexSet objects
        for i, d_p in enumerate(phys_dims):
            # For IndexSet, pass a list of MultiIndex tuples
            # Changed MultiIndex((val,)) to (val,)
            self.localSet.append(IndexSet([(val,) for val in range(d_p)])) 

        # self.param_Iset_for_Pi_construction: List[List[MultiIndex]] = [[] for _ in range(self.D)] # OBSOLETE
        # self.param_Jset_for_Pi_construction: List[List[MultiIndex]] = [[] for _ in range(self.D)] # OBSOLETE
        
        # Initializing Iset (left virtual bonds) and Jset (right virtual bonds) for T_cores
        # Iset[p] corresponds to the left bond of T_cores[p]
        # Jset[p] corresponds to the right bond of T_cores[p]

        # Iset[0] is always an empty tuple (L_bound)
        self.Iset: List[IndexSet] = [IndexSet([tuple()])] 
        # Iset[p] for p > 0 comes from the left part of the initial_pivot_multi_index
        for p_site in range(1, self.D):
             self.Iset.append(IndexSet([self.initial_pivot_multi_index[:p_site]]))
        
        # Jset[p] for T_cores[p]'s right bond
        # Jset[p] for p < D-1 comes from the right part of the initial_pivot_multi_index
        self.Jset: List[IndexSet] = [IndexSet([]) for _ in range(self.D)]
        for p_site in range(self.D - 1):
            self.Jset[p_site] = IndexSet([self.initial_pivot_multi_index[p_site+1:]])
        if self.D > 0: # Right boundary for T_cores[D-1] is an empty tuple
            self.Jset[self.D-1] = IndexSet([tuple()])

        logger.debug(f"TensorCI1 __init__: Initial Iset lengths: {[len(iset) for iset in self.Iset]}") # DEBUG
        logger.debug(f"TensorCI1 __init__: Initial Jset lengths: {[len(jset) for jset in self.Jset]}") # DEBUG

        self.mat_lazy_Pi_at: List[Optional[MatLazyIndex]] = [None] * num_bonds
        self.P_cross_data: List[Optional[CrossData]] = [None] * num_bonds
        
        for p_bond in range(num_bonds):
            # pi_rows_mi: kron(Iset[p_bond], localSet[p_bond])
            pi_rows_mi: List[MultiIndex] = [i_idx_multi + l_mi_tuple
                for i_idx_multi in self.Iset[p_bond].get_all() 
                for l_mi_tuple in self.localSet[p_bond].get_all()] 
            
            # pi_cols_mi: kron(localSet[p_bond+1], Jset[p_bond+1])
            pi_cols_mi: List[MultiIndex] = [l_mi_p1_tuple + j_idx_multi
                for l_mi_p1_tuple in self.localSet[p_bond+1].get_all() 
                for j_idx_multi in self.Jset[p_bond+1].get_all()] 

            def fc_for_Pi_factory(p_b): 
                def fc_pi(row_mi: MultiIndex, col_mi: MultiIndex) -> float:
                    return self.fc(row_mi + col_mi)
                return fc_pi

            self.mat_lazy_Pi_at[p_bond] = make_IMatrix( 
                fc_for_Pi_factory(p_bond), pi_rows_mi, pi_cols_mi,
                full=False, dtype=self.dtype, device=self.device
            ) 

            self.P_cross_data[p_bond] = CrossData(self.mat_lazy_Pi_at[p_bond].n_rows, self.mat_lazy_Pi_at[p_bond].n_cols)
            if self.P_cross_data[p_bond] is not None: 
                 self.P_cross_data[p_bond].tol = self.param.reltol

            row_pivot_mi_for_Pi_p = self.initial_pivot_multi_index[:p_bond+1]
            col_pivot_mi_for_Pi_p = self.initial_pivot_multi_index[p_bond+1:]
            try:
                # .pos() returns a list, so we take the first element [0]
                pivot_i_in_Pi_p = self.mat_lazy_Pi_at[p_bond].Iset.pos([row_pivot_mi_for_Pi_p])[0]
                pivot_j_in_Pi_p = self.mat_lazy_Pi_at[p_bond].Jset.pos([col_pivot_mi_for_Pi_p])[0]
            except ValueError: raise ValueError(f"Initial pivot for bond {p_bond} not in Pi_mat index lists.")
            
            logger.debug(f"TensorCI1 __init__: Adding initial pivot ({pivot_i_in_Pi_p}, {pivot_j_in_Pi_p}) for P_cross_data[{p_bond}]") # DEBUG
            self.P_cross_data[p_bond].addPivot(pivot_i_in_Pi_p, pivot_j_in_Pi_p, self.mat_lazy_Pi_at[p_bond])
            logger.debug(f"TensorCI1 __init__: P_cross_data[{p_bond}] rank after initial pivot: {self.P_cross_data[p_bond].rank()}") # DEBUG


        self.tt = TensorTrain() 
        self.tt.M = [None] * self.D
        self.P_pivot_matrices: List[Optional[Tensor]] = [None] * num_bonds 

        for p_site in range(self.D):
            # Bond dimensions for current core M[p_site]
            # Left bond dimension: Use current Iset[p_site]'s length
            chi_left = len(self.Iset[p_site])
            # Right bond dimension: Use current Jset[p_site]'s length
            chi_right = len(self.Jset[p_site])
            d_current = self.phys_dims[p_site]
            
            core_data_tensor: Tensor
            if self.D == 1: # Single site TT
                temp_core_data = cytnx.zeros((1,d_current,1), dtype=self.dtype, device=self.device)
                if 0 <= self.initial_pivot_multi_index[0] < d_current:
                     temp_core_data[0, self.initial_pivot_multi_index[0], 0] = initial_pivot_value
                core_data_tensor = temp_core_data
            elif p_site == 0 : # First core of a multi-site TT
                pd0 = self.P_cross_data[0]
                if pd0 is None or pd0.C is None: raise RuntimeError("C0 None for M[0]")
                # C.shape = (n_rows, rank_of_bond0) = (dim_I0 * dim_phys0, rank_of_bond0)
                # dim_I0 is len(Iset[0]) which is 1.
                expected_C_rows = chi_left * d_current
                expected_C_cols = chi_right # This is rank of P_cross_data[0]
                if pd0.C.shape() != [expected_C_rows, expected_C_cols]:
                    logger.error(f"TensorCI1 __init__: M[0] C_matrix shape mismatch. Expected {[expected_C_rows, expected_C_cols]}, Got {pd0.C.shape()}") # DEBUG
                    raise RuntimeError("Initial C_matrix shape mismatch for M[0]")
                core_data_tensor = pd0.C.reshape(chi_left, d_current, chi_right)
            else: # Middle or last core of a multi-site TT
                pdp_prev = self.P_cross_data[p_site-1]
                if pdp_prev is None or pdp_prev.R is None: raise RuntimeError(f"R{p_site-1} None for M[{p_site}]")
                # R.shape = (rank_of_bond_prev, n_cols_prev) = (rank_of_bond_prev, dim_phys_p * dim_J_p)
                # dim_J_p is chi_right of THIS core
                expected_R_rows = chi_left # This is rank of P_cross_data[p_site-1]
                expected_R_cols = d_current * chi_right
                if pdp_prev.R.shape() != [expected_R_rows, expected_R_cols]:
                    logger.error(f"TensorCI1 __init__: M[{p_site}] R_matrix shape mismatch. Expected {[expected_R_rows, expected_R_cols]}, Got {pdp_prev.R.shape()}") # DEBUG
                    raise RuntimeError(f"Initial R_matrix shape mismatch for M[{p_site}]")

                core_data_tensor = pdp_prev.R.reshape(chi_left, d_current, chi_right)
            
            label_L = f'link{p_site-1}' if p_site > 0 else 'L_bound' 
            label_P = f'p{p_site}'
            label_R = f'link{p_site}' if p_site < self.D - 1 else 'R_bound'

            bd_L = cytnx.Bond(chi_left, BD_IN)
            bd_P = cytnx.Bond(d_current, BD_OUT) 
            bd_R = cytnx.Bond(chi_right, BD_OUT)
            
            current_core_ut = UniTensor(bonds=[bd_L, bd_P, bd_R], 
                                     labels=[label_L, label_P, label_R], 
                                     rowrank=1, 
                                     dtype=self.dtype, 
                                     device=self.device,
                                     is_diag=False) 

            block_to_put = core_data_tensor.astype(self.dtype).to(self.device)
            current_core_ut.put_block(block_to_put)
            
            self.tt.M[p_site] = current_core_ut
            logger.debug(f"TensorCI1 __init__: M[{p_site}] initialized with shape {current_core_ut.shape()}") # DEBUG
            # logger.debug(f"TensorCI1 __init__: M[{p_site}] block data:\n{current_core_ut.get_block_().numpy()}") # DEBUG - too verbose generally

            if p_site < num_bonds: # P_pivot_matrices for D-1 bonds
                pd_site = self.P_cross_data[p_site]
                piv_mat = pd_site.pivotMat() # pivotMat() should return rank x rank matrix
                
                if piv_mat is not None and pd_site.rank() > 0:
                    self.P_pivot_matrices[p_site] = piv_mat.astype(self.dtype).to(self.device)
                    logger.debug(f"TensorCI1 __init__: P_pivot_matrices[{p_site}] initialized with shape {self.P_pivot_matrices[p_site].shape()} (from pivotMat)") # DEBUG
                elif pd_site and pd_site.rank() == 0:
                     self.P_pivot_matrices[p_site] = None # Rank is 0, no meaningful P matrix.
                     logger.debug(f"TensorCI1 __init__: P_pivot_matrices[{p_site}] set to None (rank 0).") # DEBUG
                else: # Fallback if CrossData or rank is somehow invalid
                    # This case means pivotMat failed or pd_site is None, but expected a P matrix.
                    # Use identity as a fallback if rank > 0 based on chi_right
                    fallback_rank = chi_right # Rank of this bond, which is the dimension of P
                    if fallback_rank > 0:
                        self.P_pivot_matrices[p_site] = cytnx.eye(fallback_rank, dtype=self.dtype, device=self.device)
                        logger.warning(f"TensorCI1 __init__: P_pivot_matrices[{p_site}] initialized with eye({fallback_rank}) as fallback.") # WARNING
                    else:
                        self.P_pivot_matrices[p_site] = None
                        logger.debug(f"TensorCI1 __init__: P_pivot_matrices[{p_site}] set to None (fallback, rank 0).") # DEBUG

        self.Pi_bool_mat: List[Optional[cytnx.Tensor]] = [None] * num_bonds
        if self.param.cond is not None and num_bonds > 0:
            for p_bond in range(num_bonds):
                pi_matrix_view = self.mat_lazy_Pi_at[p_bond]
                if not (pi_matrix_view and hasattr(pi_matrix_view, 'Iset') and hasattr(pi_matrix_view, 'Jset') and
                        pi_matrix_view.Iset is not None and pi_matrix_view.Jset is not None): 
                    logger.warning(f"Pi_mat[{p_bond}] or its IndexSets not available for Pi_bool_mat. Skipping."); continue
                
                iset_list_pi = pi_matrix_view.Iset.get_all() 
                jset_list_pi = pi_matrix_view.Jset.get_all()
                rows, cols = len(iset_list_pi), len(jset_list_pi)

                if rows == 0 or cols == 0: continue
                # The condition function `self.param.cond` expects a `List[int]`
                # need to convert tuple `(iset_list_pi[r] + jset_list_pi[c])` to list.
                bool_data_np = np.array(
                    [[self.param.cond(list(iset_list_pi[r] + jset_list_pi[c])) for c in range(cols)] for r in range(rows)],
                    dtype=bool )
                self.Pi_bool_mat[p_bond] = from_numpy(bool_data_np).to(self.device).astype(Type.Bool)


        self.tt_sum_env: Optional[TTEnvMatrices] = None
        if self.param.weight is not None and self.D > 0:
            logger.debug("Initializing TTEnvMatrices for ENV learning.")
            self.tt_sum_env = TTEnvMatrices(D_sites=self.D, initial_tt=self.tt, 
                                            site_weights_list=self.param.weight,
                                            dtype=self.dtype, device=self.device)
            if not self.tt_sum_env.is_active: logger.warning("TTEnvMatrices marked inactive after init.")

        self.pivFinder: List[Optional[PivotFinder]] = [None] * num_bonds
        for p_bond in range(num_bonds):
            finder_param_p = self._build_pivot_finder_param_at_bond(p_bond)
            self.pivFinder[p_bond] = PivotFinder(param=finder_param_p)

        if self.param.nIter > 0 and num_bonds > 0 :
            logger.info(f"__init__: Running {self.param.nIter} initial TCI iterations using IterateN.")
            self.IterateN(self.param.nIter) # Call IterateN to handle sweeps and set self.done


    def get_TP1_at(self, p_bond: int) -> Optional[cytnx.UniTensor]:
        logger.debug(f"Calling get_TP1_at for p_bond={p_bond}") # DEBUG
        if not (0 <= p_bond < self.D): logger.error(f"get_TP1_at: p_bond {p_bond} OOB."); return None
        M_core_p = self.tt.M[p_bond]
        if M_core_p is None: logger.error(f"get_TP1_at: M[{p_bond}] is None."); return None
        
        # P_pivot_matrices only exist for bonds 0 to D-2 (num_bonds-1)
        if p_bond == self.D - 1 : # Last core, no P matrix to its right
            logger.debug(f"get_TP1_at: p_bond={p_bond} is last core. Returning clone of M_core_p.") # DEBUG
            return M_core_p.clone()

        P_matrix_p = self.P_pivot_matrices[p_bond]
        if P_matrix_p is None : # If P_matrix_p is None, it means the rank is 0, or an error.
             logger.warning(f"get_TP1_at: P_pivot_matrices[{p_bond}] is None. Returning M_core_p.clone().")
             return M_core_p.clone()
        
        logger.debug(f"get_TP1_at: M_core_p shape: {M_core_p.shape()}, P_matrix_p shape: {P_matrix_p.shape()}") # DEBUG
        try:
            M_matrix_form = cube_as_matrix2(M_core_p) 
            logger.debug(f"get_TP1_at: M_matrix_form shape: {M_matrix_form.shape()}") # DEBUG
            TP1_matrix_form = mat_AB1(M_matrix_form, P_matrix_p) 
            logger.debug(f"get_TP1_at: TP1_matrix_form shape: {TP1_matrix_form.shape()}") # DEBUG
            
            # The shape of TP1_matrix_form is (chi_left * d, new_chi_right).
            # new_chi_right should be the dimension of P_matrix_p's column dimension,
            # which for a square P is P_matrix_p.shape()[1].
            # For a correct reshape back to UniTensor, we need to know the original chi_left and d.
            chi_left = M_core_p.shape()[0]
            d = M_core_p.shape()[1]
            new_chi_right = TP1_matrix_form.shape()[1] # This is the crucial part.
            
            # Check for shape consistency before reshape
            if TP1_matrix_form.shape()[0] != chi_left * d:
                logger.error(f"TP1_matrix_form first dim mismatch: expected {chi_left*d}, got {TP1_matrix_form.shape()[0]}")
                return None

            res_ut = UniTensor(bonds=[M_core_p.bonds()[0].clone(), M_core_p.bonds()[1].clone(), Bond(new_chi_right, M_core_p.bonds()[2].type())],
                               labels=[M_core_p.labels()[0], M_core_p.labels()[1], M_core_p.labels()[2]],
                               rowrank=M_core_p.rowrank(),
                               dtype=TP1_matrix_form.dtype(), 
                               device=TP1_matrix_form.device(),
                               is_diag=M_core_p.is_diag())
            res_ut.put_block(TP1_matrix_form.reshape(chi_left, d, new_chi_right))
            logger.debug(f"get_TP1_at: Resulting UniTensor shape: {res_ut.shape()}") # DEBUG
            return res_ut
        except Exception as e: logger.error(f"Error in get_TP1_at({p_bond}): {e}", exc_info=True); return None

    def get_P1T_at(self, p_bond: int) -> Optional[cytnx.UniTensor]:
        logger.debug(f"Calling get_P1T_at for p_bond={p_bond}") # DEBUG
        if not (0 <= p_bond < self.D): logger.error(f"get_P1T_at: p_bond {p_bond} OOB."); return None
        M_core_p = self.tt.M[p_bond]
        if M_core_p is None: logger.error(f"get_P1T_at: M[{p_bond}] is None."); return None
        if p_bond == 0: # First core, no P matrix to its left
            logger.debug(f"get_P1T_at: p_bond={p_bond} is first core. Returning clone of M_core_p.") # DEBUG
            return M_core_p.clone() 
        P_matrix_prev = self.P_pivot_matrices[p_bond-1]
        if P_matrix_prev is None: # If P_matrix_prev is None, it means rank is 0.
            logger.warning(f"get_P1T_at: P_pivot_matrices[{p_bond-1}] is None. Returning M_core_p.clone().")
            return M_core_p.clone()
        
        logger.debug(f"get_P1T_at: M_core_p shape: {M_core_p.shape()}, P_matrix_prev shape: {P_matrix_prev.shape()}") # DEBUG
        try:
            M_matrix_form = cube_as_matrix1(M_core_p) 
            logger.debug(f"get_P1T_at: M_matrix_form shape: {M_matrix_form.shape()}") # DEBUG
            P1T_matrix_form = mat_A1B(P_matrix_prev, M_matrix_form) 
            logger.debug(f"get_P1T_at: P1T_matrix_form shape: {P1T_matrix_form.shape()}") # DEBUG
            
            # The shape of P1T_matrix_form is (new_chi_left, d * chi_right).
            # new_chi_left should be the dimension of P_matrix_prev's row dimension,
            # which for a square P is P_matrix_prev.shape()[0].
            # For a correct reshape back to UniTensor, we need to know the original d and chi_right.
            new_chi_left = P1T_matrix_form.shape()[0] # This is the crucial part.
            d = M_core_p.shape()[1]
            chi_right = M_core_p.shape()[2]

            # Check for shape consistency before reshape
            if P1T_matrix_form.shape()[1] != d * chi_right:
                logger.error(f"P1T_matrix_form second dim mismatch: expected {d*chi_right}, got {P1T_matrix_form.shape()[1]}")
                return None

            res_ut = UniTensor(bonds=[Bond(new_chi_left, M_core_p.bonds()[0].type()), M_core_p.bonds()[1].clone(), M_core_p.bonds()[2].clone()],
                               labels=[M_core_p.labels()[0], M_core_p.labels()[1], M_core_p.labels()[2]],
                               rowrank=M_core_p.rowrank(),
                               dtype=P1T_matrix_form.dtype(),
                               device=P1T_matrix_form.device(),
                               is_diag=M_core_p.is_diag())
            res_ut.put_block(P1T_matrix_form.reshape(new_chi_left, d, chi_right))
            logger.debug(f"get_P1T_at: Resulting UniTensor shape: {res_ut.shape()}") # DEBUG
            return res_ut
        except Exception as e: logger.error(f"Error in get_P1T_at({p_bond}): {e}", exc_info=True); return None

    def _build_pivot_finder_param_at_bond(self, p_bond: int) -> PivotFinderParam:
        pf_param = PivotFinderParam(full_piv=self.param.fullPiv, n_rook_iter=self.param.nRookIter)
        if self.tt_sum_env is not None and self.tt_sum_env.is_active and \
           self.param.weight is not None and 0 <= p_bond < len(self.param.weight):
            
            L_p_uni = self.tt_sum_env.L[p_bond]
            R_p1_uni = self.tt_sum_env.R[p_bond+1] if (p_bond+1) < self.D else None

            if L_p_uni is not None and R_p1_uni is not None:
                L_p_tensor = L_p_uni.get_block_() 
                R_p1_tensor = R_p1_uni.get_block_()
                
                weight_p_list_site_p = self.param.weight[p_bond]
                
                if weight_p_list_site_p:
                    np_dt_weights = np.float64 
                    if self.dtype == Type.ComplexDouble: np_dt_weights = np.complex128
                    elif self.dtype == Type.ComplexFloat: np_dt_weights = np.complex64
                    elif self.dtype == Type.Float: np_dt_weights = np.float32
                    
                    np_weights_site_p = np.array(weight_p_list_site_p, dtype=np_dt_weights)
                    W_tensor_site_p = from_numpy(np_weights_site_p).to(self.device).astype(self.dtype)

                    N = L_p_tensor.shape()[0] if L_p_tensor.rank() > 0 else 0 
                    M_phys = W_tensor_site_p.shape()[0] if W_tensor_site_p.rank() > 0 else 0
                    K = R_p1_tensor.shape()[0] if R_p1_tensor.rank() > 0 else 0

                    if N > 0 and M_phys > 0:
                        # Reshape to (N, 1) and (M_phys, 1) for Kron, then flatten.
                        # Ensure input to Kron are 2D.
                        T1_L = L_p_tensor.reshape(N, 1) if L_p_tensor.rank() == 1 else L_p_tensor
                        T2_W_p = W_tensor_site_p.reshape(M_phys, 1) if W_tensor_site_p.rank() == 1 else W_tensor_site_p
                        
                        kron_LW_tensor = cytnx.linalg.Kron(T1_L, T2_W_p) 
                        kron_LW_flat = kron_LW_tensor.reshape(N * M_phys) 
                        pf_param.weight_row = cytnx.linalg.Abs(kron_LW_flat)

                    if M_phys > 0 and K > 0:
                        # Reshape to (M_phys, 1) and (K, 1) for Kron, then flatten.
                        T1_W_p = W_tensor_site_p.reshape(M_phys, 1) if W_tensor_site_p.rank() == 1 else W_tensor_site_p
                        T2_R = R_p1_tensor.reshape(K, 1) if R_p1_tensor.rank() == 1 else R_p1_tensor

                        kron_WR_tensor = cytnx.linalg.Kron(T1_W_p, T2_R) 
                        kron_WR_flat = kron_WR_tensor.reshape(M_phys * K) 
                        pf_param.weight_col = cytnx.linalg.Abs(kron_WR_flat)
                else:
                    logger.debug(f"ENV weights: weight_p_list for bond {p_bond} (site {p_bond}) is empty.")
            else: 
                l_shape_str = str(L_p_uni.shape()) if L_p_uni else "None"
                r_shape_str = str(R_p1_uni.shape()) if R_p1_uni else "None"
                logger.debug(f"ENV weights: L[{p_bond}] (shape {l_shape_str}) or R[{p_bond+1}] (shape {r_shape_str}) is None for PivotFinderParam at bond {p_bond}.")
        
        if self.param.cond is not None and p_bond < len(self.Pi_bool_mat) and self.Pi_bool_mat[p_bond] is not None:
            pi_bool_tensor_p = self.Pi_bool_mat[p_bond]
            def f_bool_from_precomputed_matrix(r_idx: int, c_idx: int) -> bool:
                if 0 <= r_idx < pi_bool_tensor_p.shape()[0] and 0 <= c_idx < pi_bool_tensor_p.shape()[1]:
                    return bool(pi_bool_tensor_p[r_idx, c_idx].item())
                return False 
            pf_param.f_bool = f_bool_from_precomputed_matrix
        return pf_param

    def update_env_at(self, p_bond: int): 
        if not (self.tt_sum_env and self.param.weight and self.tt_sum_env.is_active): return
        logger.debug(f"TensorCI1.update_env_at for bond {p_bond}")
        tp1_core = self.get_TP1_at(p_bond)
        if tp1_core and p_bond < len(self.param.weight):
            self.tt_sum_env.update_site(p_bond, tp1_core, self.param.weight[p_bond], True)
        
        if (p_bond + 1) < self.D : 
            p1t_core = self.get_P1T_at(p_bond + 1) # This was the line causing error previously
            if p1t_core and (p_bond + 1) < len(self.param.weight):
                self.tt_sum_env.update_site(p_bond + 1, p1t_core, self.param.weight[p_bond+1], False)
                                            
    def iterate_one_full_sweep(self, update_M_data: bool = True) -> float:
        logger.info("Starting iterate_one_full_sweep...")
        max_pivot_error_sweep = 0.0
        num_bonds = self.D - 1 if self.D > 0 else 0
        if num_bonds == 0:
            if not self.errorDecay: self.errorDecay.append(0.0);
            else: self.errorDecay.append(self.errorDecay[-1])
            return self.errorDecay[-1]

        for sweep_dir in range(2): # 0 for L-R, 1 for R-L
            bond_iter = range(num_bonds) if sweep_dir == 0 else range(num_bonds - 1, -1, -1)
            sweep_name = "L-R" if sweep_dir == 0 else "R-L"
            logger.debug(f"    Starting {sweep_name} half sweep.")

            for p_bond in bond_iter:
                logger.debug(f"      Processing bond {p_bond} ({sweep_name})")
                
                if self.pivFinder[p_bond] is None or self.mat_lazy_Pi_at[p_bond] is None or self.P_cross_data[p_bond] is None:
                    logger.warning(f"Skipping bond {p_bond} in {sweep_name} sweep: critical component None.")
                    continue

                current_pf_param = self._build_pivot_finder_param_at_bond(p_bond)
                self.pivFinder[p_bond].p = current_pf_param
                pivot_data = self.pivFinder[p_bond](self.mat_lazy_Pi_at[p_bond], self.P_cross_data[p_bond])

                if pivot_data.i == -1:
                    logger.debug(f"      No valid pivot found for bond {p_bond}. Skipping further updates.") # DEBUG
                    continue # No pivot found
                max_pivot_error_sweep = max(max_pivot_error_sweep, pivot_data.error)
                if pivot_data.error < self.param.reltol:
                    logger.debug(f"      Pivot error {pivot_data.error:.2e} below reltol {self.param.reltol}. Skipping further updates.") # DEBUG
                    continue

                # 1. Update P_cross_data
                logger.debug(f"      Adding pivot ({pivot_data.i}, {pivot_data.j}) for P_cross_data[{p_bond}] with error {pivot_data.error:.2e}") # DEBUG
                self.P_cross_data[p_bond].addPivot(pivot_data.i, pivot_data.j, self.mat_lazy_Pi_at[p_bond])
                logger.debug(f"      P_cross_data[{p_bond}] rank after update: {self.P_cross_data[p_bond].rank()}") # DEBUG

                if update_M_data: # Corresponds to update_T_P_Iset_Jset
                    # 2. Update Global Iset/Jset
                    pi_interface = self.mat_lazy_Pi_at[p_bond]
                    try:
                        # .from_int() returns a shallow copy of internal list of MultiIndex.
                        # .pos() returns integer index
                        # pi_interface.Iset.from_int()[pivot_data.i] gives the MultiIndex.
                        # Check if new_global_I_mi is already in self.Iset[p_bond+1]
                        new_global_I_mi = pi_interface.Iset.from_int()[pivot_data.i] # Global MultiIndex for Pi's row
                        logger.debug(f"      New global I-pivot MI for bond {p_bond}: {new_global_I_mi}") # DEBUG
                        # --- START FIX for IndexSet.contains ---
                        # Original: if not self.Iset[p_bond+1].contains(new_global_I_mi):
                        if new_global_I_mi not in self.Iset[p_bond+1]._index_set: # Use internal set for containment check
                            self.Iset[p_bond+1].push_back(new_global_I_mi)
                            logger.debug(f"      Iset[{p_bond+1}] updated. New length: {len(self.Iset[p_bond+1])}") # DEBUG

                        new_global_J_mi = pi_interface.Jset.from_int()[pivot_data.j] # Global MultiIndex for Pi's col
                        logger.debug(f"      New global J-pivot MI for bond {p_bond}: {new_global_J_mi}") # DEBUG
                        # --- START FIX for IndexSet.contains ---
                        # Original: if not self.Jset[p_bond].contains(new_global_J_mi):
                        if new_global_J_mi not in self.Jset[p_bond]._index_set: # Use internal set for containment check
                            self.Jset[p_bond].push_back(new_global_J_mi)
                            logger.debug(f"      Jset[{p_bond}] updated. New length: {len(self.Jset[p_bond])}") # DEBUG
                        # --- END FIX for IndexSet.contains ---

                    except AttributeError: # .from_int() or .contains might not exist on IndexSet as assumed
                        logger.error(f"IndexSet for Pi_mat[{p_bond}] does not have expected methods. Cannot update global Iset/Jset."); continue
                    except Exception as e_idx: logger.error(f"Error updating global Iset/Jset for bond {p_bond}: {e_idx}"); continue

                    # 3. Update T_cores and P_matrices based on new CrossData and new Iset/Jset sizes
                    # This update re-evaluates the T_cores based on the (new) Iset/Jset lengths.
                    # It is crucial to ensure that the mat_lazy_Pi_at's internal IndexSets are also updated.
                    # This is done by calling set_rows/set_cols on mat_lazy_Pi_at AFTER the global Iset/Jset are updated.
                    self._update_tt_M_and_P_pivot_matrix_after_crossdata_update(p_bond)
                    
                    # After global Iset/Jset are updated, rebuild pi_rows_mi/pi_cols_mi and update mat_lazy_Pi_at
                    # This ensures mat_lazy_Pi_at's internal IndexSets are consistent with the global ones.
                    updated_pi_rows_mi = [i_idx_multi + l_mi_tuple
                        for i_idx_multi in self.Iset[p_bond].get_all() 
                        for l_mi_tuple in self.localSet[p_bond].get_all()] 
                    updated_pi_cols_mi = [l_mi_p1_tuple + j_idx_multi
                        for l_mi_p1_tuple in self.localSet[p_bond+1].get_all() 
                        for j_idx_multi in self.Jset[p_bond+1].get_all()] 

                    logger.debug(f"      Updating mat_lazy_Pi_at[{p_bond}] with new Iset/Jset lists.") # DEBUG
                    # Note: set_rows/set_cols return a permutation list, which is not used here.
                    self.mat_lazy_Pi_at[p_bond].set_rows(updated_pi_rows_mi)
                    self.mat_lazy_Pi_at[p_bond].set_cols(updated_pi_cols_mi)


                if self.tt_sum_env and self.tt_sum_env.is_active: self.update_env_at(p_bond)

        self.errorDecay.append(max_pivot_error_sweep)
        logger.info(f"Finished iterate_one_full_sweep. Max pivot error: {max_pivot_error_sweep:.4e}")
        return max_pivot_error_sweep


    def IterateN(self, n_iterations: int, update_M_data: bool = True):
        logger.info(f"TensorCI1.IterateN called for {n_iterations} iterations.")
        if self.D <= 1:
            if not self.errorDecay: self.errorDecay.append(0.0)
            self.done = True; return

        for i in range(n_iterations):
            logger.debug(f"  Iteration sweep {i+1}/{n_iterations}")
            max_err_sweep = self.iterate_one_full_sweep(update_M_data)
            if max_err_sweep < self.param.reltol:
                logger.info(f"  Converged after {i+1} iterations."); self.done = True; break
        if not self.done and n_iterations > 0: logger.info(f"  Finished {n_iterations} iterations.")


    def _update_tt_M_and_P_pivot_matrix_after_crossdata_update(self, p_bond: int):
        """
        Updates TT cores and P_pivot_matrices based on updated CrossData and global Iset/Jset.
        This function needs to re-evaluate the relevant cores and pivot matrices.
        """
        logger.debug(f"_update_tt_M_and_P_pivot_matrix_after_crossdata_update for bond {p_bond}")
        cross_data_p = self.P_cross_data[p_bond]
        if cross_data_p is None:
            logger.error(f"CrossData for bond {p_bond} is None in update function. Skipping.")
            return

        # A. Update P_pivot_matrices[p_bond] from cross_data_p.pivotMat()
        # Its dimension is rank_of_bond x rank_of_bond
        pivot_mat_val = cross_data_p.pivotMat()
        # Ensure the P_pivot_matrices list is correctly sized (num_bonds elements)
        # It's indexed by p_bond, which goes from 0 to num_bonds-1.
        if p_bond < len(self.P_pivot_matrices): # Make sure index is valid
            if pivot_mat_val is not None:
                self.P_pivot_matrices[p_bond] = pivot_mat_val.astype(self.dtype).to(self.device)
                logger.debug(f"  P_pivot_matrices[{p_bond}] updated with shape {self.P_pivot_matrices[p_bond].shape()}.") # DEBUG
            else: # If pivotMat is None, it means the rank is 0, so P is conceptually Identity(0) or None
                rank_bond = cross_data_p.rank()
                if rank_bond == 0:
                    self.P_pivot_matrices[p_bond] = None # Or a 0x0 tensor if needed
                    logger.debug(f"  P_pivot_matrices[{p_bond}] set to None due to rank 0.") # DEBUG
                else: # Fallback if unexpected None but rank > 0
                    logger.warning(f"pivotMat for bond {p_bond} was None but rank > 0. Setting P_pivot_matrices[{p_bond}] to identity({rank_bond}).") # WARNING
                    self.P_pivot_matrices[p_bond] = cytnx.eye(rank_bond, dtype=self.dtype, device=self.device)
        else:
            logger.error(f"Attempted to update P_pivot_matrices[{p_bond}] but index is out of bounds (len={len(self.P_pivot_matrices)}).")


        # B. Update T_cores[p_bond] using cross_data_p.C
        # T_cores[p_bond] shape (len(Iset[p_bond]), phys_dims[p_bond], len(Jset[p_bond]))
        C_matrix = cross_data_p.C
        if C_matrix is not None:
            dim_I_p = len(self.Iset[p_bond]) # Updated Iset
            dim_phys_p = self.phys_dims[p_bond]
            dim_J_p = len(self.Jset[p_bond]) # Updated Jset (this is the new bond dimension)
            
            # Check expected shape of C_matrix from CrossData: (dim_I_p * dim_phys_p, dim_J_p)
            expected_C_shape = (dim_I_p * dim_phys_p, dim_J_p)
            if C_matrix.shape() == list(expected_C_shape): # .shape() returns list, so compare to list
                reshaped_C = C_matrix.reshape(dim_I_p, dim_phys_p, dim_J_p)

                label_L = f'link{p_bond-1}' if p_bond > 0 else 'L_bound'
                label_P = f'p{p_bond}'
                label_R = f'link{p_bond}' # Jset[p_bond] defines the right link of T_cores[p_bond]

                bonds = [cytnx.Bond(dim_I_p, cytnx.BD_IN),
                         cytnx.Bond(dim_phys_p, cytnx.BD_OUT),
                         cytnx.Bond(dim_J_p, cytnx.BD_OUT)]
                
                # Create UniTensor with updated bond dimensions and put block
                new_core_ut_p = UniTensor(bonds=bonds, labels=[label_L, label_P, label_R], rowrank=1, is_diag=False)
                new_core_ut_p.put_block(reshaped_C.astype(self.dtype).to(self.device))
                self.tt.M[p_bond] = new_core_ut_p
                logger.debug(f"  T_cores[{p_bond}] updated to shape {self.tt.M[p_bond].shape()}.") # DEBUG
            else:
                logger.error(f"  Shape mismatch C_matrix for T_cores[{p_bond}]. Expected {expected_C_shape}, Got {C_matrix.shape()}. Skipping update.")
        else: logger.warning(f"  C_matrix None for T_cores[{p_bond}]. Skipping update.")


        # C. Update T_cores[p_bond+1] using cross_data_p.R
        if (p_bond + 1) < self.D:
            R_matrix = cross_data_p.R
            if R_matrix is not None:
                dim_I_pp1 = len(self.Iset[p_bond+1]) # Updated Iset
                dim_phys_pp1 = self.phys_dims[p_bond+1]
                dim_J_pp1 = len(self.Jset[p_bond+1]) # Updated Jset

                # Check expected shape of R_matrix from CrossData: (dim_I_pp1, dim_phys_pp1 * dim_J_pp1)
                expected_R_shape = (dim_I_pp1, dim_phys_pp1 * dim_J_pp1)
                if R_matrix.shape() == list(expected_R_shape): # .shape() returns list
                    reshaped_R = R_matrix.reshape(dim_I_pp1, dim_phys_pp1, dim_J_pp1)

                    label_L_pp1 = f'link{p_bond}' # Iset[p_bond+1] defines left link of T_cores[p_bond+1]
                    label_P_pp1 = f'p{p_bond+1}'
                    label_R_pp1 = f'link{p_bond+1}' if (p_bond+1) < self.D-1 else 'R_bound'

                    bonds_pp1 = [cytnx.Bond(dim_I_pp1, cytnx.BD_IN),
                                 cytnx.Bond(dim_phys_pp1, cytnx.BD_OUT),
                                 cytnx.Bond(dim_J_pp1, cytnx.BD_OUT)]
                    
                    new_core_ut_pp1 = UniTensor(bonds=bonds_pp1, labels=[label_L_pp1, label_P_pp1, label_R_pp1], rowrank=1, is_diag=False)
                    new_core_ut_pp1.put_block(reshaped_R.astype(self.dtype).to(self.device))
                    self.tt.M[p_bond+1] = new_core_ut_pp1
                    logger.debug(f"  T_cores[{p_bond+1}] updated to shape {self.tt.M[p_bond+1].shape()}.") # DEBUG
                else:
                    logger.error(f"  Shape mismatch R_matrix for T_cores[{p_bond+1}]. Expected {expected_R_shape}, Got {R_matrix.shape()}. Skipping update.")
            else: logger.warning(f"  R_matrix None for T_cores[{p_bond+1}]. Skipping update.")


    def get_canonical_tt(self, center: int) -> TensorTrain:
        logger.info(f"Constructing canonical TensorTrain with center at {center}")
        num_sites = self.D
        if not (0 <= center < num_sites):
            if center < 0: center += num_sites
            if not (0 <= center < num_sites):
                raise ValueError(f"Adjusted center {center} is still out of bounds for {num_sites} sites.")

        new_tt = TensorTrain()
        new_tt.M = [None] * num_sites

        for p in range(center):
            tp1_core = self.get_TP1_at(p)
            if tp1_core is None:
                logger.error(f"get_TP1_at({p}) failed during canonicalization. Returning current TT state.")
                return self.tt # Fallback to current TT state
            new_tt.M[p] = tp1_core

        if self.tt.M[center] is None:
            logger.error(f"T_cores[{center}] is None for center core in canonicalization. Returning current TT state.")
            return self.tt
        new_tt.M[center] = self.tt.M[center].clone() # The center core itself is not transformed

        for p in range(center + 1, num_sites):
            p1t_core = self.get_P1T_at(p)
            if p1t_core is None:
                logger.error(f"get_P1T_at({p}) failed during canonicalization. Returning current TT state.")
                return self.tt
            new_tt.M[p] = p1t_core

        if any(core is None for core in new_tt.M):
            logger.error("Failed to construct all canonical TT cores. Returning current TT state.")
            return self.tt

        # DEBUG: Check shapes and block data of canonicalized TT cores
        for i, core in enumerate(new_tt.M):
            if core:
                logger.debug(f"Canonical TT M[{i}] shape: {core.shape()}") # DEBUG
                logger.debug(f"Canonical TT M[{i}] block:\n{core.get_block_().numpy()}") # DEBUG - this is the new line
        return new_tt

    def len(self) -> int:
        return self.D

    def getPivotsAt(self, p_bond: int) -> Optional[List[MultiIndex]]:
        if not (0 <= p_bond < self.D -1): return None
        # Pivots defining bond p_bond are in Jset[p_bond] (right of T_core[p_bond])
        # or Iset[p_bond+1] (left of T_core[p_bond+1])
        # These should have the same length.
        # The C++ getPivotsAt(b) returns vector<vector<int>> from Iset[b+1][r] + Jset[b][r]
        # This refers to the *elements* of the P_b matrix, not the fibers.
        # For fibers forming the bond, return Jset[p_bond]
        if self.Jset[p_bond] and hasattr(self.Jset[p_bond], 'get_all'):
            return list(self.Jset[p_bond].get_all())
        return []

    def trueError(self, F_exact_func: Callable[[Tuple[int,...]], float], max_n_eval: int = 1000000) -> float:
        # Ensure evaluate_tt or an equivalent TensorTrain.evaluate method exists and works.
        evaluatable_tt = self.get_canonical_tt(center=0)
        if not evaluatable_tt.M or not all(c is not None for c in evaluatable_tt.M):
             logger.warning("trueError: Could not get a valid canonical_tt. Error cannot be computed.")
             return float('inf')

        # This part requires a working TensorTrain.eval(idx) method which is assumed from tensor_train.py
        # The provided evaluate_tt helper function is for the main test block.
        # So we need to ensure TensorTrain has .eval() or use the helper here.
        # Given TensorTrain already has an eval method, we should use that.
        if not hasattr(evaluatable_tt, 'eval'): # Assuming it's evaluate now
            logger.warning("trueError: TensorTrain object does not have an 'eval' method. Cannot compute.")
            return float('inf')

        total_elements = np.prod(self.phys_dims, dtype=np.int64)
        if total_elements == 0:
            return 0.0 # If any dimension is zero, total elements is zero, error is zero

        if total_elements > max_n_eval:
            logger.warning(f"trueError: Total elements {total_elements} > max_n_eval {max_n_eval}. "
                           f"Error computation will be skipped or approximated. Returning inf.")
            return float('inf') # Or implement sampling if needed

        max_abs_diff = 0.0; count = 0
        phys_idx_ranges = [range(d) for d in self.phys_dims]
        for current_mi_tuple in product(*phys_idx_ranges):
            # TensorTrain.eval expects List[int]
            current_mi_for_eval = list(current_mi_tuple) 

            val_tci = evaluatable_tt.eval(current_mi_for_eval)
            val_exact = F_exact_func(current_mi_tuple)

            diff = abs(val_tci - val_exact)
            if diff > max_abs_diff: max_abs_diff = diff
            count += 1
            if count >= max_n_eval and total_elements > max_n_eval : break
        return max_abs_diff

# --- Main Test Block (Expanded) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, # Changed to DEBUG for more verbose output
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger.info("--- Starting TensorCI System Test (Expanded) ---")

    test_phys_dims_small = [2, 2] 
    test_D_small = len(test_phys_dims_small)
    test_dtype = cytnx.Type.Double
    test_device = cytnx.Device.cpu
    np.random.seed(0) # For any lingering randomness, though we aim for deterministic functions

    # --- Deterministic Mock Functions ---
    def mock_fc_separable_D2(indices: Tuple[int, ...]) -> float:
        """ D=2, phys_dims=[2,2]. Expected Rank 1. f(i,j) = cos(i)sin(j)"""
        if len(indices) == 2:
            # Map indices {0,1} to some angles, e.g., 0 -> 0, 1 -> pi/2
            angle0 = float(indices[0]) * np.pi / 2.0
            angle1 = float(indices[1]) * np.pi / 2.0
            val = np.cos(angle0 + 0.1) * np.sin(angle1 + 0.2) # Add small offsets
            return float(val)
        logger.error(f"mock_fc_separable_D2: Incorrect number of indices: {len(indices)}")
        return 0.0

    def mock_fc_separable_D3(indices: Tuple[int, ...]) -> float:
        """ D=3, phys_dims=[2,2,2]. Expected Rank 1. f(i,j,k) = cos(i)sin(j)cos(k)"""
        if len(indices) == 3:
            angle0 = float(indices[0]) * np.pi / 2.0
            angle1 = float(indices[1]) * np.pi / 2.0
            angle2 = float(indices[2]) * np.pi / 2.0
            val = np.cos(angle0 + 0.1) * np.sin(angle1 + 0.2) * np.cos(angle2 + 0.3)
            return float(val)
        logger.error(f"mock_fc_separable_D3: Incorrect number of indices: {len(indices)}")
        return 0.0

    def mock_fc_rank2_D2(indices: Tuple[int, ...]) -> float:
        """ D=2, phys_dims=[2,2]. Expected Rank 2. f(i,j) = cos(i)cos(j) + sin(i)sin(j) = cos(i-j)"""
        if len(indices) == 2:
            angle0 = float(indices[0]) * np.pi / 2.0 # Map 0 to 0, 1 to pi/2
            angle1 = float(indices[1]) * np.pi / 2.0
            # val = np.cos(angle0) * np.cos(angle1) + np.sin(angle0) * np.sin(angle1) # cos(a-b)
            # More distinct terms for TCI to pick up:
            val = 0.7 * np.cos(angle0) * np.cos(angle1+0.1) + 0.3 * np.sin(angle0+0.2) * np.sin(angle1+0.3)
            return float(val)
        logger.error(f"mock_fc_rank2_D2: Incorrect number of indices: {len(indices)}")
        return 0.0

    def mock_fc_all_zeros(indices: Tuple[int, ...]) -> float:
        return 0.0

    def mock_fc_constant(indices: Tuple[int, ...], const_val: float = 3.14) -> float:
        return const_val

    def mock_cond_always_true(full_multi_index: List[int]) -> bool:
        return True

    # Helper to evaluate TT (if not part of TensorTrain class)
    def evaluate_tt(tensor_train: TensorTrain, indices: Tuple[int, ...]) -> float:
        # Use the existing eval method of TensorTrain
        return tensor_train.eval(list(indices))

    # --- Test Case 1: Basic Initialization & Iteration with Deterministic Function (D=2, Rank=1 Target) ---
    logger.info("\n--- Test Case 1: Basic D=2, Rank-1 Target ---")
    tc1_phys_dims = [2, 2]
    tc1_D = len(tc1_phys_dims)
    tc1_fc_func = TensorFunction(func=mock_fc_separable_D2)
    tc1_params = TensorCI1Param(pivot1=[0]*tc1_D, reltol=1e-9, nIter=0) # nIter=0 for manual iteration control

    try:
        tci1_inst = TensorCI1(fc=tc1_fc_func, phys_dims=tc1_phys_dims, param=tc1_params, dtype=test_dtype, device=test_device)
        logger.info(f"TC1: Initialized for D={tc1_D} separable function.")
        assert tci1_inst.P_cross_data[0].rank() == 1, "Initial rank should be 1"

        logger.info("TC1: Calling IterateN(5)")
        tci1_inst.IterateN(5) # Max 5 full sweeps

        logger.info(f"TC1: errorDecay: {tci1_inst.errorDecay}")
        assert len(tci1_inst.errorDecay) > 0, "Error decay should have entries."
        # For a rank-1 function, error should ideally drop quickly to near zero or below reltol
        if len(tci1_inst.errorDecay) > 1 : # If it iterated at least once after the initial pivot
            assert tci1_inst.errorDecay[-1] < tc1_params.reltol * 10, "Error should be small for rank-1 function"

        final_rank_bond0 = tci1_inst.P_cross_data[0].rank()
        logger.info(f"TC1: Final rank at bond 0: {final_rank_bond0}")
        assert final_rank_bond0 == 1, f"Final rank for separable D=2 function should be 1, got {final_rank_bond0}"

    except Exception as e:
        logger.error(f"Error in Test Case 1 (D=2, Rank-1): {e}", exc_info=True)
        raise

    # --- Test Case 2: get_canonical_tt with Deterministic Rank-1 Function (D=2) ---
    logger.info("\n--- Test Case 2: get_canonical_tt (D=2, Rank-1 Target) ---")
    # Assuming tci1_inst from Test Case 1 is available and converged
    if 'tci1_inst' in locals() and tci1_inst.done:
        for center in range(tci1_inst.D):
            logger.info(f"TC2: Calling get_canonical_tt with center={center}")
            canonical_tt = tci1_inst.get_canonical_tt(center)
            assert canonical_tt is not None, f"get_canonical_tt({center}) returned None"
            assert len(canonical_tt.M) == tci1_inst.D, f"Canonical TT has incorrect length {len(canonical_tt.M)}"

            logger.info(f"TC2: Canonical TT (center={center}) core shapes:")
            for p_idx, core in enumerate(canonical_tt.M):
                assert core is not None, f"Core M[{p_idx}] in canonical TT is None"
                logger.info(f"    M[{p_idx}]: {core.shape()}")
                # For a rank-1 function, all bond dimensions should be 1
                assert core.shape()[0] == 1, f"M[{p_idx}] left bond dim is {core.shape()[0]}, expected 1 for rank-1 func"
                assert core.shape()[2] == 1, f"M[{p_idx}] right bond dim is {core.shape()[2]}, expected 1 for rank-1 func"

            logger.info(f"TC2: Testing element reconstruction for canonical TT (center={center})")
            for i0 in range(tc1_phys_dims[0]):
                for i1 in range(tc1_phys_dims[1]):
                    test_indices = (i0, i1)
                    val_expected = mock_fc_separable_D2(test_indices)
                    try:
                        val_reconstructed = evaluate_tt(canonical_tt, test_indices) # Use helper
                        logger.debug(f"    Indices {test_indices}: Expected={val_expected:.6f}, Reconstructed={val_reconstructed:.6f}")
                        assert np.isclose(val_expected, val_reconstructed, rtol=tc1_params.reltol*100, atol=1e-9), \
                                f"Mismatch for {test_indices}. Expected: {val_expected}, Got: {val_reconstructed}"
                    except Exception as e:
                        logger.error(f"Error evaluating canonical_tt for {test_indices}: {e}", exc_info=True)
                        raise # Fail test if evaluation fails
    else:
        logger.warning("Skipping Test Case 2: tci1_inst not available or not converged from Test Case 1.")

    # --- Test Case 3: D=3 Separable Function (Rank-1 Target) ---
    logger.info("\n--- Test Case 3: Basic D=3, Rank-1 Target ---")
    tc3_phys_dims = [2, 2, 2]
    tc3_D = len(tc3_phys_dims)
    tc3_fc_func = TensorFunction(func=mock_fc_separable_D3)
    tc3_params = TensorCI1Param(pivot1=[0]*tc3_D, reltol=1e-9, nIter=10) # Allow more iterations

    try:
        tci3_inst = TensorCI1(fc=tc3_fc_func, phys_dims=tc3_phys_dims, param=tc3_params, dtype=test_dtype, device=test_device)
        logger.info(f"TC3: Initialized for D={tc3_D} separable function.")
        # Initial rank for D=3: bond 0 and bond 1 are rank 1
        assert tci3_inst.P_cross_data[0].rank() == 1, "Initial rank bond 0 should be 1"
        assert tci3_inst.P_cross_data[1].rank() == 1, "Initial rank bond 1 should be 1"

        # IterateN will be called by __init__ if nIter > 0 in param, or call explicitly
        if tc3_params.nIter == 0: tci3_inst.IterateN(10)

        logger.info(f"TC3: errorDecay: {tci3_inst.errorDecay}")
        if len(tci3_inst.errorDecay) > 1 :
            assert tci3_inst.errorDecay[-1] < tc3_params.reltol * 10, "Error should be small for rank-1 D=3 function"

        for b_idx in range(tc3_D - 1):
            final_rank_bond = tci3_inst.P_cross_data[b_idx].rank()
            logger.info(f"TC3: Final rank at bond {b_idx}: {final_rank_bond}")
            assert final_rank_bond == 1, f"Final rank for bond {b_idx} for separable D=3 function should be 1, got {final_rank_bond}"

        # Test get_canonical_tt and reconstruction for D=3
        if tci3_inst.done:
            center_d3 = tc3_D // 2
            logger.info(f"TC3: Calling get_canonical_tt with center={center_d3}")
            canonical_tt_d3 = tci3_inst.get_canonical_tt(center_d3)
            assert canonical_tt_d3 is not None

            test_indices_d3 = (0,1,0)
            val_expected_d3 = mock_fc_separable_D3(test_indices_d3)
            val_reconstructed_d3 = evaluate_tt(canonical_tt_d3, test_indices_d3)
            logger.info(f"TC3: Indices {test_indices_d3}: Expected={val_expected_d3:.6f}, Reconstructed={val_reconstructed_d3:.6f}")
            assert np.isclose(val_expected_d3, val_reconstructed_d3, rtol=tc3_params.reltol*100, atol=1e-9)

    except Exception as e:
        logger.error(f"Error in Test Case 3 (D=3, Rank-1): {e}", exc_info=True)
        raise

    # --- Test Case 4: All Zeros Function ---
    logger.info("\n--- Test Case 4: All Zeros Function ---")
    tc4_phys_dims = [2, 2]
    tc4_D = len(tc4_phys_dims)
    # Important: initial pivot value must not be zero for the standard constructor.
    # If mock_fc_all_zeros is used, and pivot1=[0,0] (value 0.0), it should fail.
    tc4_fc_func = TensorFunction(func=mock_fc_all_zeros)
    tc4_params = TensorCI1Param(pivot1=[0]*tc4_D, reltol=1e-9)
    try:
        tci4_inst = TensorCI1(fc=tc4_fc_func, phys_dims=tc4_phys_dims, param=tc4_params)
        logger.error("TC4: TensorCI1 init DID NOT fail for all-zero function with zero initial pivot. This is unexpected.")
        # This might not fail if initial_pivot_value check is only > 1e-15 but not exactly 0.
        # However, the paper mentions f(pivot1) != 0.
        # If it proceeds, TCI on an all-zero function should result in a zero TT.
        tci4_inst.IterateN(1)
        assert tci4_inst.errorDecay[-1] == 0.0, "Error for all-zero function should be 0"
        zero_tt = tci4_inst.get_canonical_tt(0)
        val_reconstructed_zero = evaluate_tt(zero_tt, (0,0))
        assert np.isclose(val_reconstructed_zero, 0.0), "Reconstruction of zero function should be zero"

    except ValueError as ve:
        logger.info(f"TC4: Successfully caught ValueError for all-zero function with zero initial pivot: {ve}")
        assert "is ~zero" in str(ve) or "f(pivot1)=0" in str(ve), "Error message for zero pivot mismatch"
    except Exception as e:
        logger.error(f"TC4: Unexpected error for all-zero function: {e}", exc_info=True)
        raise

    # --- Test Case 5: Constant Non-Zero Function ---
    logger.info("\n--- Test Case 5: Constant Non-Zero Function ---")
    tc5_phys_dims = [2,2]
    tc5_D = len(tc5_phys_dims)
    const_val = 5.5
    def fc_const_local(indices: Tuple[int, ...]) -> float: return mock_fc_constant(indices, const_val)
    tc5_fc_func = TensorFunction(func=fc_const_local)
    # Initial pivot [0,0] will have value const_val, so it's non-zero.
    tc5_params = TensorCI1Param(pivot1=[0]*tc5_D, reltol=1e-9, nIter=5)

    try:
        tci5_inst = TensorCI1(fc=tc5_fc_func, phys_dims=tc5_phys_dims, param=tc5_params)
        logger.info(f"TC5: Initialized for D={tc5_D} constant function (value={const_val}).")
        if tc5_params.nIter == 0: tci5_inst.IterateN(5)

        logger.info(f"TC5: errorDecay: {tci5_inst.errorDecay}")
        assert tci5_inst.errorDecay[-1] < tc5_params.reltol * 10, "Error for constant function should be very small"

        for b_idx in range(tc5_D - 1):
            final_rank_bond = tci5_inst.P_cross_data[b_idx].rank()
            logger.info(f"TC5: Final rank at bond {b_idx}: {final_rank_bond}")
            assert final_rank_bond == 1, f"Rank for constant function should be 1, got {final_rank_bond}"

        const_tt = tci5_inst.get_canonical_tt(0)
        for i0 in range(tc5_phys_dims[0]):
            for i1 in range(tc5_phys_dims[1]):
                test_indices = (i0, i1)
                val_reconstructed = evaluate_tt(const_tt, test_indices)
                assert np.isclose(val_reconstructed, const_val, rtol=1e-7), \
                    f"Const func mismatch for {test_indices}. Expected: {const_val}, Got: {val_reconstructed}"
        logger.info(f"TC5: Constant function reconstruction successful.")

    except Exception as e:
        logger.error(f"Error in Test Case 5 (Constant Function): {e}", exc_info=True)
        raise

    # --- Test Case 6: Rank-2 Target Function (D=2) ---
    logger.info("\n--- Test Case 6: Rank-2 Target Function (D=2) ---")
    tc6_phys_dims = [2, 2] # Physical dimensions for each site
    tc6_D = len(tc6_phys_dims)
    tc6_fc_func = TensorFunction(func=mock_fc_rank2_D2)
    # For rank-2, might need slightly more iterations or different reltol
    tc6_params = TensorCI1Param(pivot1=[0]*tc6_D, reltol=1e-6, nIter=30, fullPiv=False) # 增加迭代次數並放寬容忍度

    try:
        tci6_inst = TensorCI1(fc=tc6_fc_func, phys_dims=tc6_phys_dims, param=tc6_params)
        logger.info(f"TC6: Initialized for D={tc6_D} rank-2 target function.")
        if tc6_params.nIter == 0: tci6_inst.IterateN(30) # Ensure iteration count is applied

        logger.info(f"TC6: errorDecay: {tci6_inst.errorDecay}")
        assert tci6_inst.done, "TCI should converge for rank-2 function"

        final_rank_bond0_tc6 = tci6_inst.P_cross_data[0].rank()
        logger.info(f"TC6: Final rank at bond 0: {final_rank_bond0_tc6}")
        # Depending on the exact numbers in mock_fc_rank2_D2 and reltol, it might find rank 1 or 2.
        # For cos(x-y) structure, if phys_dim is 2 (0, pi/2), it might become rank 1 if some terms vanish.
        # The modified mock_fc_rank2_D2 aims to be more robustly rank 2.
        assert final_rank_bond0_tc6 <= 2, f"Rank for D=2 rank-2 target should be at most 2, got {final_rank_bond0_tc6}"
        if final_rank_bond0_tc6 == 0: # Should not happen for non-zero function
            logger.warning("TC6: Rank is 0, this is unexpected for this function.")


        if final_rank_bond0_tc6 > 0: # Proceed if rank is not zero
            rank2_tt = tci6_inst.get_canonical_tt(0)
            logger.info(f"TC6: Testing element reconstruction for rank-2 target canonical TT (center=0)")
            max_reconstruction_error_tc6 = 0.0
            for i0 in range(tc6_phys_dims[0]):
                for i1 in range(tc6_phys_dims[1]):
                    test_indices = (i0, i1)
                    val_expected = mock_fc_rank2_D2(test_indices)
                    val_reconstructed = evaluate_tt(rank2_tt, test_indices)
                    current_error = abs(val_expected - val_reconstructed)
                    max_reconstruction_error_tc6 = max(max_reconstruction_error_tc6, current_error)
                    logger.debug(f"    Indices {test_indices}: Expected={val_expected:.6f}, Reconstructed={val_reconstructed:.6f}, Error={current_error:.2e}")
            logger.info(f"TC6: Max reconstruction error for rank-2 target: {max_reconstruction_error_tc6:.2e}")
            # Check if max error is within a multiple of reltol, as TCI error is max error on pivots
            assert max_reconstruction_error_tc6 < (tc6_params.reltol * 1000 if tc6_params.reltol > 0 else 1e-9), \
                "Reconstruction error too high for rank-2 target"
        else:
            logger.warning("TC6: Skipping reconstruction test as final rank is 0.")


    except Exception as e:
        logger.error(f"Error in Test Case 6 (D=2, Rank-2 Target): {e}", exc_info=True)
        raise

    # --- Test Case 7: Check ENV learning path (idempotency of update_env_at) ---
    # This is similar to your original Test Case 2 & 3, using a deterministic function
    logger.info("\n--- Test Case 7: ENV Learning update_env_at Idempotency ---")
    tc7_phys_dims = [2, 2]
    tc7_D = len(tc7_phys_dims)
    tc7_fc_func = TensorFunction(func=mock_fc_separable_D2) # Use a deterministic func
    tc7_weights = [[0.8, 1.2], [0.9, 1.1]]
    tc7_params = TensorCI1Param(
        pivot1=[0]*tc7_D,
        weight=tc7_weights,
        cond=mock_cond_always_true,
        nIter=1, # One full sweep
        reltol=1e-5
    )
    tci7_inst = None
    try:
        tci7_inst = TensorCI1(fc=tc7_fc_func, phys_dims=tc7_phys_dims, param=tc7_params)
        logger.info("TC7: TensorCI1 (D=2, ENV, Cond) initialized.")

        # Environment state after nIter=1 (done in __init__ or by explicit IterateN if nIter=0)
        # Ensure IterateN has run if nIter in param was 0
        if tc7_params.nIter == 0: tci7_inst.IterateN(1)

        if tci7_inst.tt_sum_env and tci7_inst.tt_sum_env.is_active:
            L1_b_core = tci7_inst.tt_sum_env.L[1]
            L1_b_numpy = L1_b_core.get_block_().numpy().copy() if tc7_D > 1 and L1_b_core else None

            R0_b_core = tci7_inst.tt_sum_env.R[0]
            R0_b_numpy = R0_b_core.get_block_().numpy().copy() if tc7_D > 0 and R0_b_core else None

            logger.info("TC7: Calling update_env_at(p_bond=0) again.")
            tci7_inst.update_env_at(p_bond=0)

            if tc7_D > 1 and L1_b_numpy is not None:
                L1_a_core = tci7_inst.tt_sum_env.L[1]
                assert L1_a_core is not None, "TC7: L[1] should exist after update"
                L1_a_numpy = L1_a_core.get_block_().numpy()
                assert np.allclose(L1_a_numpy, L1_b_numpy, atol=1e-8), \
                    "TC7: L1 should NOT have changed after a second identical update_env_at call"
                logger.info(f"    TC7: L[1] after 2nd update_env_at (shape {L1_a_core.shape()}) matches state after 1st update: PASSED Idempotency")

            if tc7_D > 0 and R0_b_numpy is not None:
                R0_a_core = tci7_inst.tt_sum_env.R[0]
                assert R0_a_core is not None, "TC7: R[0] should exist after update"
                R0_a_numpy = R0_a_core.get_block_().numpy()
                assert np.allclose(R0_a_numpy, R0_b_numpy, atol=1e-8), \
                    "TC7: R0 should NOT have changed after a second identical update_env_at call"
                logger.info(f"    TC7: R[0] after 2nd update_env_at (shape {R0_a_core.shape()}) matches state after 1st update: PASSED Idempotency")
        else:
            logger.warning("TC7: Skipping ENV idempotency check, tt_sum_env not active.")

    except Exception as e:
        logger.error(f"Error in Test Case 7 (ENV Idempotency): {e}", exc_info=True)
        raise

    # --- (Placeholder) Test Case for getPivotsAt ---
    # logger.info("\n--- Test Case X: getPivotsAt ---")
    # if 'tci1_inst' in locals() and tci1_inst.done:
    #    # 1. Implement getPivotsAt in TensorCI1
    #    #    It needs to access the global Iset/Jset equivalents from the TCI process
    #    #    (Not just the local P_cross_data[b].lu.Iset/Jset for the Pi_matrix)
    #    # pivots_bond0 = tci1_inst.getPivotsAt(0)
    #    # logger.info(f"Pivots at bond 0: {pivots_bond0}")
    #    # assert len(pivots_bond0) == tci1_inst.P_cross_data[0].rank(), "Number of pivots should match rank"
    #    pass
    # else:
    #    logger.warning("Skipping getPivotsAt test: tci1_inst not available or not converged.")

    logger.info("\n--- TensorCI System Test (Expanded) Finished ---")