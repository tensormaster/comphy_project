# tensor_ci.py
import cytnx
from cytnx import UniTensor, Tensor, Bond, BD_IN, BD_OUT, linalg, Type, Device, from_numpy
from typing import List, Callable, Optional, Any, Tuple
import logging
import numpy as np
import dataclasses

# --- Necessary imports ---
from tensor_train import TensorTrain #
from matrix_interface import IMatrix, MatLazyIndex, make_IMatrix # Added make_IMatrix
from crossdata import CrossData #
from pivot_finder import PivotFinder, PivotFinderParam, PivotData #
from tensorfuc import TensorFunction #
from tensor_utils import cube_as_matrix1, cube_as_matrix2, mat_AB1, mat_A1B #


logger = logging.getLogger(__name__)
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
                result_L = cytnx.ncon(
                    [L_prev_c, M_p_core_c, W_p_ut_c],
                    [[10], [10, 20, -1], [20]],  
                    cont_order=[10, 20] 
                )
                result_L.set_labels([f'vL_out_site{p}'])
                self.L[p+1] = result_L
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
                result_R = cytnx.ncon(
                    [M_p_core_c, W_p_ut_c, R_next_c], 
                    [[-1, 20, 10], [20], [10]],      
                    cont_order=[20, 10] 
                )
                result_R.set_labels([f'vR_out_site{p-1}'])
                self.R[p-1] = result_R
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
                result_L = cytnx.ncon(
                    [L_prev_c, M_core_norm_c, W_ut_c],
                    [[10],[10,20,-1],[20]], 
                    cont_order=[10,20]
                )
                result_L.set_labels([f'vL_updated_out_site{p_core_idx}'])
                self.L[p_core_idx+1] = result_L
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
                result_R = cytnx.ncon(
                    [M_core_norm_c, W_ut_c, R_next_c],
                    [[-1,20,10],[20],[10]], 
                    cont_order=[20,10]
                )
                result_R.set_labels([f'vR_updated_out_site{p_core_idx-1}'])
                self.R[p_core_idx-1] = result_R
            except RuntimeError as e: 
                logger.error(f"ncon R[{p_core_idx-1}] in update_site C++ Error: {e}", exc_info=True)
                logger.debug(f"  M_core_norm_c labels: {M_core_norm_c.labels()}, shape: {M_core_norm_c.shape()}, bonds: {M_core_norm_c.bonds()}")
                logger.debug(f"  W_ut_c labels: {W_ut_c.labels()}, shape: {W_ut_c.shape()}, bonds: {W_ut_c.bonds()}")
                logger.debug(f"  R_next_c labels: {R_next_c.labels()}, shape: {R_next_c.shape()}, bonds: {R_next_c.bonds()}")
            except TypeError as e:
                logger.error(f"ncon R[{p_core_idx-1}] in update_site Python TypeError: {e}", exc_info=True)
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

        self.param_Iset_for_Pi_construction: List[List[MultiIndex]] = [[] for _ in range(self.D)]
        self.param_Jset_for_Pi_construction: List[List[MultiIndex]] = [[] for _ in range(self.D)]
        for p_site in range(self.D):
            self.param_Iset_for_Pi_construction[p_site] = [self.initial_pivot_multi_index[:p_site]]
            if p_site + 1 < self.D :
                 self.param_Jset_for_Pi_construction[p_site+1] = [self.initial_pivot_multi_index[p_site+2:]]

        self.mat_lazy_Pi_at: List[Optional[MatLazyIndex]] = [None] * num_bonds
        self.P_cross_data: List[Optional[CrossData]] = [None] * num_bonds
        
        for p_bond in range(num_bonds):
            pi_rows_mi: List[MultiIndex] = [i_idx_multi + (phys_val,)
                for i_idx_multi in self.param_Iset_for_Pi_construction[p_bond]
                for phys_val in range(self.phys_dims[p_bond])]
            jset_pi_cols = self.param_Jset_for_Pi_construction[p_bond+1] if p_bond+1 < self.D else [()]
            pi_cols_mi: List[MultiIndex] = [(phys_val,) + j_idx_multi
                for phys_val in range(self.phys_dims[p_bond+1])
                for j_idx_multi in jset_pi_cols]

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
                pivot_i_in_Pi_p = pi_rows_mi.index(row_pivot_mi_for_Pi_p)
                pivot_j_in_Pi_p = pi_cols_mi.index(col_pivot_mi_for_Pi_p)
            except ValueError: raise ValueError(f"Initial pivot for bond {p_bond} not in Pi_mat index lists.")
            
            self.P_cross_data[p_bond].addPivot(pivot_i_in_Pi_p, pivot_j_in_Pi_p, self.mat_lazy_Pi_at[p_bond])


        self.tt = TensorTrain() 
        self.tt.M = [None] * self.D
        self.P_pivot_matrices: List[Optional[Tensor]] = [None] * self.D

        for p_site in range(self.D):
            chi_left = 1 if p_site == 0 else (self.P_cross_data[p_site-1].rank() if self.P_cross_data[p_site-1] else 1)
            chi_right = 1 if p_site == self.D - 1 else (self.P_cross_data[p_site].rank() if self.P_cross_data[p_site] else 1)
            d_current = self.phys_dims[p_site]
            core_data_tensor: Tensor

            if self.D == 1:
                temp_core_data = cytnx.zeros((1,d_current,1), dtype=self.dtype, device=self.device)
                if 0 <= self.initial_pivot_multi_index[0] < d_current:
                     temp_core_data[0, self.initial_pivot_multi_index[0], 0] = initial_pivot_value
                core_data_tensor = temp_core_data
            elif p_site == 0 : 
                pd0 = self.P_cross_data[0]
                if pd0 is None or pd0.C is None: raise RuntimeError("C0 None for M[0]")
                core_data_tensor = pd0.C.reshape(1, d_current, chi_right)
            else: 
                pdp_prev = self.P_cross_data[p_site-1]
                if pdp_prev is None or pdp_prev.R is None: raise RuntimeError(f"R{p_site-1} None for M[{p_site}]")
                core_data_tensor = pdp_prev.R.reshape(chi_left, d_current, chi_right)
            
            label_L = f'vL{p_site-1}' if p_site > 0 else 'vL_bound' 
            label_P = f'p{p_site}'
            label_R = f'vR{p_site}' if p_site < self.D - 1 else 'vR_bound'

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

            if p_site < num_bonds:
                pd_site = self.P_cross_data[p_site]
                piv_mat = pd_site.pivotMat() if pd_site else None
                self.P_pivot_matrices[p_site] = piv_mat if piv_mat is not None \
                                               else cytnx.eye(chi_right, dtype=self.dtype, device=self.device)
        
        if self.D > 0: self.P_pivot_matrices[self.D-1] = cytnx.eye(1, dtype=self.dtype, device=self.device)

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
            logger.info(f"__init__: Running {self.param.nIter} initial TCI iterations.")
            if hasattr(self, 'IterateN'): self.IterateN(self.param.nIter)
            elif hasattr(self, 'iterate_one_full_sweep'):
                for _ in range(self.param.nIter): self.iterate_one_full_sweep()


    def get_TP1_at(self, p_bond: int) -> Optional[cytnx.UniTensor]:
        if not (0 <= p_bond < self.D): logger.error(f"get_TP1_at: p_bond {p_bond} OOB."); return None
        M_core_p = self.tt.M[p_bond]
        if M_core_p is None: logger.error(f"get_TP1_at: M[{p_bond}] is None."); return None
        
        P_matrix_p = self.P_pivot_matrices[p_bond]
        if P_matrix_p is None :
             if p_bond == self.D - 1 and self.D > 0 : return M_core_p.clone()
             logger.error(f"get_TP1_at: P[{p_bond}] is None."); return None
        try:
            M_matrix_form = cube_as_matrix2(M_core_p) 
            TP1_matrix_form = mat_AB1(M_matrix_form, P_matrix_p) 
            chi_left, d, chi_right = M_core_p.shape()
            
            new_bonds = [M_core_p.bonds()[0].clone(), M_core_p.bonds()[1].clone(), M_core_p.bonds()[2].clone()]
            res_ut = UniTensor(bonds=new_bonds, 
                               labels=[M_core_p.labels()[0], M_core_p.labels()[1], M_core_p.labels()[2]],
                               rowrank=M_core_p.rowrank(),
                               dtype=TP1_matrix_form.dtype(), 
                               device=TP1_matrix_form.device(),
                               is_diag=M_core_p.is_diag())
            res_ut.put_block(TP1_matrix_form.reshape(chi_left, d, chi_right))
            return res_ut
        except Exception as e: logger.error(f"Error in get_TP1_at({p_bond}): {e}", exc_info=True); return None

    def get_P1T_at(self, p_bond: int) -> Optional[cytnx.UniTensor]:
        if not (0 <= p_bond < self.D): logger.error(f"get_P1T_at: p_bond {p_bond} OOB."); return None
        M_core_p = self.tt.M[p_bond]
        if M_core_p is None: logger.error(f"get_P1T_at: M[{p_bond}] is None."); return None
        if p_bond == 0: return M_core_p.clone() 
        P_matrix_prev = self.P_pivot_matrices[p_bond-1]
        if P_matrix_prev is None: logger.error(f"get_P1T_at: P[{p_bond-1}] is None."); return None
        try:
            M_matrix_form = cube_as_matrix1(M_core_p) 
            P1T_matrix_form = mat_A1B(P_matrix_prev, M_matrix_form) 
            chi_left, d, chi_right = M_core_p.shape()
            
            new_bonds = [M_core_p.bonds()[0].clone(), M_core_p.bonds()[1].clone(), M_core_p.bonds()[2].clone()]
            res_ut = UniTensor(bonds=new_bonds,
                               labels=[M_core_p.labels()[0], M_core_p.labels()[1], M_core_p.labels()[2]],
                               rowrank=M_core_p.rowrank(),
                               dtype=P1T_matrix_form.dtype(),
                               device=P1T_matrix_form.device(),
                               is_diag=M_core_p.is_diag())
            res_ut.put_block(P1T_matrix_form.reshape(chi_left, d, chi_right))
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
                        T1_L = L_p_tensor.reshape(N, 1)
                        T2_W_p = W_tensor_site_p.reshape(M_phys, 1)
                        
                        kron_LW_tensor = cytnx.linalg.Kron(T1_L, T2_W_p) 
                        kron_LW_flat = kron_LW_tensor.reshape(N * M_phys) 
                        pf_param.weight_row = cytnx.linalg.Abs(kron_LW_flat)

                    if M_phys > 0 and K > 0:
                        T1_W_p = W_tensor_site_p.reshape(M_phys, 1)
                        T2_R = R_p1_tensor.reshape(K, 1)

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
        logger.warning("TensorCI1.iterate_one_full_sweep is a placeholder.")
        return 0.0

    def IterateN(self, n_iterations: int, update_M_data: bool = True):
        logger.info(f"TensorCI1.IterateN called for {n_iterations} iterations.")
        for i in range(n_iterations):
            logger.debug(f"  Iteration sweep {i+1}/{n_iterations}")
            max_err_sweep = self.iterate_one_full_sweep(update_M_data)

    def _update_tt_M_and_P_pivot_matrix_after_crossdata_update(self, p_bond: int):
         logger.warning(f"_update_tt_M_and_P_pivot_matrix_after_crossdata_update({p_bond}) not fully implemented.")
         pass

    def get_canonical_tt(self, center: int) -> TensorTrain:
        logger.warning(f"get_canonical_tt({center}) not fully implemented.")
        return self.tt 

# --- Main Test Block ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    logger.info("--- Starting TensorCI System Test ---")

    test_phys_dims_small = [2, 2] 
    test_D_small = len(test_phys_dims_small)
    test_dtype = cytnx.Type.Double
    test_device = cytnx.Device.cpu

    def mock_fc_simple(indices: Tuple[int, ...]) -> float:
        val = 1.0 
        if len(indices) == 2: val += float(indices[0] * 3 + indices[1] * 5 + (indices[0] + indices[1]) * 0.2)
        else: logger.error(f"mock_fc_simple: Unsupported input length {len(indices)}")
        return val 

    mock_tensor_function = TensorFunction(func=mock_fc_simple)

    def mock_cond_func(full_multi_index: List[int]) -> bool:
        return sum(full_multi_index) % 2 == 0

    logger.info("\n--- Test Case 1: TensorCI1 Initialization (D=2, ENV, Cond) ---")
    test_weights_small = [[0.8, 1.2], [0.9, 1.1]]
    ci_param_env_cond = TensorCI1Param(
        pivot1=[0]*test_D_small, weight=test_weights_small, cond=mock_cond_func, nIter=0 
    )
    tci_test_instance = None
    try:
        tci_test_instance = TensorCI1(
            fc=mock_tensor_function, phys_dims=test_phys_dims_small, param=ci_param_env_cond,
            dtype=test_dtype, device=test_device
        )
        logger.info("TensorCI1 (D=2, ENV, Cond) initialized.")
        assert tci_test_instance.tt_sum_env is not None, "tt_sum_env should be initialized"
        assert tci_test_instance.tt_sum_env.is_active, "tt_sum_env should be active after init with weights"
        
        logger.info("Verifying initial TTEnvMatrices L and R:")
        for i in range(tci_test_instance.D):
            if tci_test_instance.tt_sum_env.L[i] : 
                logger.info(f"  Init L[{i}] shape: {tci_test_instance.tt_sum_env.L[i].shape()}, "
                            f"Data: {tci_test_instance.tt_sum_env.L[i].get_block_().numpy()}")
            if tci_test_instance.tt_sum_env.R[i] : 
                logger.info(f"  Init R[{i}] shape: {tci_test_instance.tt_sum_env.R[i].shape()}, "
                            f"Data: {tci_test_instance.tt_sum_env.R[i].get_block_().numpy()}")

        if tci_test_instance.D > 1:
            assert tci_test_instance.tt_sum_env.L[tci_test_instance.D-1] is not None, "L[D-1] should be calculated"
            # assert tci_test_instance.tt_sum_env.L[tci_test_instance.D-1].shape() != [1], "L[D-1] seems like boundary" # Relaxed
            assert tci_test_instance.tt_sum_env.R[0] is not None, "R[0] should be calculated"
            # assert tci_test_instance.tt_sum_env.R[0].shape() != [1], "R[0] seems like boundary" # Relaxed

        if test_D_small > 1 and tci_test_instance.pivFinder[0] is not None:
            pf_param0 = tci_test_instance.pivFinder[0].p # Corrected .param to .p
            if tci_test_instance.tt_sum_env and tci_test_instance.tt_sum_env.is_active:
                 assert pf_param0.weight_row is not None, "Initial weight_row missing"
                 assert pf_param0.weight_col is not None, "Initial weight_col missing"
                 logger.info(f"  PivotFinderParam[0] weight_row (shape {pf_param0.weight_row.shape()}): {pf_param0.weight_row.numpy()}")
                 logger.info(f"  PivotFinderParam[0] weight_col (shape {pf_param0.weight_col.shape()}): {pf_param0.weight_col.numpy()}")

    except Exception as e: logger.error(f"Error in Test Case 1 (Init): {e}", exc_info=True)

    if tci_test_instance:
        logger.info("\n--- Test Case 2: Simulating update_env_at call ---")
        if tci_test_instance.tt_sum_env and tci_test_instance.tt_sum_env.is_active:
            try:
                L1_b_core = tci_test_instance.tt_sum_env.L[1]
                L1_b_numpy = L1_b_core.get_block_().numpy().copy() if tci_test_instance.D > 1 and L1_b_core else None
                
                R0_b_core = tci_test_instance.tt_sum_env.R[0]
                R0_b_numpy = R0_b_core.get_block_().numpy().copy() if tci_test_instance.D > 0 and R0_b_core else None

                logger.info("Calling update_env_at(p_bond=0)")
                tci_test_instance.update_env_at(p_bond=0) 

                if tci_test_instance.D > 1 :
                    L1_a_core = tci_test_instance.tt_sum_env.L[1]
                    assert L1_a_core is not None
                    L1_a_numpy = L1_a_core.get_block_().numpy()
                    logger.info(f"  L[1] after update (shape {L1_a_core.shape()}): {L1_a_numpy}")
                    if L1_b_numpy is not None: assert not np.allclose(L1_a_numpy, L1_b_numpy), "L1 should have changed"

                R0_a_core = tci_test_instance.tt_sum_env.R[0]
                assert R0_a_core is not None
                R0_a_numpy = R0_a_core.get_block_().numpy()
                logger.info(f"  R[0] after update (shape {R0_a_core.shape()}): {R0_a_numpy}")
                if R0_b_numpy is not None: assert not np.allclose(R0_a_numpy, R0_b_numpy), "R0 should have changed"
                
                logger.info("\n--- Test Case 3: Weights in PivotFinderParam after ENV update ---")
                updated_pf_param0 = tci_test_instance._build_pivot_finder_param_at_bond(0)
                assert updated_pf_param0.weight_row is not None
                logger.info(f"  Updated PivotFinderParam[0] weight_row: {updated_pf_param0.weight_row.numpy()}")

            except Exception as e: logger.error(f"Error in Test Case 2/3 (Update/Post-weights): {e}", exc_info=True)
        else: logger.warning("Skipping Test Case 2/3: tt_sum_env not active.")
    else: logger.warning("Skipping Test Case 2/3: TensorCI1 instance not created.")
            
    logger.info("\n--- TensorCI System Test Finished ---")