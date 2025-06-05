import cytnx
from cytnx import linalg
import pickle
import numpy as np
from itertools import product
import logging
from typing import List, Any, Union, Optional
from mat_decomp import MatprrLUFixedTol

# Initialize logger
logger = logging.getLogger(__name__)

# Set logging level to INFO by default. For more detailed debug, set to logging.DEBUG in main block.
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # Moved to __main__ for better control

# Cytnx core imports
from cytnx import Tensor, UniTensor, Type, BD_IN, BD_OUT, from_numpy

def _utt_assert_rank3(A_ut: cytnx.UniTensor, func_name: str):
    """
    Asserts the input UniTensor is rank 3 and is a valid UniTensor instance.
    """
    logger.debug(f"DEBUG: Entering _utt_assert_rank3 for {func_name}. UniTensor type: {type(A_ut)}") # ADDED DEBUG
    if not isinstance(A_ut, cytnx.UniTensor):
        logger.error(f"ERROR: {func_name} expects a cytnx.UniTensor, got {type(A_ut)}") # ADDED ERROR
        raise TypeError(f"{func_name} expects a cytnx.UniTensor, got {type(A_ut)}")
    
    if A_ut.rank() != 3:
        logger.error(f"ERROR: {func_name} expects a rank-3 UniTensor, got rank {A_ut.rank()}. Labels: {A_ut.labels()}") # ADDED ERROR
        raise ValueError(f"{func_name} expects a rank-3 UniTensor, got rank {A_ut.rank()}. Labels: {A_ut.labels()}")
    if len(A_ut.labels()) != 3:
        logger.error(f"ERROR: {func_name} expects a UniTensor with 3 labels, got {len(A_ut.labels())}: {A_ut.labels()}") # ADDED ERROR
        raise ValueError(f"{func_name} expects a UniTensor with 3 labels, got {len(A_ut.labels())}: {A_ut.labels()}")
    logger.debug(f"DEBUG: _utt_assert_rank3 passed for {func_name}. UniTensor labels: {A_ut.labels()}, shape: {A_ut.shape()}") # ADDED DEBUG

def cube_as_matrix1_utt(A_ut: cytnx.UniTensor) -> cytnx.Tensor:
    """
    Takes a rank-3 UniTensor A_ut (conventionally with labels L,P,R for Left, Physical, Right)
    and effectively combines its 2nd (Physical) and 3rd (Right) bonds to form 
    the columns of a matrix, while the 1st bond (Left) forms the rows.

    Args:
        A_ut (cytnx.UniTensor): The input rank-3 UniTensor. 
                               It is assumed that its labels are ordered [L, P, R]
                               where L is the bond to become the row index, and P, R
                               are to be combined into the column index.

    Returns:
        cytnx.Tensor: A rank-2 cytnx.Tensor of shape (dim_L, dim_P * dim_R).
    """
    logger.debug(f"DEBUG: Entering cube_as_matrix1_utt. Input UniTensor labels: {A_ut.labels()}, shape: {A_ut.shape()}") # ADDED DEBUG
    _utt_assert_rank3(A_ut, "cube_as_matrix1_utt")
    
    original_labels = A_ut.labels()
    label_L, label_P, label_R = original_labels[0], original_labels[1], original_labels[2]
    
    A_permuted = A_ut.permute([label_L, label_P, label_R], rowrank=1)
    A_permuted.contiguous_() 

    block = A_permuted.get_block_() 
    
    dim_L_val = block.shape()[0]
    dim_P_val = block.shape()[1]
    dim_R_val = block.shape()[2]
    
    reshaped_tensor = block.reshape(dim_L_val, dim_P_val * dim_R_val)
    logger.debug(f"DEBUG: cube_as_matrix1_utt output shape: {reshaped_tensor.shape()}") # ADDED DEBUG
    return reshaped_tensor

def cube_as_matrix2_utt(A_ut: cytnx.UniTensor) -> cytnx.Tensor:
    """
    Takes a rank-3 UniTensor A_ut (conventionally with labels L,P,R) and effectively 
    combines its 1st (Left) and 2nd (Physical) bonds to form the rows of a matrix,
    while the 3rd bond (Right) forms the columns.

    Args:
        A_ut (cytnx.UniTensor): The input rank-3 UniTensor.
                               It is assumed that its labels are ordered [L, P, R]
                               where L, P are to be combined into the row index, 
                               and R becomes the column index.
    Returns:
        cytnx.Tensor: A rank-2 cytnx.Tensor of shape (dim_L * dim_P, dim_R).
    """
    logger.debug(f"DEBUG: Entering cube_as_matrix2_utt. Input UniTensor labels: {A_ut.labels()}, shape: {A_ut.shape()}") # ADDED DEBUG
    _utt_assert_rank3(A_ut, "cube_as_matrix2_utt")

    original_labels = A_ut.labels()
    label_L, label_P, label_R = original_labels[0], original_labels[1], original_labels[2]

    dim_L_val = A_ut.bond(label_L).dim()
    dim_P_val = A_ut.bond(label_P).dim()
    dim_R_val = A_ut.bond(label_R).dim()

    A_permuted = A_ut.permute([label_L, label_P, label_R], rowrank=2)
    A_permuted.contiguous_()

    block = A_permuted.get_block_() 
    
    reshaped_tensor = block.reshape(dim_L_val * dim_P_val, dim_R_val)
    logger.debug(f"DEBUG: cube_as_matrix2_utt output shape: {reshaped_tensor.shape()}") # ADDED DEBUG
    return reshaped_tensor


class TensorTrain:

    def __init__(self, M: Optional[List[Union[Tensor, UniTensor]]] = None):
        logger.info(f"INFO: TensorTrain.__init__ called with {len(M) if M is not None else 0} cores.") # ADDED INFO
        self.M: List[UniTensor] = []

        if M is None or not M:
            logger.warning("WARNING: TensorTrain initialized with no cores.") # ADDED WARNING
            return

        target_device = M[0].device()

        final_requires_complex = False
        final_requires_double_precision = False

        for core_input_scan in M:
            if isinstance(core_input_scan, UniTensor):
                dt_int = core_input_scan.dtype()
                if dt_int == Type.ComplexDouble or dt_int == Type.ComplexFloat:
                    final_requires_complex = True
                if dt_int == Type.Double or dt_int == Type.ComplexDouble:
                    final_requires_double_precision = True
            elif isinstance(core_input_scan, Tensor):
                dt_int = core_input_scan.dtype()
                if dt_int % 2 == 1:
                    final_requires_complex = True
                if dt_int == Type.Double or dt_int == Type.ComplexDouble:
                    final_requires_double_precision = True
            else:
                logger.error(f"ERROR: Input core list M must contain cytnx.Tensor or cytnx.UniTensor objects. Got {type(core_input_scan)}") # ADDED ERROR
                raise TypeError(
                    f"Input core list M must contain cytnx.Tensor or cytnx.UniTensor objects. "
                    f"Got {type(core_input_scan)}"
                )
            if final_requires_complex and final_requires_double_precision:
                break
        
        if final_requires_complex:
            target_dtype = Type.ComplexDouble if final_requires_double_precision else Type.ComplexFloat
        else:
            target_dtype = Type.Double if final_requires_double_precision else Type.Float
        logger.debug(f"DEBUG: Determined target dtype: {target_dtype} and device: {target_device} for TensorTrain.") # ADDED DEBUG

        num_cores = len(M)
        for k, core_input in enumerate(M):
            label_left = "L_bound" if k == 0 else f"link{k-1}"
            label_phys = f"p{k}"
            label_right = "R_bound" if k == num_cores - 1 else f"link{k}"
            expected_labels = [label_left, label_phys, label_right]

            processed_core: UniTensor

            if isinstance(core_input, Tensor):
                logger.debug(f"DEBUG: Processing Tensor input for core {k}. Original shape: {core_input.shape()}, dtype: {core_input.dtype_str()}") # ADDED DEBUG
                tensor_core = core_input.contiguous().to(target_device).astype(target_dtype)
                
                if len(tensor_core.shape()) != 3:
                    logger.error(f"ERROR: Input Tensor at index {k} must be 3D, got shape {tensor_core.shape()}") # ADDED ERROR
                    raise ValueError(f"Input Tensor at index {k} must be 3D, got shape {tensor_core.shape()}")

                r_k, d_k, r_k1 = tensor_core.shape()[0], tensor_core.shape()[1], tensor_core.shape()[2]

                if k == 0 and r_k != 1:
                    logger.warning(f"WARNING: Input Tensor at index 0 ('{label_left}') has left bond dim {r_k}, expected 1.") # ADDED WARNING
                if k == num_cores - 1 and r_k1 != 1:
                    logger.warning(f"WARNING: Input Tensor at index {k} ('{label_right}') has right bond dim {r_k1}, expected 1.") # ADDED WARNING
                
                if k > 0 and self.M:
                    prev_core_right_bond_dim = self.M[k-1].bonds()[2].dim() 
                    if r_k != prev_core_right_bond_dim:
                        logger.error(f"ERROR: Bond dimension mismatch for link 'link{k-1}': Tensor core {k} (left_dim={r_k}) vs UniTensor core {k-1} (right_dim={prev_core_right_bond_dim}).") # ADDED ERROR
                        raise ValueError(
                            f"Bond dimension mismatch for link 'link{k-1}': Tensor core {k} (left_dim={r_k}) "
                            f"vs UniTensor core {k-1} (right_dim={prev_core_right_bond_dim})."
                        )
                
                bd_left = cytnx.Bond(r_k, BD_IN)
                bd_phys = cytnx.Bond(d_k, BD_OUT)
                bd_right = cytnx.Bond(r_k1, BD_OUT)
                
                processed_core = UniTensor(bonds=[bd_left, bd_phys, bd_right], 
                                           labels=expected_labels, 
                                           rowrank=1) 
                processed_core.put_block(tensor_core)
                logger.debug(f"DEBUG: Core {k} (Tensor) processed. Labels: {processed_core.labels()}, Shape: {processed_core.shape()}") # ADDED DEBUG
                
            elif isinstance(core_input, UniTensor):
                logger.debug(f"DEBUG: Processing UniTensor input for core {k}. Original labels: {core_input.labels()}, shape: {core_input.shape()}, dtype: {core_input.dtype_str()}") # ADDED DEBUG
                ut_temp = core_input.contiguous().to(target_device).astype(target_dtype)

                if ut_temp.rank() != 3:
                    logger.error(f"ERROR: Input UniTensor at index {k} must be rank 3, got rank {ut_temp.rank()}") # ADDED ERROR
                    raise ValueError(f"Input UniTensor at index {k} must be rank 3, got rank {ut_temp.rank()}")

                bonds_temp = ut_temp.bonds()
                r_k, d_k, r_k1 = bonds_temp[0].dim(), bonds_temp[1].dim(), bonds_temp[2].dim()

                if k == 0 and r_k != 1:
                    logger.warning(f"WARNING: Input UniTensor at index 0 ('{label_left}') has left bond dim {r_k}, expected 1.") # ADDED WARNING
                if k == num_cores - 1 and r_k1 != 1:
                    logger.warning(f"WARNING: Input UniTensor at index {k} ('{label_right}') has right bond dim {r_k1}, expected 1.") # ADDED WARNING

                if k > 0 and self.M:
                    prev_core_right_bond_dim = self.M[k-1].bonds()[2].dim() 
                    if r_k != prev_core_right_bond_dim:
                        logger.error(f"ERROR: Bond dimension mismatch for link 'link{k-1}': UniTensor core {k} (left_dim={r_k}) vs UniTensor core {k-1} (right_dim={prev_core_right_bond_dim}).") # ADDED ERROR
                        raise ValueError(
                            f"Bond dimension mismatch for link 'link{k-1}': UniTensor core {k} (left_dim={r_k}) "
                            f"vs UniTensor core {k-1} (right_dim={prev_core_right_bond_dim})."
                        )
                
                ut_temp.set_labels(expected_labels)
                processed_core = ut_temp
                logger.debug(f"DEBUG: Core {k} (UniTensor) processed. Labels: {processed_core.labels()}, Shape: {processed_core.shape()}") # ADDED DEBUG
            else: 
                logger.error(f"ERROR: Input core at index {k} is not a cytnx.Tensor or cytnx.UniTensor. Got {type(core_input)}") # ADDED ERROR
                raise TypeError(f"Input core at index {k} is not a cytnx.Tensor or cytnx.UniTensor. Got {type(core_input)}")

            try:
                logger.debug(f"DEBUG: Calling _assert_core_validity for core {k} in __init__.") # Original print, now using logger
                TensorTrain._assert_core_validity(processed_core, k, num_cores, check_data_presence=True)
            except Exception as e:
                import traceback
                logger.critical(f"CRITICAL ERROR from _assert_core_validity for core {k} during __init__:") # ADDED CRITICAL
                logger.critical(f"Error Type: {type(e).__name__}")
                logger.critical(f"Error Message: {e}")
                logger.critical("Traceback:\n" + traceback.format_exc()) # ADDED CRITICAL
                raise

            self.M.append(processed_core)
    
    @staticmethod
    def _assert_core_validity(core: UniTensor, k: int, L: int, check_data_presence: bool = False):
        logger.debug(f"DEBUG: Entering _assert_core_validity for core {k}. Current labels: {core.labels()}, shape: {core.shape()}") # ADDED DEBUG
        if not isinstance(core, UniTensor):
            logger.error(f"ERROR: Core at site {k} must be a cytnx.UniTensor, got {type(core)}.") # ADDED ERROR
            raise TypeError(f"Core at site {k} must be a cytnx.UniTensor, got {type(core)}.")
        if core.rank() != 3:
            logger.error(f"ERROR: Core at site {k} is not rank 3, got rank {core.rank()}. Labels: {core.labels()}") # ADDED ERROR
            raise ValueError(f"Core at site {k} is not rank 3, got rank {core.rank()}. Labels: {core.labels()}")

        enum_accessible_for_direct_comparison = False
        try:
            _ = cytnx.bondType.BD_KET
            logger.debug(f"DEBUG (Core {k}): Path cytnx.bondType.BD_KET is accessible.") # Original print, now using logger
            enum_accessible_for_direct_comparison = True
        except AttributeError as e:
            logger.debug(f"DEBUG (Core {k}): Failed to access cytnx.bondType.BD_KET directly via attribute path: {e}") # Original print, now using logger
            logger.debug(f"DEBUG (Core {k}): Will rely solely on .value comparison for bond types.") # Original print, now using logger

        exp_label_left = "L_bound" if k == 0 else f"link{k-1}"
        exp_label_phys = f"p{k}"
        exp_label_right = "R_bound" if k == L - 1 else f"link{k}"
        expected_labels = [exp_label_left, exp_label_phys, exp_label_right]
        current_labels = core.labels()
        if current_labels != expected_labels:
            logger.error(f"ERROR: Core at site {k} has incorrect labels. Expected {expected_labels}, got {current_labels}.") # ADDED ERROR
            raise ValueError(
                f"Core at site {k} has incorrect labels. Expected {expected_labels}, got {current_labels}."
            )

        bonds = core.bonds()
        current_bond_types_retrieved = [b.type() for b in bonds]

        expected_bond_type_values = [-1, 1, 1]  # KET, BRA, BRA

        for i in range(3):
            retrieved_type_obj = current_bond_types_retrieved[i]
            if not (hasattr(retrieved_type_obj, 'value') and retrieved_type_obj.value == expected_bond_type_values[i]):
                error_msg = (
                    f"Core at site {k} (label: {current_labels[i]}) has incorrect bond type for bond {i}. "
                    f"Expected numeric value {expected_bond_type_values[i]}, "
                    f"got object {retrieved_type_obj}"
                )
                if hasattr(retrieved_type_obj, 'value'):
                    error_msg += f" with value {retrieved_type_obj.value}."
                else:
                    error_msg += " which does not have a '.value' attribute."
                logger.error(f"ERROR: {error_msg}") # ADDED ERROR
                raise ValueError(error_msg)
        
        logger.debug(f"DEBUG (Core {k}): Bond type .value checks PASSED. Current bond types: {[b.type() for b in bonds]}") # Original print, now using logger

        if enum_accessible_for_direct_comparison:
            try:
                expected_enums = [cytnx.bondType.BD_KET, cytnx.bondType.BD_BRA, cytnx.bondType.BD_BRA]
                if current_bond_types_retrieved[0] != expected_enums[0] or \
                   current_bond_types_retrieved[1] != expected_enums[1] or \
                   current_bond_types_retrieved[2] != expected_enums[2]:
                    logger.debug(f"DEBUG (Core {k}): Direct enum object comparison MISMATCHED, even if .value comparison passed. Left: {current_bond_types_retrieved[0]} vs {expected_enums[0]}, Phys: {current_bond_types_retrieved[1]} vs {expected_enums[1]}, Right: {current_bond_types_retrieved[2]} vs {expected_enums[2]}") # Original print, now using logger
                else:
                    logger.debug(f"DEBUG (Core {k}): Direct enum object comparison PASSED.") # Original print, now using logger
            except Exception as e_direct_comp:
                 logger.debug(f"DEBUG (Core {k}): Error during optional direct enum object comparison: {e_direct_comp}") # Original print, now using logger

        if core.rowrank() != 1:
            logger.error(f"ERROR: Core at site {k} (labels: {current_labels}) has incorrect rowrank. Expected 1, got {core.rowrank()}.") # ADDED ERROR
            raise ValueError(
                f"Core at site {k} (labels: {current_labels}) has incorrect rowrank. Expected 1, got {core.rowrank()}."
            )
        
        if k == 0 and bonds[0].dim() != 1:
            logger.warning(f"WARNING (Core {k}): Assertion: Core at site 0 ('{exp_label_left}') has left bond_dim {bonds[0].dim()}, expected 1.") # Original print, now using logger
        if k == L - 1 and bonds[2].dim() != 1:
            logger.warning(f"WARNING (Core {k}): Assertion: Core at site {L - 1} ('{exp_label_right}') has right bond_dim {bonds[2].dim()}, expected 1.") # Original print, now using logger

        if check_data_presence:
            logger.debug(f"DEBUG (Core {k}): Performing check_data_presence.") # Original print, now using logger
            try:
                block_accessed = core.get_block_() 
                logger.debug(f"DEBUG (Core {k}): core.get_block_() succeeded for data presence check. Block shape: {block_accessed.shape()}") # Original print, now using logger

            except RuntimeError as e: 
                error_msg = (
                    f"Core at site {k} (labels: {core.labels()}) data block access error (RuntimeError): {e}. "
                    f"This might indicate the UniTensor is uninitialized or 'void'."
                )
                logger.error(f"ERROR: {error_msg}") # ADDED ERROR
                raise ValueError(error_msg) from e
        
        logger.debug(f"DEBUG (Core {k}): All checks in _assert_core_validity PASSED.") # Original print, now using logger


    def eval(self, idx: List[int]) -> Any:
        logger.debug(f"DEBUG: Entering eval. Index: {idx}") # ADDED DEBUG
        if not self.M:
            if not idx:
                logger.warning("WARNING: Eval on empty TensorTrain with empty index, returning 1.0.") # ADDED WARNING
                return 1.0
            else:
                logger.error("ERROR: Cannot evaluate non-empty idx on an empty TensorTrain.") # ADDED ERROR
                raise ValueError("Cannot evaluate non-empty idx on an empty TensorTrain.")

        if len(idx) != len(self.M):
            logger.error(f"ERROR: Index length {len(idx)} does not match tensor train length {len(self.M)}.") # ADDED ERROR
            raise ValueError(
                f"Index length {len(idx)} does not match tensor train length {len(self.M)}."
            )

        num_cores = len(self.M)
        comp_device = self.M[0].device()
        comp_dtype = self.M[0].dtype()
        logger.debug(f"DEBUG: Eval: Using device {comp_device}, dtype {comp_dtype}.") # ADDED DEBUG

        first_core_left_bond_label = self.M[0].labels()[0]
        
        if self.M[0].bonds()[0].dim() != 1:
            logger.error(f"ERROR: Left boundary bond of the first core ('{first_core_left_bond_label}') must have dimension 1 for eval.") # ADDED ERROR
            raise ValueError(f"Left boundary bond of the first core ('{first_core_left_bond_label}') must have dimension 1 for eval.")

        prod_ut_bond = cytnx.Bond(1, BD_OUT)
        prod_ut = UniTensor(bonds=[prod_ut_bond], 
                            labels=[first_core_left_bond_label], 
                            rowrank=1,
                            device=comp_device, 
                            dtype=comp_dtype)
        prod_ut.get_block_()[0] = 1.0 # Corrected from blk[0] = 1.0 assuming prod_ut is the block itself

        for k, Mk_ut in enumerate(self.M):
            logger.debug(f"DEBUG: Eval loop core {k}. Current prod_ut labels: {prod_ut.labels()}, shape: {prod_ut.shape()}") # ADDED DEBUG
            # TensorTrain._assert_core_validity(Mk_ut, k, num_cores) # Optional, re-enable for deep debugging

            phys_bond_on_Mk = Mk_ut.bonds()[1]
            phys_label_on_Mk = Mk_ut.labels()[1]
            
            i_k = idx[k]
            if not (0 <= i_k < phys_bond_on_Mk.dim()):
                logger.error(f"ERROR: Index {i_k} out of bounds for physical dimension {phys_bond_on_Mk.dim()} at site {k} (label '{phys_label_on_Mk}').") # ADDED ERROR
                raise IndexError(
                    f"Index {i_k} out of bounds for physical dimension {phys_bond_on_Mk.dim()} "
                    f"at site {k} (label '{phys_label_on_Mk}')."
                )

            selector_bond = phys_bond_on_Mk.redirect()
            
            data_for_selector = cytnx.zeros(phys_bond_on_Mk.dim(), dtype=comp_dtype, device=comp_device)
            data_for_selector[i_k] = 1.0
            selector_ut = UniTensor(bonds=[selector_bond],
                                    labels=[phys_label_on_Mk],
                                    rowrank=0,
                                    dtype=comp_dtype,
                                    device=comp_device)
            selector_ut.put_block(data_for_selector)
            logger.debug(f"DEBUG: Eval core {k}: Selector created with label {phys_label_on_Mk}, value at index {i_k}.") # ADDED DEBUG


            Mk_selected = cytnx.Contract(Mk_ut, selector_ut)
            logger.debug(f"DEBUG: Eval core {k}: Mk_selected (Mk_ut * selector_ut) labels: {Mk_selected.labels()}, shape: {Mk_selected.shape()}") # ADDED DEBUG
            
            prod_ut = cytnx.Contract(prod_ut, Mk_selected)
            logger.debug(f"DEBUG: Eval core {k}: prod_ut updated. New labels: {prod_ut.labels()}, shape: {prod_ut.shape()}") # ADDED DEBUG

        if not (prod_ut.rank() == 1 and prod_ut.shape() == [1]):
            final_shape = prod_ut.shape() if prod_ut.rank() > 0 else "scalar (rank 0)"
            final_labels = prod_ut.labels()
            logger.error(f"ERROR: TensorTrain.eval did not collapse to a rank-1 UniTensor of shape [1]. Final rank: {prod_ut.rank()}, shape: {final_shape}, labels: {final_labels}.") # ADDED ERROR
            raise ValueError(
                f"TensorTrain.eval did not collapse to a rank-1 UniTensor of shape [1]. "
                f"Final rank: {prod_ut.rank()}, shape: {final_shape}, labels: {final_labels}."
            )
        
        if self.M[-1].bonds()[2].dim() != 1:
            logger.error(f"ERROR: Right boundary bond of the last core ('{self.M[-1].labels()[2]}') must have dimension 1 for eval to result in a scalar.") # ADDED ERROR
            raise ValueError(f"Right boundary bond of the last core ('{self.M[-1].labels()[2]}') must have dimension 1 for eval to result in a scalar.")

        result = prod_ut.item()
        logger.debug(f"DEBUG: Eval finished. Result: {result}") # ADDED DEBUG
        return result


    def overlap(self, tt: 'TensorTrain') -> float:
        logger.info(f"INFO: Entering overlap operation. self length: {len(self.M)}, other TT length: {len(tt.M)}") # ADDED INFO
        if len(self.M) != len(tt.M):
            logger.error("ERROR: Tensor trains for overlap must have the same length.") # ADDED ERROR
            raise ValueError("Tensor trains for overlap must have the same length.")
        if not self.M:
            logger.warning("WARNING: Overlap on empty TensorTrains, returning 1.0.") # ADDED WARNING
            return 1.0 

        num_cores = len(self.M)
        comp_device = self.M[0].device()

        final_requires_complex_for_C, final_requires_double_for_C = False, False
        for k_check in range(num_cores):
            for core_scan in [self.M[k_check], tt.M[k_check]]:
                dt_cs = core_scan.dtype()
                if dt_cs == Type.ComplexFloat or dt_cs == Type.ComplexDouble:
                    final_requires_complex_for_C = True
                if dt_cs == Type.Double or dt_cs == Type.ComplexDouble:
                    final_requires_double_for_C = True
                if final_requires_complex_for_C and final_requires_double_for_C: break
            if final_requires_complex_for_C and final_requires_double_for_C: break
        
        if final_requires_complex_for_C:
            comp_dtype_C = Type.ComplexDouble if final_requires_double_for_C else Type.ComplexFloat
        else:
            comp_dtype_C = Type.Double if final_requires_double_for_C else Type.Float
        logger.debug(f"DEBUG: Overlap: Determined composite dtype: {comp_dtype_C}, device: {comp_device}.") # ADDED DEBUG

        actual_L_label_A = self.M[0].labels()[0] 
        actual_L_label_B = tt.M[0].labels()[0]  

        if self.M[0].bonds()[0].dim() != 1 or tt.M[0].bonds()[0].dim() != 1:
            logger.error("ERROR: Left boundary bonds for overlap must have dimension 1.") # ADDED ERROR
            raise ValueError("Left boundary bonds for overlap must have dimension 1.")

        c_env_label_A = "__c_env_A_leg__"
        c_env_label_B = "__c_env_B_leg__"

        C_ut_bonds = [cytnx.Bond(1, BD_OUT), cytnx.Bond(1, BD_OUT)]
        C_ut = UniTensor(bonds=C_ut_bonds, 
                         labels=[c_env_label_A, c_env_label_B], 
                         rowrank=1, 
                         device=comp_device, 
                         dtype=comp_dtype_C)
        C_ut.get_block_()[0,0] = 1.0
        logger.debug(f"DEBUG: Overlap: Initial C_ut labels: {C_ut.labels()}, shape: {C_ut.shape()}. Value: {C_ut.item()}") # ADDED DEBUG

        for k in range(num_cores):
            logger.debug(f"DEBUG: Overlap loop core {k}.") # ADDED DEBUG
            Ak_ut_orig = self.M[k] 
            Bk_ut_orig = tt.M[k]   

            Ak_ut = Ak_ut_orig.astype(comp_dtype_C).to(comp_device)
            Bk_ut = Bk_ut_orig.astype(comp_dtype_C).to(comp_device)
            Ak_c = Ak_ut.Conj() 
            logger.debug(f"DEBUG: Overlap core {k}: Ak_ut_orig labels {Ak_ut_orig.labels()}, Ak_ut labels {Ak_ut.labels()}, Ak_c labels {Ak_c.labels()}") # ADDED DEBUG
            
            akc_actual_L_lbl = Ak_c.labels()[0] 
            akc_actual_P_lbl = Ak_c.labels()[1]
            akc_actual_R_lbl = Ak_c.labels()[2] 

            bk_actual_L_lbl = Bk_ut.labels()[0]
            bk_actual_P_lbl = Bk_ut.labels()[1]
            bk_actual_R_lbl = Bk_ut.labels()[2]

            Ak_c_ready = Ak_c.relabel(akc_actual_L_lbl, c_env_label_A)
            temp_CA = cytnx.Contract(C_ut, Ak_c_ready)
            logger.debug(f"DEBUG: Overlap core {k}: temp_CA (C_ut * Ak_c_ready) labels: {temp_CA.labels()}, shape: {temp_CA.shape()}") # ADDED DEBUG

            Bk_mod = Bk_ut.clone() 
            phys_bond_idx = 1 
            Bk_mod.bonds()[phys_bond_idx].redirect_() 

            unique_R_A_label_for_output = f"__temp_out_A_{k}__"
            unique_R_B_label_for_output = f"__temp_out_B_{k}__"

            temp_CA_relabeled_for_Bk_contract = temp_CA.relabel(akc_actual_R_lbl, unique_R_A_label_for_output)
            
            Bk_mod_ready_for_final_contract = Bk_mod.relabel(bk_actual_L_lbl, c_env_label_B)
            Bk_mod_ready_for_final_contract.relabel_(bk_actual_P_lbl, akc_actual_P_lbl)
            Bk_mod_ready_for_final_contract.relabel_(bk_actual_R_lbl, unique_R_B_label_for_output)
            logger.debug(f"DEBUG: Overlap core {k}: Bk_mod_ready_for_final_contract labels: {Bk_mod_ready_for_final_contract.labels()}, shape: {Bk_mod_ready_for_final_contract.shape()}") # ADDED DEBUG
            
            C_ut_next_data_container = cytnx.Contract(temp_CA_relabeled_for_Bk_contract, Bk_mod_ready_for_final_contract)
            logger.debug(f"DEBUG: Overlap core {k}: C_ut_next_data_container (temp_CA * Bk_mod) labels: {C_ut_next_data_container.labels()}, shape: {C_ut_next_data_container.shape()}") # ADDED DEBUG

            if k < num_cores - 1:
                next_bond_dim_A = Ak_ut_orig.bonds()[2].dim() 
                next_bond_dim_B = Bk_ut_orig.bonds()[2].dim() 

                next_C_bonds = [cytnx.Bond(next_bond_dim_A, BD_OUT), 
                                cytnx.Bond(next_bond_dim_B, BD_OUT)]
                
                C_next_iter_shell = UniTensor(bonds=next_C_bonds,
                                              labels=[c_env_label_A, c_env_label_B],
                                              rowrank=1, 
                                              device=comp_device, 
                                              dtype=comp_dtype_C)
                
                C_ut_next_data_container.permute_([unique_R_A_label_for_output, unique_R_B_label_for_output], 
                                                  rowrank=1) 
                
                C_next_iter_shell.put_block(C_ut_next_data_container.get_block_())
                C_ut = C_next_iter_shell
                logger.debug(f"DEBUG: Overlap core {k}: C_ut updated for next iter. New labels: {C_ut.labels()}, shape: {C_ut.shape()}") # ADDED DEBUG
            else: 
                C_ut = C_ut_next_data_container
                logger.debug(f"DEBUG: Overlap final C_ut labels: {C_ut.labels()}, shape: {C_ut.shape()}") # ADDED DEBUG
        
        if not (C_ut.rank() == 2 and C_ut.shape() == [1,1] and C_ut.bonds()[0].dim()==1 and C_ut.bonds()[1].dim()==1):
             final_shape = C_ut.shape(); final_labels = C_ut.labels()
             logger.error(f"ERROR: Overlap: Result not 1x1. Rank={C_ut.rank()}, Shape={final_shape}, Labels={final_labels}.") # ADDED ERROR
             raise ValueError(f"Overlap: Result not 1x1. Rank={C_ut.rank()}, Shape={final_shape}, Labels={final_labels}.")
        final_scalar_tensor = C_ut.get_block_()[0,0]
        result = float(final_scalar_tensor.item())
        logger.info(f"INFO: Overlap finished. Result: {result}") # ADDED INFO
        return result

    def norm2(self) -> float:
        logger.info("INFO: Entering norm2 operation.") # ADDED INFO
        result = self.overlap(self)
        logger.info(f"INFO: Norm2 finished. Result: {result}") # ADDED INFO
        return result

    def compressLU(self, reltol: float = 1e-12, maxBondDim: int = 0):
        logger.info(f"INFO: Entering compressLU with reltol={reltol}, maxBondDim={maxBondDim}.") # ADDED INFO
        if not hasattr(self, 'M') or len(self.M) < 2:
            logger.warning("WARNING: compressLU called on empty or too short TensorTrain. Skipping.") # ADDED WARNING
            return

        num_cores = len(self.M)
        try:
            decomp = MatprrLUFixedTol(tol=reltol, rankMax=maxBondDim if maxBondDim > 0 else 0)
            logger.debug("DEBUG: MatprrLUFixedTol decomposer created.") # ADDED DEBUG
        except NameError:
            logger.error("ERROR: MatprrLUFixedTol not found. Please ensure it's imported/defined.") # ADDED ERROR
            raise

        comp_dtype = self.M[0].dtype()
        comp_device = self.M[0].device()
        logger.debug(f"DEBUG: CompressLU: Using device {comp_device}, dtype {comp_dtype}.") # ADDED DEBUG

        for i in range(num_cores - 1):
            logger.debug(f"DEBUG: CompressLU loop core {i} -> {i+1}.") # ADDED DEBUG
            core_i = self.M[i]
            core_i_plus_1 = self.M[i+1]

            logger.debug(f"DEBUG: Core {i} before compression: labels={core_i.labels()}, shape={core_i.shape()}") # ADDED DEBUG
            logger.debug(f"DEBUG: Core {i+1} before compression: labels={core_i_plus_1.labels()}, shape={core_i_plus_1.shape()}") # ADDED DEBUG

            orig_L_bond_i = core_i.bonds()[0].clone()
            orig_P_bond_i = core_i.bonds()[1].clone()
            orig_L_label_i = core_i.labels()[0]
            orig_P_label_i = core_i.labels()[1]
            middle_bond_label_i = core_i.labels()[2]

            A_mat_tensor = cube_as_matrix2_utt(core_i)
            logger.debug(f"DEBUG: Core {i} converted to matrix for decomposition. Shape: {A_mat_tensor.shape()}") # ADDED DEBUG
            
            L_out, R_out = decomp(A_mat_tensor)
            logger.debug(f"DEBUG: Decomposition results: L_out shape {L_out.shape()}, R_out shape {R_out.shape()}") # ADDED DEBUG

            if isinstance(L_out, np.ndarray):
                L_tensor = from_numpy(L_out).to(comp_device).astype(comp_dtype)
            else:
                L_tensor = L_out.to(comp_device).astype(comp_dtype)

            if isinstance(R_out, np.ndarray):
                R_tensor = from_numpy(R_out).to(comp_device).astype(comp_dtype)
            else:
                R_tensor = R_out.to(comp_device).astype(comp_dtype)
            logger.debug(f"DEBUG: L_tensor (cytnx): {L_tensor.shape()}, R_tensor (cytnx): {R_tensor.shape()}") # ADDED DEBUG

            dim_L_i_val = orig_L_bond_i.dim()
            dim_P_i_val = orig_P_bond_i.dim()
            new_bond_dim = L_tensor.shape()[1]
            logger.debug(f"DEBUG: New bond dimension determined: {new_bond_dim}.") # ADDED DEBUG

            if L_tensor.shape()[0] != dim_L_i_val * dim_P_i_val:
                logger.error(f"ERROR: Shape mismatch for L_tensor at site {i}. Expected first dim {dim_L_i_val * dim_P_i_val}, got {L_tensor.shape()[0]}") # ADDED ERROR
                raise ValueError(f"Shape mismatch for L_tensor at site {i}. Expected first dim {dim_L_i_val * dim_P_i_val}, got {L_tensor.shape()[0]}")

            L_reshaped = L_tensor.reshape(dim_L_i_val, dim_P_i_val, new_bond_dim)
            
            new_core_i = UniTensor(
                bonds=[orig_L_bond_i, orig_P_bond_i, cytnx.Bond(new_bond_dim, BD_OUT)],
                labels=[orig_L_label_i, orig_P_label_i, middle_bond_label_i],
                rowrank=1, 
            )
            new_core_i.put_block(L_reshaped)
            self.M[i] = new_core_i
            logger.debug(f"DEBUG: Core {i} updated. New labels: {self.M[i].labels()}, shape: {self.M[i].shape()}") # ADDED DEBUG
            TensorTrain._assert_core_validity(self.M[i], i, num_cores, check_data_presence=True) # Re-enabled assertion for debug

            if core_i_plus_1.labels()[0] != middle_bond_label_i:
                logger.error(f"ERROR: Label mismatch for absorption: M[{i}].right_label ('{middle_bond_label_i}') != M[{i+1}].left_label ('{core_i_plus_1.labels()[0]}')") # ADDED ERROR
                raise ValueError(f"Label mismatch for absorption: M[{i}].right_label ('{middle_bond_label_i}') "
                                 f"!= M[{i+1}].left_label ('{core_i_plus_1.labels()[0]}')")

            orig_P_bond_i1 = core_i_plus_1.bonds()[1].clone()
            orig_R_bond_i1 = core_i_plus_1.bonds()[2].clone()
            orig_P_label_i1 = core_i_plus_1.labels()[1]
            orig_R_label_i1 = core_i_plus_1.labels()[2]

            B_mat_tensor = cube_as_matrix1_utt(core_i_plus_1)
            logger.debug(f"DEBUG: Core {i+1} converted to matrix for absorption. Shape: {B_mat_tensor.shape()}") # ADDED DEBUG
                                                              
            C_mat_tensor = cytnx.linalg.Matmul(R_tensor, B_mat_tensor) 
            C_mat_tensor = C_mat_tensor.to(comp_device).astype(comp_dtype)
            logger.debug(f"DEBUG: Absorbed matrix C_mat_tensor shape: {C_mat_tensor.shape()}") # ADDED DEBUG

            dim_P_i1_val = orig_P_bond_i1.dim()
            dim_R_i1_val = orig_R_bond_i1.dim()
            
            if C_mat_tensor.shape()[0] != new_bond_dim or \
               C_mat_tensor.shape()[1] != dim_P_i1_val * dim_R_i1_val:
                logger.error(f"ERROR: Shape mismatch for C_mat_tensor at site {i+1}. Expected ({new_bond_dim}, {dim_P_i1_val * dim_R_i1_val}), got {C_mat_tensor.shape()}") # ADDED ERROR
                raise ValueError(f"Shape mismatch for C_mat_tensor at site {i+1}. Expected ({new_bond_dim}, {dim_P_i1_val * dim_R_i1_val}), got {C_mat_tensor.shape()}")

            C_reshaped = C_mat_tensor.reshape(new_bond_dim, dim_P_i1_val, dim_R_i1_val)

            new_core_i_plus_1 = UniTensor(
                bonds=[cytnx.Bond(new_bond_dim, BD_IN), orig_P_bond_i1, orig_R_bond_i1],
                labels=[middle_bond_label_i, orig_P_label_i1, orig_R_label_i1],
                rowrank=1,
            )
            new_core_i_plus_1.put_block(C_reshaped)
            self.M[i+1] = new_core_i_plus_1
            logger.debug(f"DEBUG: Core {i+1} updated. New labels: {self.M[i+1].labels()}, shape: {self.M[i+1].shape()}") # ADDED DEBUG
            TensorTrain._assert_core_validity(self.M[i+1], i + 1, num_cores, check_data_presence=True) # Re-enabled assertion for debug
        logger.info("INFO: compressLU finished.") # ADDED INFO

    def __add__(self, other: 'TensorTrain') -> 'TensorTrain':
        logger.info("INFO: Entering TensorTrain.__add__ operation.") # ADDED INFO
        if not self.M:
            logger.warning("WARNING: __add__: self is empty. Returning clone of other.") # ADDED WARNING
            cloned_other_cores = []
            if other.M:
                for core in other.M:
                    if isinstance(core, UniTensor):
                        cloned_other_cores.append(core.clone())
                    else:
                        cloned_other_cores.append(core)
            return TensorTrain(cloned_other_cores if cloned_other_cores else None)

        if not other.M:
            logger.warning("WARNING: __add__: other is empty. Returning clone of self.") # ADDED WARNING
            cloned_self_cores = []
            if self.M:
                for core in self.M:
                    if isinstance(core, UniTensor):
                        cloned_self_cores.append(core.clone())
                    else:
                        cloned_self_cores.append(core)
            return TensorTrain(cloned_self_cores if cloned_self_cores else None)

        if len(self.M) != len(other.M):
            logger.error(f"ERROR: Cannot add tensor trains of different lengths. Self: {len(self.M)}, Other: {len(other.M)}") # ADDED ERROR
            raise ValueError("Cannot add tensor trains of different lengths.")

        num_cores = len(self.M)

        comp_device = self.M[0].device() 
        final_requires_complex, final_requires_double_precision = False, False
        for k_check in range(num_cores): 
            for core_scan in [self.M[k_check], other.M[k_check]]:
                dt_cs = core_scan.dtype()
                if dt_cs == Type.ComplexFloat or dt_cs == Type.ComplexDouble:
                    final_requires_complex = True
                if dt_cs == Type.Double or dt_cs == Type.ComplexDouble:
                    final_requires_double_precision = True
                if final_requires_complex and final_requires_double_precision: break
            if final_requires_complex and final_requires_double_precision: break
        
        if final_requires_complex:
            comp_dtype = Type.ComplexDouble if final_requires_double_precision else Type.ComplexFloat
        else:
            comp_dtype = Type.Double if final_requires_double_precision else Type.Float
        
        type_id_to_name = {
            Type.Void: "Void", Type.ComplexDouble: "ComplexDouble", 
            Type.ComplexFloat: "ComplexFloat", Type.Double: "Double", 
            Type.Float: "Float", Type.Uint64: "Uint64", Type.Int64: "Int64",
            Type.Uint32: "Uint32", Type.Int32: "Int32", Type.Uint16: "Uint16", 
            Type.Int16: "Int16", Type.Bool: "Bool"
        }
        comp_dtype_name_str = type_id_to_name.get(comp_dtype, f"UnknownTypeID({comp_dtype})")
        logger.debug(f"DEBUG: __add__ Op: Target comp_dtype for new cores: {comp_dtype} ({comp_dtype_name_str}), Target comp_device: {comp_device}") # Original debug, now using logger

        new_UniTensor_cores: List[UniTensor] = []
        for k in range(num_cores):
            logger.debug(f"DEBUG: __add__ loop core {k}.") # ADDED DEBUG
            A_ut_processed = self.M[k].astype(comp_dtype).to(comp_device)
            A_ut_processed.contiguous_()

            B_ut_processed = other.M[k].astype(comp_dtype).to(comp_device)
            B_ut_processed.contiguous_()
            
            LA_l_orig, PA_l_orig, RA_l_orig = self.M[k].labels() 

            if A_ut_processed is None:
                logger.error(f"ERROR: __add__: A_ut_processed became None at core {k} for self.M.") # ADDED ERROR
                raise ValueError(f"__add__: A_ut_processed became None at core {k} for self.M.")
            if B_ut_processed is None:
                logger.error(f"ERROR: __add__: B_ut_processed became None at core {k} for other.M.") # ADDED ERROR
                raise ValueError(f"__add__: B_ut_processed became None at core {k} for other.M.")

            rA, dA, sA = A_ut_processed.shape()
            rB, dB, sB = B_ut_processed.shape()
            logger.debug(f"DEBUG: __add__ core {k}: A_shape={A_ut_processed.shape()}, B_shape={B_ut_processed.shape()}") # ADDED DEBUG

            if dA != dB:
                logger.error(f"ERROR: Physical dimension mismatch at core {k}: {dA} (self) vs {dB} (other)") # ADDED ERROR
                raise ValueError(f"Physical dimension mismatch at core {k}: {dA} (self) vs {dB} (other)")
            
            new_r_left: int
            new_d_phys: int = dA 
            new_s_right: int
            
            A_block = A_ut_processed.get_block_()
            B_block = B_ut_processed.get_block_()

            if k == 0:
                if rA != 1: logger.warning(f"WARNING: __add__ k=0: Left bond of A ({rA}) is not 1 after processing.") # Original warning, now using logger
                if rB != rA :
                     logger.warning(f"WARNING: __add__ k=0: rB ({rB}) != rA ({rA}) for B placement. This might lead to unexpected block summation if C++ logic expects exact dimension match for sub-cube assignment.") # Original warning, now using logger
                new_r_left = rA 
                new_s_right = sA + sB
                new_data_tensor = cytnx.zeros([new_r_left, new_d_phys, new_s_right], dtype=comp_dtype, device=comp_device)
                new_data_tensor[0:rA, :, 0:sA] = A_block
                new_data_tensor[0:min(rA,rB), :, sA : sA+sB] = B_block[0:min(rA,rB), :, :]

            elif k == num_cores - 1:
                if sB != 1: logger.warning(f"WARNING: __add__ k={k}: Right bond of B ({sB}) is not 1 after processing.") # Original warning, now using logger
                if sA != sB :
                    logger.warning(f"WARNING: __add__ k={k}: sA ({sA}) != sB ({sB}). C++ logic implies sA should match sB for A's placement. Adapting.") # Original warning, now using logger
                new_r_left = rA + rB
                new_s_right = sB 
                new_data_tensor = cytnx.zeros([new_r_left, new_d_phys, new_s_right], dtype=comp_dtype, device=comp_device)
                new_data_tensor[0:rA, :, 0:min(sA,sB)] = A_block[:, :, 0:min(sA,sB)]
                new_data_tensor[rA : rA+rB, :, 0:sB] = B_block
            
            else:
                new_r_left = rA + rB
                new_s_right = sA + sB
                new_data_tensor = cytnx.zeros([new_r_left, new_d_phys, new_s_right], dtype=comp_dtype, device=comp_device)
                new_data_tensor[0:rA, :, 0:sA] = A_block
                new_data_tensor[rA : rA+rB, :, sA : sA+sB] = B_block
            
            bd_L_new = cytnx.Bond(new_r_left, BD_IN)
            bd_P_new = A_ut_processed.bonds()[1].clone()
            bd_R_new = cytnx.Bond(new_s_right, BD_OUT)
            new_core_labels = [LA_l_orig, PA_l_orig, RA_l_orig]

            new_core_ut = UniTensor(bonds=[bd_L_new, bd_P_new, bd_R_new], 
                                    labels=new_core_labels, 
                                    rowrank=1)
            new_core_ut.put_block(new_data_tensor)
            new_UniTensor_cores.append(new_core_ut)
            logger.debug(f"DEBUG: __add__ core {k} created. Labels: {new_core_ut.labels()}, shape: {new_core_ut.shape()}") # ADDED DEBUG
        
        result_tt = TensorTrain(new_UniTensor_cores)
        logger.info("INFO: TensorTrain.__add__ finished.") # ADDED INFO
        return result_tt

    def sum(self, weights: List[List[float]]) -> Any:
        logger.info("INFO: Entering sum operation.") # ADDED INFO
        if not self.M:
            if not weights: 
                logger.warning("WARNING: Sum on empty TensorTrain with empty weights, returning 1.0.") # ADDED WARNING
                return 1.0
            else: 
                logger.error("ERROR: Cannot apply non-empty weights to an empty TensorTrain.") # ADDED ERROR
                raise ValueError("Cannot apply non-empty weights to an empty TensorTrain.")

        if len(weights) != len(self.M):
            logger.error(f"ERROR: Weights list length {len(weights)} does not match tensor train length {len(self.M)}.") # ADDED ERROR
            raise ValueError(
                f"Weights list length {len(weights)} does not match "
                f"tensor train length {len(self.M)}."
            )

        num_cores = len(self.M)
        comp_device = self.M[0].device()
        comp_dtype = self.M[0].dtype() 
        logger.debug(f"DEBUG: Sum: Using device {comp_device}, dtype {comp_dtype}.") # ADDED DEBUG
        
        first_core_L_label = self.M[0].labels()[0]
        if self.M[0].bonds()[0].dim() != 1:
            logger.error(f"ERROR: Sum: Left boundary bond '{first_core_L_label}' of core 0 must have dimension 1.") # ADDED ERROR
            raise ValueError(
                f"Sum: Left boundary bond '{first_core_L_label}' of core 0 "
                f"must have dimension 1."
            )

        prod_ut_bond = cytnx.Bond(1, BD_OUT)
        prod_ut = UniTensor(bonds=[prod_ut_bond], 
                            labels=[first_core_L_label], 
                            rowrank=1,
                            device=comp_device, 
                            dtype=comp_dtype)
        prod_ut.get_block_()[0] = 1.0

        for k, Mk_ut in enumerate(self.M):
            logger.debug(f"DEBUG: Sum loop core {k}. Current prod_ut labels: {prod_ut.labels()}, shape: {prod_ut.shape()}") # ADDED DEBUG
            # self._assert_core_validity(Mk_ut, k, num_cores) # Optional validation

            w_k_list = weights[k]
            phys_bond_on_Mk = Mk_ut.bonds()[1]
            phys_label_on_Mk = Mk_ut.labels()[1]
            phys_dim = phys_bond_on_Mk.dim()

            if len(w_k_list) != phys_dim:
                logger.error(f"ERROR: Weights list at index {k} (length {len(w_k_list)}) does not match physical dimension {phys_dim} of core {k} (label '{phys_label_on_Mk}').") # ADDED ERROR
                raise ValueError(
                    f"Weights list at index {k} (length {len(w_k_list)}) does not match "
                    f"physical dimension {phys_dim} of core {k} (label '{phys_label_on_Mk}')."
                )

            weights_bond = phys_bond_on_Mk.redirect() 
            
            if comp_dtype == Type.ComplexDouble or comp_dtype == Type.Double:
                numpy_equiv_dtype = np.float64
            elif comp_dtype == Type.ComplexFloat or comp_dtype == Type.Float:
                numpy_equiv_dtype = np.float32
            else:
                logger.warning(f"WARNING: Sum: Core dtype {comp_dtype} is not a standard float/complex type. Defaulting weights to np.float64.") # Original warning, now using logger
                numpy_equiv_dtype = np.float64
            
            weights_np_array = np.array(w_k_list, dtype=numpy_equiv_dtype)
            weights_data_tensor = from_numpy(weights_np_array).to(comp_device).astype(comp_dtype)
            
            weights_vec_ut = UniTensor(bonds=[weights_bond], 
                                       labels=[phys_label_on_Mk], 
                                       rowrank=0
                                      ) 
            weights_vec_ut.put_block(weights_data_tensor)
            logger.debug(f"DEBUG: Sum core {k}: Weights vector created with label {phys_label_on_Mk}, shape: {weights_vec_ut.shape()}") # ADDED DEBUG
            
            weighted_slice_ut = cytnx.Contract(Mk_ut, weights_vec_ut)
            logger.debug(f"DEBUG: Sum core {k}: weighted_slice_ut (Mk_ut * weights_vec_ut) labels: {weighted_slice_ut.labels()}, shape: {weighted_slice_ut.shape()}") # ADDED DEBUG
            
            prod_ut = cytnx.Contract(prod_ut, weighted_slice_ut)
            logger.debug(f"DEBUG: Sum core {k}: prod_ut updated. New labels: {prod_ut.labels()}, shape: {prod_ut.shape()}") # ADDED DEBUG

        if not (prod_ut.rank() == 1 and prod_ut.shape() == [1]):
            final_shape = prod_ut.shape() if prod_ut.rank() > 0 else "scalar (rank 0)"
            final_labels = prod_ut.labels()
            logger.error(f"ERROR: TensorTrain.sum did not collapse to a rank-1 UniTensor of shape [1]. Final rank: {prod_ut.rank()}, shape: {final_shape}, labels: {final_labels}.") # ADDED ERROR
            raise ValueError(
                f"TensorTrain.sum did not collapse to a rank-1 UniTensor of shape [1]. "
                f"Final rank: {prod_ut.rank()}, shape: {final_shape}, labels: {final_labels}."
            )
        
        if self.M[-1].bonds()[2].dim() != 1:
            logger.error(f"ERROR: Sum: Right boundary bond '{self.M[-1].labels()[2]}' of core {num_cores-1} must have dimension 1 for sum to result in a scalar.") # ADDED ERROR
            raise ValueError(
                f"Sum: Right boundary bond '{self.M[-1].labels()[2]}' of core {num_cores-1} "
                f"must have dimension 1 for sum to result in a scalar."
            )

        result = prod_ut.item()
        logger.info(f"INFO: Sum finished. Result: {result}") # ADDED INFO
        return result
    
    def sum1(self) -> Any:
        logger.info("INFO: Entering sum1 operation.") # ADDED INFO
        if not self.M:
            logger.warning("WARNING: Sum1 on empty TensorTrain, returning 1.0.") # ADDED WARNING
            return 1.0
        
        weights_for_sum1 = [[1.0] * core.bonds()[1].dim() for core in self.M]
        result = self.sum(weights_for_sum1)
        logger.info(f"INFO: Sum1 finished. Result: {result}") # ADDED INFO
        return result
    
    def trueError(self, other: 'TensorTrain', max_eval: int = int(1e6)) -> float:
        logger.info(f"INFO: Entering trueError calculation with max_eval={max_eval}.") # ADDED INFO
        if not hasattr(self, 'M') or not self.M:
            if hasattr(other, 'M') and not other.M and (not hasattr(other, 'M') or len(self.M) == len(other.M)):
                 logger.warning("WARNING: trueError on two empty TensorTrains, returning 0.0.") # ADDED WARNING
                 return 0.0

        if not hasattr(other, 'M'):
            logger.error("ERROR: The 'other' object provided to trueError is not a valid TensorTrain (missing 'M').") # ADDED ERROR
            raise TypeError("The 'other' object provided to trueError is not a valid TensorTrain (missing 'M').")

        if len(self.M) != len(other.M):
            logger.error(f"ERROR: Tensor trains for trueError must have the same length. Self: {len(self.M)}, Other: {len(other.M)}") # ADDED ERROR
            raise ValueError("Tensor trains for trueError must have the same length.")
        
        if not self.M:
            logger.warning("WARNING: trueError on two empty TensorTrains, returning 0.0 (after length check).") # ADDED WARNING
            return 0.0

        try:
            dims = []
            for k_core, Mk_core in enumerate(self.M):
                if not isinstance(Mk_core, UniTensor) or Mk_core.rank() < 2:
                    logger.error(f"ERROR: Core {k_core} in 'self' is not a valid UniTensor or has insufficient rank for trueError.") # ADDED ERROR
                    raise ValueError(f"Core {k_core} in 'self' is not a valid UniTensor or has insufficient rank for trueError.")
                dims.append(Mk_core.bonds()[1].dim())
        except IndexError:
            logger.error("ERROR: Could not retrieve physical dimensions from cores in 'self'. Ensure cores are valid rank-3 UniTensors.") # ADDED ERROR
            raise ValueError("Could not retrieve physical dimensions from cores in 'self'. Ensure cores are valid rank-3 UniTensors.")
        
        try:
            for k_core, Mk_core_other in enumerate(other.M):
                 if not isinstance(Mk_core_other, UniTensor) or Mk_core_other.rank() < 2:
                     logger.error(f"ERROR: Core {k_core} in 'other' is not a valid UniTensor or has insufficient rank for trueError.") # ADDED ERROR
                     raise ValueError(f"Core {k_core} in 'other' is not a valid UniTensor or has insufficient rank for trueError.")
                 if Mk_core_other.bonds()[1].dim() != dims[k_core]:
                     logger.error(f"ERROR: Physical dimension mismatch at core {k_core} between self and other.") # ADDED ERROR
                     raise ValueError(f"Physical dimension mismatch at core {k_core} between self and other.")
        except IndexError:
            logger.error("ERROR: Could not retrieve/validate physical dimensions from cores in 'other'.") # ADDED ERROR
            raise ValueError("Could not retrieve/validate physical dimensions from cores in 'other'.")


        total_configs = np.prod(dims, dtype=np.int64) 

        if total_configs == 0 and not any(d == 0 for d in dims): 
             logger.warning(
                 f"WARNING: Product of dimensions ({np.prod(dims)}) resulted in {total_configs}, "
                 f"which might indicate an overflow if no physical dimension is zero. "
                 f"Number of configurations could be very large."
            )
        
        if total_configs > max_eval:
            actual_total_str = str(np.prod(dims)) if total_configs == 0 and not any(d==0 for d in dims) else str(total_configs)
            logger.error(f"ERROR: Too many index combinations ({actual_total_str} if no overflow, or {total_configs} if overflowed to 0) for trueError, exceeds max_eval={max_eval}") # ADDED ERROR
            raise ValueError(
                f"Too many index combinations ({actual_total_str} if no overflow, "
                f"or {total_configs} if overflowed to 0) for trueError, "
                f"exceeds max_eval={max_eval}"
            )
        
        if total_configs == 0 and any(d == 0 for d in dims): 
            logger.info("INFO: Total configurations is 0 due to zero physical dimension. Returning 0.0 error.") # ADDED INFO
            return 0.0

        max_err = 0.0
        for idx_tuple in product(*[range(d) for d in dims]):
            idx = list(idx_tuple) 
            
            v1 = self.eval(idx) 
            v2 = other.eval(idx)
            
            current_err_val: float
            if isinstance(v1, complex) or isinstance(v2, complex):
                current_err_val = abs(np.complex128(v1) - np.complex128(v2))
            else: 
                current_err_val = abs(float(v1) - float(v2))

            if current_err_val > max_err:
                max_err = current_err_val
        logger.info(f"INFO: trueError finished. Max error: {max_err}") # ADDED INFO
        return float(max_err)

    def save(self, file_name: str):
        logger.info(f"INFO: Entering save operation for file: {file_name}") # ADDED INFO
        if not hasattr(self, 'M'):
            logger.warning("WARNING: Attempting to save a TensorTrain with no 'M' attribute or M is None. Saving empty list.") # Original warning, now using logger
            core_data_list = []
        elif not self.M:
            core_data_list = []
            logger.info("INFO: TensorTrain is empty. Saving empty list.") # ADDED INFO
        else:
            core_data_list = []
            for k, Mk_ut in enumerate(self.M):
                logger.debug(f"DEBUG: Saving core {k}. Labels: {Mk_ut.labels()}, Shape: {Mk_ut.shape()}, Dtype: {Mk_ut.dtype_str()}") # ADDED DEBUG
                if not isinstance(Mk_ut, UniTensor):
                    logger.error(f"ERROR: Core at index {k} must be a UniTensor for saving, got {type(Mk_ut)}.") # ADDED ERROR
                    raise TypeError(f"Core at index {k} must be a UniTensor for saving, got {type(Mk_ut)}.")
                
                Mk_ut_contiguous = Mk_ut.contiguous()

                bonds_params = []
                for b_idx, b in enumerate(Mk_ut_contiguous.bonds()):
                    bond_type_enum_val = b.type().value 
                    qnums_list_of_list = b.qnums()
                    sym_type_ints = [sym.stype() for sym in b.syms()]

                    bond_param = {
                        "dim": b.dim(),
                        "type_val": bond_type_enum_val, 
                        "qnums": qnums_list_of_list, 
                        "sym_type_ints": sym_type_ints 
                    }
                    bonds_params.append(bond_param)
                    logger.debug(f"DEBUG: Core {k} bond {b_idx} saved. Dim: {b.dim()}, TypeVal: {bond_type_enum_val}, Qnums: {qnums_list_of_list}, SymTypes: {sym_type_ints}") # ADDED DEBUG

                core_dict = {
                    "labels": Mk_ut_contiguous.labels(),
                    "bonds_params": bonds_params,
                    "rowrank": Mk_ut_contiguous.rowrank(),
                    "dtype_int": Mk_ut_contiguous.dtype(),
                    "device_int": Mk_ut_contiguous.device(),
                    "block_numpy": Mk_ut_contiguous.get_block_().numpy()
                }
                core_data_list.append(core_dict)
        
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(core_data_list, f)
            logger.info(f"INFO: TensorTrain saved successfully to {file_name} with {len(core_data_list)} cores.") # Original info, now using logger
        except Exception as e:
            logger.error(f"ERROR: Error saving TensorTrain to {file_name}: {e}") # Original error, now using logger
            raise

    @staticmethod
    def load(file_name: str) -> 'TensorTrain':
        logger.info(f"INFO: Entering load operation for file: {file_name}") # ADDED INFO
        try:
            with open(file_name, 'rb') as f:
                core_data_list = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"ERROR: Error loading TensorTrain: File {file_name} not found.") # Original error, now using logger
            raise
        except Exception as e:
            logger.error(f"ERROR: Error loading TensorTrain data from {file_name}: {e}") # Original error, now using logger
            raise

        if not isinstance(core_data_list, list):
            logger.error(f"ERROR: Data loaded from {file_name} is not a list as expected for TensorTrain.") # ADDED ERROR
            raise TypeError(f"Data loaded from {file_name} is not a list as expected for TensorTrain.")

        if not core_data_list:
            logger.info(f"INFO: Loaded an empty TensorTrain from {file_name}.") # Original info, now using logger
            return TensorTrain([]) 

        loaded_cores_M: List[UniTensor] = []
        for i, core_dict in enumerate(core_data_list):
            logger.debug(f"DEBUG: Loading core {i}.") # ADDED DEBUG
            if not isinstance(core_dict, dict):
                logger.error(f"ERROR: Core data at index {i} in {file_name} is not a dictionary.") # ADDED ERROR
                raise TypeError(f"Core data at index {i} in {file_name} is not a dictionary.")

            try:
                labels = core_dict["labels"]
                bonds_params = core_dict["bonds_params"]
                rowrank = core_dict["rowrank"]
                dtype_int = core_dict["dtype_int"]
                device_int = core_dict["device_int"]
                block_numpy = core_dict["block_numpy"]
                logger.debug(f"DEBUG: Core {i} metadata loaded. Labels: {labels}, Rowrank: {rowrank}, Dtype_int: {dtype_int}, Device_int: {device_int}") # ADDED DEBUG

                reconstructed_bonds = []
                for bp_idx, bp_param in enumerate(bonds_params):
                    dim = bp_param["dim"]
                    bond_type_val = bp_param["type_val"] 
                    qnums = bp_param.get("qnums", [])
                    sym_type_ints = bp_param.get("sym_type_ints", [])
                    logger.debug(f"DEBUG: Core {i} bond {bp_idx} params: Dim={dim}, TypeVal={bond_type_val}, Qnums={qnums}, SymTypeInts={sym_type_ints}") # ADDED DEBUG

                    bond_type_enum_member: cytnx.bondType
                    if bond_type_val == cytnx.bondType.BD_KET.value:
                        bond_type_enum_member = cytnx.bondType.BD_IN
                    elif bond_type_val == cytnx.bondType.BD_BRA.value:
                        bond_type_enum_member = cytnx.bondType.BD_OUT
                    else:
                        logger.warning(f"WARNING: Unknown bond_type_val {bond_type_val} for core {i}, bond {bp_idx}. Defaulting to BD_REG.") # Original warning, now using logger
                        bond_type_enum_member = cytnx.bondType.BD_REG


                    if not qnums and not sym_type_ints:
                        reconstructed_bonds.append(
                            cytnx.Bond(dim, bond_type_enum_member)
                        )
                    else:
                        symmetries = []
                        for stype_int in sym_type_ints:
                            symmetries.append(cytnx.Symmetry.from_stype(stype_int))
                        
                        if not all(isinstance(q_sec, list) for q_sec in qnums):
                             logger.error(f"ERROR: Symmetric bond loading for core {i}, bond {bp_idx}: Saved qnums format requires 'degs' which are not explicitly saved/loaded in this version, or qnums are not in List[List[int]] format. Full symmetric bond reconstruction needs review.") # ADDED ERROR
                             raise NotImplementedError(
                                f"Symmetric bond loading for core {i}, bond {bp_idx}: "
                                f"Saved qnums format requires 'degs' which are not explicitly saved/loaded in this version, "
                                f"or qnums are not in List[List[int]] format. Full symmetric bond reconstruction needs review."
                            )
                        
                        degs_assumed = [1] * len(qnums)
                        if not qnums:
                            reconstructed_bonds.append(cytnx.Bond(dim, bond_type_enum_member))
                            logger.warning(f"WARNING: Core {i}, bond {bp_idx}: Qnums empty but syms present. Falling back to non-symmetric bond.") # ADDED WARNING
                        else:
                            try:
                                reconstructed_bonds.append(
                                    cytnx.Bond(bond_type_enum_member, qnums, degs_assumed, symmetries)
                                )
                                logger.debug(f"DEBUG: Core {i}, bond {bp_idx}: Symmetric bond reconstructed with qnums: {qnums}, assumed degs: {degs_assumed}.") # ADDED DEBUG
                            except Exception as e_symm:
                                logger.error(f"ERROR: Failed to reconstruct symmetric bond for core {i}, bond {bp_idx} with assumed degs: {e_symm}. Falling back to non-symmetric or re-raise.") # Original error, now using logger
                                reconstructed_bonds.append(cytnx.Bond(dim, bond_type_enum_member))
                                # If strictness is required, remove the fallback and re-raise

                ut = UniTensor(bonds=reconstructed_bonds, 
                               labels=labels, 
                               rowrank=rowrank,
                               dtype=dtype_int,
                               device=device_int)
                
                data_tensor = from_numpy(block_numpy).astype(dtype_int).to(device_int)
                ut.put_block(data_tensor)
                
                loaded_cores_M.append(ut)
                logger.debug(f"DEBUG: Core {i} successfully loaded. Labels: {ut.labels()}, Shape: {ut.shape()}, Dtype: {ut.dtype_str()}") # ADDED DEBUG
            except KeyError as ke:
                logger.error(f"ERROR: Missing key {ke} in loaded core data at index {i} from {file_name}.") # Original error, now using logger
                raise
            except Exception as e:
                logger.error(f"ERROR: Error reconstructing UniTensor core at index {i} from {file_name}: {e}") # Original error, now using logger
                import traceback
                traceback.print_exc()
                raise
        
        result_tt = TensorTrain(loaded_cores_M)
        logger.info(f"INFO: Loaded TensorTrain from {file_name} with {len(loaded_cores_M)} cores.") # ADDED INFO
        return result_tt
    

if __name__ == "__main__":
    import numpy as _np
    from cytnx import from_numpy, cytnx

    # Configure logging for better feedback during tests
    # Setting level to DEBUG will show all the new debug messages.
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s') # Changed to DEBUG
    logger = logging.getLogger(__name__) # Get module-specific logger, or root logger if desired
    # logger.setLevel(logging.DEBUG) # This is already set by basicConfig above.

    _np.random.seed(128)
    
    phys_dims = [3, 4, 5]
    L = len(phys_dims)

    print(f"\nDEBUG: Preparing to create {L} UniTensor cores with phys_dims {phys_dims}...")
    ut_cores = []
    current_device = cytnx.Device.cpu
    current_dtype = Type.Double 

    for k in range(L):
        d_k = phys_dims[k]
        r_left = 1
        r_right = 1
        
        if not (isinstance(r_left, int) and r_left > 0 and 
                isinstance(d_k, int) and d_k > 0 and
                isinstance(r_right, int) and r_right > 0):
            raise ValueError(f"Invalid dimensions for core {k}: r_left={r_left}, d_k={d_k}, r_right={r_right}")

        bd_left = cytnx.Bond(r_left, BD_IN)
        bd_phys = cytnx.Bond(d_k, BD_OUT)
        bd_right = cytnx.Bond(r_right, BD_OUT)
        
        label_left = "L_bound" if k == 0 else f"link{k-1}"
        label_phys = f"p{k}"
        label_right = "R_bound" if k == L - 1 else f"link{k}"
        expected_labels = [label_left, label_phys, label_right]
        
        ut_core = UniTensor(bonds=[bd_left, bd_phys, bd_right], 
                            labels=expected_labels, 
                            rowrank=1)
        
        arr_np = _np.random.rand(r_left, d_k, r_right) 
        tensor_data = from_numpy(arr_np).to(current_device).astype(current_dtype)
        ut_core.put_block(tensor_data)
        ut_cores.append(ut_core)
    
    print(f"DEBUG: Finished creating list of {len(ut_cores)} UniTensors.")
    print("DEBUG: Initializing original_tt...")
    original_tt = TensorTrain(ut_cores.copy()) 
    print("DEBUG: original_tt initialized.")
    print("DEBUG: Initializing to_compress_tt...")
    to_compress_tt = TensorTrain(ut_cores.copy()) 
    print("DEBUG: to_compress_tt initialized.")

    if original_tt.M:
        print(f"\n--- Info for the first core of original_tt (phys_dims: {phys_dims}) ---")
        first_core = original_tt.M[0]
        print(f"Type of core: {type(first_core)}")
        print(f"Labels: {first_core.labels()}")
        print(f"Shape: {first_core.shape()}")
        bond_type_strs = [str(b.type()) for b in first_core.bonds()]
        print(f"Bond types (as string): {bond_type_strs}")
        print(f"Rowrank: {first_core.rowrank()}")
        print(f"Device: {first_core.device_str()}")
        print(f"Dtype: {first_core.dtype_str()}")
        
        print("DEBUG: Running _assert_core_validity on all cores of original_tt...")
        for k_assert in range(L):
            TensorTrain._assert_core_validity(original_tt.M[k_assert], k_assert, L, check_data_presence=True)
        print("DEBUG: _assert_core_validity PASSED for all cores in original_tt.")

    print("\nSuccessfully created TensorTrain with UniTensor cores.")
    print(f"Number of cores in original_tt: {len(original_tt.M)}")
    if original_tt.M:
        print(f"First core labels of original_tt: {original_tt.M[0].labels()}")

    print("\n--- Testing Eval ---")
    idx_to_eval = [0] * L
    try:
        val_eval = original_tt.eval(idx_to_eval)
        print(f"original_tt.eval({idx_to_eval}) = {val_eval}")
        
        manual_prod = 1.0
        for k_manual in range(L):
            core_block_np = original_tt.M[k_manual].get_block_().numpy()
            manual_prod *= core_block_np[0, idx_to_eval[k_manual], 0]
        print(f"Manual product for eval({idx_to_eval}) = {manual_prod}")
        assert _np.isclose(val_eval, manual_prod), f"Eval mismatch: {val_eval} vs {manual_prod}"
        print("Eval test PASSED.")
    except Exception as e:
        print(f"ERROR in Eval test: {e}")
        logger.exception("Eval test exception details:")


    print("\n--- Testing Overlap & Norm2 ---")
    try:
        tt_b_for_overlap = TensorTrain(ut_cores.copy()) 

        overlap_ab = original_tt.overlap(tt_b_for_overlap)
        norm2_a = original_tt.norm2()
        print(f"original_tt.overlap(tt_b_for_overlap) (self-overlap) = {overlap_ab}")
        print(f"original_tt.norm2() = {norm2_a}")
        assert _np.isclose(overlap_ab, norm2_a), f"Overlap with self ({overlap_ab}) should be equal to norm2 ({norm2_a})"
        print("Overlap & Norm2 test PASSED.")
    except Exception as e:
        print(f"ERROR in Overlap/Norm2 test: {e}")
        logger.exception("Overlap/Norm2 test exception details:")


    print("\n--- Testing Add ---")
    try:
        tt_c = original_tt + tt_b_for_overlap 
        print(f"tt_c (original_tt + tt_b_for_overlap) has {len(tt_c.M)} cores.")
        if tt_c.M and L > 0:
            first_core_c = tt_c.M[0]
            print(f"First core of tt_c: Labels={first_core_c.labels()}, Shape={first_core_c.shape()}")
            
            expected_L_dim_first_core = original_tt.M[0].bonds()[0].dim()
            expected_P_dim_first_core = original_tt.M[0].bonds()[1].dim()
            expected_R_dim_first_core = original_tt.M[0].bonds()[2].dim() + tt_b_for_overlap.M[0].bonds()[2].dim()

            assert first_core_c.bonds()[0].dim() == expected_L_dim_first_core, \
                f"Sum TT First Core: Left bond dim mismatch. Expected {expected_L_dim_first_core}, Got {first_core_c.bonds()[0].dim()}"
            assert first_core_c.bonds()[1].dim() == expected_P_dim_first_core, \
                f"Sum TT First Core: Phys bond dim mismatch. Expected {expected_P_dim_first_core}, Got {first_core_c.bonds()[1].dim()}"
            assert first_core_c.bonds()[2].dim() == expected_R_dim_first_core, \
                f"Sum TT First Core: Right bond dim mismatch. Expected {expected_R_dim_first_core}, Got {first_core_c.bonds()[2].dim()}"

            if L > 1:
                last_core_c = tt_c.M[L-1]
                print(f"Last core of tt_c: Labels={last_core_c.labels()}, Shape={last_core_c.shape()}")

                expected_L_dim_last_core = original_tt.M[L-1].bonds()[0].dim() + tt_b_for_overlap.M[L-1].bonds()[0].dim()
                expected_P_dim_last_core = original_tt.M[L-1].bonds()[1].dim()
                expected_R_dim_last_core = tt_b_for_overlap.M[L-1].bonds()[2].dim()

                assert last_core_c.bonds()[0].dim() == expected_L_dim_last_core, \
                    f"Sum TT Last Core: Left bond dim mismatch. Expected {expected_L_dim_last_core}, Got {last_core_c.bonds()[0].dim()}"
                assert last_core_c.bonds()[1].dim() == expected_P_dim_last_core, \
                    f"Sum TT Last Core: Phys bond dim mismatch. Expected {expected_P_dim_last_core}, Got {last_core_c.bonds()[1].dim()}"
                assert last_core_c.bonds()[2].dim() == expected_R_dim_last_core, \
                    f"Sum TT Last Core: Right bond dim mismatch. Expected {expected_R_dim_last_core}, Got {last_core_c.bonds()[2].dim()}"

            if L > 2:
                middle_core_c = tt_c.M[1]
                print(f"Middle core (k=1) of tt_c: Labels={middle_core_c.labels()}, Shape={middle_core_c.shape()}")
                expected_L_dim_middle = original_tt.M[1].bonds()[0].dim() + tt_b_for_overlap.M[1].bonds()[0].dim()
                expected_P_dim_middle = original_tt.M[1].bonds()[1].dim()
                expected_R_dim_middle = original_tt.M[1].bonds()[2].dim() + tt_b_for_overlap.M[1].bonds()[2].dim()
                assert middle_core_c.bonds()[0].dim() == expected_L_dim_middle
                assert middle_core_c.bonds()[1].dim() == expected_P_dim_middle
                assert middle_core_c.bonds()[2].dim() == expected_R_dim_middle


            print("Addition core shapes structurally verified against C++-like logic.")
            
            val_c_eval = tt_c.eval(idx_to_eval)
            print(f"tt_c.eval({idx_to_eval}) = {val_c_eval}")
            assert _np.isclose(val_c_eval, 2 * val_eval), f"tt_c.eval ({val_c_eval}) should be 2 * original_tt.eval ({val_eval})"
            print("__add__ test PASSED.")
        else:
            print("__add__ test: tt_c has no cores (unexpected if inputs were not empty).")

    except Exception as e:
        print(f"ERROR in __add__ test: {e}")
        logger.exception("__add__ test exception details:")


    print("\n--- Testing Sum & Sum1 ---")
    try:
        s1_a = original_tt.sum1()
        print(f"original_tt.sum1() = {s1_a}")
        manual_sum1 = 1.0
        for k_manual_s1 in range(L):
            core_block_s1_np = original_tt.M[k_manual_s1].get_block_().numpy()
            manual_sum1 *= _np.sum(core_block_s1_np[0, :, 0])
        print(f"Manual sum1 = {manual_sum1}")
        assert _np.isclose(s1_a, manual_sum1), f"Sum1 mismatch: {s1_a} vs {manual_sum1}"

        custom_weights = [[0.5] * d for d in phys_dims]
        s_custom_a = original_tt.sum(custom_weights)
        print(f"original_tt.sum(custom_weights) = {s_custom_a}")
        manual_sum_custom = 1.0
        for k_manual_sc in range(L):
            core_block_sc_np = original_tt.M[k_manual_sc].get_block_().numpy()
            w_k = custom_weights[k_manual_sc]
            weighted_phys_sum = 0.0
            for phys_idx in range(len(w_k)):
                weighted_phys_sum += w_k[phys_idx] * core_block_sc_np[0, phys_idx, 0]
            manual_sum_custom *= weighted_phys_sum
        print(f"Manual sum_custom = {manual_sum_custom}")
        assert _np.isclose(s_custom_a, manual_sum_custom), f"Custom sum mismatch: {s_custom_a} vs {manual_sum_custom}"
        print("Sum & Sum1 tests PASSED.")
    except Exception as e:
        print(f"ERROR in Sum/Sum1 test: {e}")
        logger.exception("Sum/Sum1 test exception details:")

    print("\n--- Testing Save & Load ---")
    fn_test = "test_tt_save_load.cytn_tt_obj"
    try:
        original_tt.save(fn_test)
        print(f"original_tt saved to {fn_test}")
        tt_loaded = TensorTrain.load(fn_test)
        print(f"TensorTrain loaded from {fn_test}, has {len(tt_loaded.M)} cores.")
        assert len(original_tt.M) == len(tt_loaded.M), "Loaded TT core count mismatch"
        
        if original_tt.M and tt_loaded.M:
            max_eval_for_test = 1
            for dim_p in phys_dims: max_eval_for_test *= dim_p
            if max_eval_for_test > 200: max_eval_for_test = 200

            error_loaded = original_tt.trueError(tt_loaded, max_eval=max_eval_for_test) 
            print(f"trueError between original_tt and tt_loaded = {error_loaded}")
            assert _np.isclose(error_loaded, 0.0), f"Loaded TT differs from original, error={error_loaded}"
        print("Save and Load test PASSED.")
    except Exception as e:
        print(f"ERROR in Save/Load test: {e}")
        logger.exception("Save/Load test exception details:")


    mat_decomp_available = False
    try:
        from mat_decomp import MatprrLUFixedTol
        mat_decomp_available = True
        print("\nDEBUG: mat_decomp.MatprrLUFixedTol imported successfully for compressLU test.")
    except ImportError:
        print("\nWARNING: mat_decomp.MatprrLUFixedTol not found. compressLU tests will be SKIPPED.")

    if mat_decomp_available:
        print("\n--- Testing CompressLU ---")
        tt_compress_ref = TensorTrain(ut_cores.copy())
        maxBondDim = 50
        
        print(f"Shape of M[0] before compress: {to_compress_tt.M[0].shape()}, Labels: {to_compress_tt.M[0].labels()}")
        if L > 1: print(f"Shape of M[1] before compress: {to_compress_tt.M[1].shape()}, Labels: {to_compress_tt.M[1].labels()}")

        try:
            to_compress_tt.compressLU(reltol=1e-7, maxBondDim=10)
            print("compressLU finished.")
            print(f"Shape of M[0] after compress: {to_compress_tt.M[0].shape()}, Labels: {to_compress_tt.M[0].labels()}")
            if L > 1: print(f"Shape of M[1] after compress: {to_compress_tt.M[1].shape()}, Labels: {to_compress_tt.M[1].labels()}")

            if L > 1:
                for k_comp_check in range(L - 1):
                    dim_link_k_right_M_k = to_compress_tt.M[k_comp_check].bonds()[2].dim()
                    dim_link_k_left_M_k1 = to_compress_tt.M[k_comp_check+1].bonds()[0].dim()
                    print(f"  Compressed bond 'link{k_comp_check}': M[{k_comp_check}]_right_dim={dim_link_k_right_M_k}, M[{k_comp_check+1}]_left_dim={dim_link_k_left_M_k1}")
                    assert dim_link_k_right_M_k == dim_link_k_left_M_k1, f"Compressed bond 'link{k_comp_check}' dimension mismatch"
                    assert dim_link_k_right_M_k <= max(1,maxBondDim), f"Compressed bond 'link{k_comp_check}' dim {dim_link_k_right_M_k} > maxBondDim {maxBondDim}"
                print("Compressed bond dimensions are consistent and within maxBondDim.")
            
            max_eval_for_compress_test = 1
            for dim_p_c in phys_dims: max_eval_for_compress_test *= dim_p_c
            if max_eval_for_compress_test > 200: max_eval_for_compress_test = 200

            error_compress = tt_compress_ref.trueError(to_compress_tt, max_eval=max_eval_for_compress_test)
            print(f"Maximum reconstruction error after compressLU: {error_compress:.3e}")

            print("CompressLU test section PASSED (execution completed, functional check heuristic).")
        except Exception as e:
            print(f"ERROR in CompressLU test: {e}")
            logger.exception("CompressLU test exception details:")
    
    print("\n--- All Implemented Tests Finished ---")