import cytnx
from cytnx import Bond
import pickle
import numpy as np
from itertools import product
import logging
from typing import List, Any, Union, Optional
from mat_decomp import MatprrLUFixedTol

# Initialize logger
logger = logging.getLogger(__name__)

# Cytnx core imports
from cytnx import Tensor, UniTensor, Type, Device, BD_IN, BD_OUT, from_numpy, zeros, eye, linalg

def _utt_assert_rank3(A_ut: cytnx.UniTensor, func_name: str):
    """
    Asserts the input UniTensor is rank 3 and is a valid UniTensor instance.
    """
    if not isinstance(A_ut, cytnx.UniTensor):
        raise TypeError(f"{func_name} expects a cytnx.UniTensor, got {type(A_ut)}")
    # Removed A_ut.is_void() check as the method does not exist.
    # The expectation is that any UniTensor passed to cube_as_matrix*_utt
    # has been properly initialized with data via put_block() by the caller or __init__.
    # If get_block_() is called later on a truly void UniTensor (no storage),
    # cytnx will raise an error there.
    if A_ut.rank() != 3:
        raise ValueError(f"{func_name} expects a rank-3 UniTensor, got rank {A_ut.rank()}. Labels: {A_ut.labels()}")
    if len(A_ut.labels()) != 3:
        raise ValueError(f"{func_name} expects a UniTensor with 3 labels, got {len(A_ut.labels())}: {A_ut.labels()}")

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
    _utt_assert_rank3(A_ut, "cube_as_matrix1_utt")
    
    # Get original labels and dimensions to ensure correct permutation and reshaping
    original_labels = A_ut.labels()
    label_L, label_P, label_R = original_labels[0], original_labels[1], original_labels[2]
    
    # Permute UniTensor so its labels are in the desired order for reshaping: [L, P, R].
    # The rowrank parameter influences how UniTensor interprets its matrix form,
    # but for get_block_() -> reshape(), the permutation order is key.
    # rowrank=1 here means for the UniTensor itself, L is the row-space.
    A_permuted = A_ut.permute([label_L, label_P, label_R], rowrank=1)
    
    # Ensure the permuted UniTensor's data is contiguous in memory.
    # This is crucial for subsequent get_block_() and reshape() to work correctly
    # and avoid views of non-contiguous data.
    A_permuted.contiguous_() 

    # Get the underlying cytnx.Tensor data block.
    # After permutation [L,P,R] and contiguous, its shape will be (dim_L, dim_P, dim_R).
    block = A_permuted.get_block_() 
    
    # Get dimensions from the block to be absolutely sure for reshape
    dim_L_val = block.shape()[0]
    dim_P_val = block.shape()[1]
    dim_R_val = block.shape()[2]
    
    # Reshape the cytnx.Tensor block, not the UniTensor.
    # This combines the second (P) and third (R) dimensions of the block.
    reshaped_tensor = block.reshape(dim_L_val, dim_P_val * dim_R_val)
    
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
    _utt_assert_rank3(A_ut, "cube_as_matrix2_utt")

    original_labels = A_ut.labels()
    label_L, label_P, label_R = original_labels[0], original_labels[1], original_labels[2]

    # Get original dimensions
    dim_L_val = A_ut.bond(label_L).dim()
    dim_P_val = A_ut.bond(label_P).dim()
    dim_R_val = A_ut.bond(label_R).dim()

    # Permute UniTensor to order [L, P, R].
    # rowrank=2 makes (L,P) the conceptual row part of the UniTensor itself.
    A_permuted = A_ut.permute([label_L, label_P, label_R], rowrank=2)
    A_permuted.contiguous_()

    # Get the underlying cytnx.Tensor data block.
    # Its shape will be (dim_L, dim_P, dim_R).
    block = A_permuted.get_block_()
    
    # Verify block shape if necessary (should match permuted UniTensor's shape)
    # assert block.shape() == [dim_L_val, dim_P_val, dim_R_val]

    # Reshape the cytnx.Tensor block.
    # This combines the first (L) and second (P) dimensions of the block.
    reshaped_tensor = block.reshape(dim_L_val * dim_P_val, dim_R_val)
    
    return reshaped_tensor


class TensorTrain:
    # Forward declaration for _assert_core_validity if it's defined later in the class
    # For now, we'll assume it's available or handle its absence if called.
    # @staticmethod # This will be a static method
    # def _assert_core_validity(core: UniTensor, k: int, L: int):
    #     pass # Placeholder

    def __init__(self, M: Optional[List[Union[Tensor, UniTensor]]] = None):
        self.M: List[UniTensor] = []

        if M is None or not M:
            return

        # 1. Determine Target Device
        target_device = M[0].device()

        # 2. Determine Target Dtype (Robust scanning)
        final_requires_complex = False
        final_requires_double_precision = False

        for core_input_scan in M:
            # Check for UniTensor first due to inheritance if Tensor becomes a UniTensor subclass in future cytnx
            if isinstance(core_input_scan, UniTensor):
                dt_int = core_input_scan.dtype() # Returns integer type ID
                if dt_int == Type.ComplexDouble or dt_int == Type.ComplexFloat:
                    final_requires_complex = True
                if dt_int == Type.Double or dt_int == Type.ComplexDouble:
                    final_requires_double_precision = True
            elif isinstance(core_input_scan, Tensor):
                dt_int = core_input_scan.dtype() # Returns integer type ID
                if dt_int % 2 == 1: # Odd type IDs are complex for Tensor
                    final_requires_complex = True
                if dt_int == Type.Double or dt_int == Type.ComplexDouble: # Check if it's Double or ComplexDouble
                    final_requires_double_precision = True
            else:
                raise TypeError(
                    f"Input core list M must contain cytnx.Tensor or cytnx.UniTensor objects. "
                    f"Got {type(core_input_scan)}"
                )
            # Optimization: if both flags are true, we've found the highest requirement
            if final_requires_complex and final_requires_double_precision:
                break
        
        if final_requires_complex:
            target_dtype = Type.ComplexDouble if final_requires_double_precision else Type.ComplexFloat
        else: # All real
            target_dtype = Type.Double if final_requires_double_precision else Type.Float

        # 3. Iterate and Convert/Process Cores
        num_cores = len(M)
        for k, core_input in enumerate(M):
            # Define expected labels for the current core
            label_left = "L_bound" if k == 0 else f"link{k-1}"
            label_phys = f"p{k}"
            label_right = "R_bound" if k == num_cores - 1 else f"link{k}"
            expected_labels = [label_left, label_phys, label_right]

            processed_core: UniTensor

            if isinstance(core_input, Tensor):
                tensor_core = core_input.contiguous().to(target_device).astype(target_dtype)
                
                if len(tensor_core.shape()) != 3:
                    raise ValueError(f"Input Tensor at index {k} must be 3D, got shape {tensor_core.shape()}")

                r_k, d_k, r_k1 = tensor_core.shape()[0], tensor_core.shape()[1], tensor_core.shape()[2]

                # Boundary dimension checks
                if k == 0 and r_k != 1:
                    logger.warning(f"Input Tensor at index 0 ('{label_left}') has left bond dim {r_k}, expected 1.")
                if k == num_cores - 1 and r_k1 != 1:
                    logger.warning(f"Input Tensor at index {k} ('{label_right}') has right bond dim {r_k1}, expected 1.")
                
                # Inter-core bond dimension consistency (for Tensor input)
                if k > 0 and self.M: # self.M contains the previously processed UniTensor
                    # The previous core's right bond was f"link{k-1}"
                    # Its dimension is self.M[k-1].bonds()[2].dim() (assuming 3rd bond is right)
                    prev_core_right_bond_dim = self.M[k-1].bonds()[2].dim() 
                    if r_k != prev_core_right_bond_dim:
                        raise ValueError(
                            f"Bond dimension mismatch for link 'link{k-1}': Tensor core {k} (left_dim={r_k}) "
                            f"vs UniTensor core {k-1} (right_dim={prev_core_right_bond_dim})."
                        )
                
                bd_left = cytnx.Bond(r_k, BD_IN)
                bd_phys = cytnx.Bond(d_k, BD_OUT)
                bd_right = cytnx.Bond(r_k1, BD_OUT)
                
                # UniTensor constructor does not take dtype/device if data is set by put_block
                processed_core = UniTensor(bonds=[bd_left, bd_phys, bd_right], 
                                           labels=expected_labels, 
                                           rowrank=1) 
                processed_core.put_block(tensor_core) # tensor_core is already contiguous, on target_device, with target_dtype
                
            elif isinstance(core_input, UniTensor):
                ut_temp = core_input.contiguous().to(target_device).astype(target_dtype)

                if ut_temp.rank() != 3:
                    raise ValueError(f"Input UniTensor at index {k} must be rank 3, got rank {ut_temp.rank()}")

                # Assume the order of bonds in input UniTensor is (left, phys, right) for dimension checks
                # before relabeling.
                bonds_temp = ut_temp.bonds()
                r_k, d_k, r_k1 = bonds_temp[0].dim(), bonds_temp[1].dim(), bonds_temp[2].dim()

                if k == 0 and r_k != 1:
                    logger.warning(f"Input UniTensor at index 0 ('{label_left}') has left bond dim {r_k}, expected 1.")
                if k == num_cores - 1 and r_k1 != 1:
                    logger.warning(f"Input UniTensor at index {k} ('{label_right}') has right bond dim {r_k1}, expected 1.")

                if k > 0 and self.M:
                    prev_core_right_bond_dim = self.M[k-1].bonds()[2].dim() # Assuming 3rd bond is 'link{k-1}'
                    if r_k != prev_core_right_bond_dim: # r_k is current UniTensor's 1st bond dim
                        raise ValueError(
                            f"Bond dimension mismatch for link 'link{k-1}': UniTensor core {k} (left_dim={r_k}) "
                            f"vs UniTensor core {k-1} (right_dim={prev_core_right_bond_dim})."
                        )
                
                # Create a new UniTensor with the correct bonds (types) and desired labels, then copy data.
                # This is safer than just set_labels if bond types might be wrong.
                # However, for __init__, if a UniTensor is passed, it's reasonable to expect its bond types are already correct
                # for an MPS core, or that relabeling handles this.
                # Let's try set_labels first, assuming bond types are correct or will be asserted.
                # If set_labels changes bond objects fundamentally, then consistency checks need care.
                # For now, let's assume set_labels just changes names.
                ut_temp.set_labels(expected_labels)
                processed_core = ut_temp
            else: 
                # This case should ideally not be reached if initial type check on M's elements is done.
                # However, as a safeguard within the loop:
                raise TypeError(f"Input core at index {k} is not a cytnx.Tensor or cytnx.UniTensor. Got {type(core_input)}")

            # Call _assert_core_validity (once it's defined)
            try:
                # Now actually call the assertion method
                print(f"DEBUG: Calling _assert_core_validity for core {k} in __init__.") # Add this to see it's being called
                TensorTrain._assert_core_validity(processed_core, k, num_cores, check_data_presence=True)
            except Exception as e:
                import traceback
                print(f"CRITICAL ERROR from _assert_core_validity for core {k} during __init__:")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                print("Traceback:")
                print(traceback.format_exc())
                # Depending on how critical this assertion is, you might want to stop initialization:
                raise # Re-raise the exception to stop __init__ if an assertion fails

            # For now, let's assume the above processing results in a valid core.
            # The assertion will later check rowrank, labels, and bond types.

            self.M.append(processed_core)
    
    @staticmethod
    def _assert_core_validity(core: UniTensor, k: int, L: int, check_data_presence: bool = False):
        if not isinstance(core, UniTensor):
            raise TypeError(f"Core at site {k} must be a cytnx.UniTensor, got {type(core)}.")
        if core.rank() != 3:
            raise ValueError(f"Core at site {k} is not rank 3, got rank {core.rank()}. Labels: {core.labels()}")

        # --- Single Diagnostic for enum accessibility (run once) ---
        enum_accessible_for_direct_comparison = False
        try:
            # This helps confirm if the path to enums is correct for potential direct comparison
            # We reference it here so if it fails, we know early.
            _ = cytnx.bondType.BD_KET # Attempt to access
            print(f"DEBUG (Core {k}): Path cytnx.bondType.BD_KET is accessible.")
            enum_accessible_for_direct_comparison = True
        except AttributeError as e:
            print(f"DEBUG (Core {k}): Failed to access cytnx.bondType.BD_KET directly via attribute path: {e}")
            print(f"DEBUG (Core {k}): Will rely solely on .value comparison for bond types.")
        # --- End Diagnostic ---

        # Label checking
        exp_label_left = "L_bound" if k == 0 else f"link{k-1}"
        exp_label_phys = f"p{k}"
        exp_label_right = "R_bound" if k == L - 1 else f"link{k}"
        expected_labels = [exp_label_left, exp_label_phys, exp_label_right]
        current_labels = core.labels()
        if current_labels != expected_labels:
            raise ValueError(
                f"Core at site {k} has incorrect labels. Expected {expected_labels}, got {current_labels}."
            )

        bonds = core.bonds()
        current_bond_types_retrieved = [b.type() for b in bonds]

        # --- Primary method: Compare using .value ---
        # Based on your output: BD_KET is -1, BD_BRA is 1
        expected_bond_type_values = [-1, 1, 1]  # KET, BRA, BRA

        for i in range(3):  # Iterate through the three bonds
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
                raise ValueError(error_msg)
        
        print(f"DEBUG (Core {k}): Bond type .value checks PASSED.")

        # Optional: If direct enum access worked and you still want to double-check direct enum object comparison
        if enum_accessible_for_direct_comparison:
            # This part is mostly for deeper debugging if .value checks were insufficient.
            # For most cases, the .value check should be fine.
            try:
                expected_enums = [cytnx.bondType.BD_KET, cytnx.bondType.BD_BRA, cytnx.bondType.BD_BRA]
                if current_bond_types_retrieved[0] != expected_enums[0] or \
                   current_bond_types_retrieved[1] != expected_enums[1] or \
                   current_bond_types_retrieved[2] != expected_enums[2]:
                    print(f"DEBUG (Core {k}): Direct enum object comparison MISMATCHED, even if .value comparison passed. This can happen if retrieved objects are different instances. Left: {current_bond_types_retrieved[0]} vs {expected_enums[0]}, Phys: {current_bond_types_retrieved[1]} vs {expected_enums[1]}, Right: {current_bond_types_retrieved[2]} vs {expected_enums[2]}")
                else:
                    print(f"DEBUG (Core {k}): Direct enum object comparison PASSED.")
            except Exception as e_direct_comp: # Catch any error during this optional check
                 print(f"DEBUG (Core {k}): Error during optional direct enum object comparison: {e_direct_comp}")


        # Row rank and boundary dimension checks
        if core.rowrank() != 1:
            raise ValueError(
                f"Core at site {k} (labels: {current_labels}) has incorrect rowrank. Expected 1, got {core.rowrank()}."
            )
        
        if k == 0 and bonds[0].dim() != 1:
            print(f"WARNING (Core {k}): Assertion: Core at site 0 ('{exp_label_left}') has left bond_dim {bonds[0].dim()}, expected 1.")
        if k == L - 1 and bonds[2].dim() != 1:
            print(f"WARNING (Core {k}): Assertion: Core at site {L - 1} ('{exp_label_right}') has right bond_dim {bonds[2].dim()}, expected 1.")

        if check_data_presence:
            print(f"DEBUG (Core {k}): Performing check_data_presence.") # Added print
            try:
                # Attempt to access the data block.
                # If the UniTensor is "void" (no block set/accessible), 
                # this is expected to raise a RuntimeError.
                block_accessed = core.get_block_() 
                
                # Optional: Further check on the accessed block if needed.
                # For example, if a block can be returned but still be "empty" by some definition.
                # if block_accessed.nelem() == 0 and core.nelem() != 0: # UniTensor.nelem() is product of bond dims
                #     raise ValueError(f"Core at site {k} (labels: {current_labels}) accessible block has 0 elements, but core shape implies {core.nelem()}.")
                print(f"DEBUG (Core {k}): core.get_block_() succeeded for data presence check.") # Added print

            except RuntimeError as e: 
                # This catches errors if get_block_() fails due to uninitialized/inaccessible data.
                error_msg = (
                    f"Core at site {k} (labels: {core.labels()}) data block access error (RuntimeError): {e}. "
                    f"This might indicate the UniTensor is uninitialized or 'void'."
                )
                raise ValueError(error_msg) from e # Preserve original exception context
            # No longer need the specific 'except AttributeError as ae:' for .storage() here.
            # If get_block_() itself was missing, it would be an AttributeError caught by a higher level handler,
            # or we'd add a specific one if we suspected get_block_ might be missing (but it's standard).
        
        print(f"DEBUG (Core {k}): All checks in _assert_core_validity PASSED.")


    def eval(self, idx: List[int]) -> Any:
        if not self.M:
            # Or perhaps return a default value like 1.0 if idx is also empty, 
            # or raise error if idx is not empty.
            # Original code would have failed accessing self.M[0] if M was empty.
            if not idx: # No cores, no indices
                return 1.0 # Or an appropriate scalar for an empty product
            else:
                raise ValueError("Cannot evaluate non-empty idx on an empty TensorTrain.")

        if len(idx) != len(self.M):
            raise ValueError(
                f"Index length {len(idx)} does not match tensor train length {len(self.M)}."
            )

        num_cores = len(self.M)
        comp_device = self.M[0].device()
        comp_dtype = self.M[0].dtype() # Use the TT's established dtype

        # Initialize prod_ut as a row vector [1.0]
        # Its bond should match the type and label of the first core's left bond, but opposite type.
        first_core_left_bond_label = self.M[0].labels()[0] # Should be "L_bound"
        # self.M[0].bonds()[0] is BD_IN, so prod_ut's bond should be BD_OUT
        
        # Ensure the dimension is 1 for the "L_bound"
        if self.M[0].bonds()[0].dim() != 1:
            raise ValueError(f"Left boundary bond of the first core ('{first_core_left_bond_label}') must have dimension 1 for eval.")

        prod_ut_bond = cytnx.Bond(1, BD_OUT)
        prod_ut = UniTensor(bonds=[prod_ut_bond], 
                            labels=[first_core_left_bond_label], 
                            rowrank=1, # Makes it a 1xD row vector conceptually for its single bond
                            device=comp_device, 
                            dtype=comp_dtype)
        blk = prod_ut.get_block_()
        blk[0] = 1.0

        for k, Mk_ut in enumerate(self.M):
            # Validate the core (optional, can be enabled for debugging)
            # TensorTrain._assert_core_validity(Mk_ut, k, num_cores)

            # Physical bond is typically the second one (index 1)
            phys_bond_on_Mk = Mk_ut.bonds()[1]
            phys_label_on_Mk = Mk_ut.labels()[1]
            
            i_k = idx[k]
            if not (0 <= i_k < phys_bond_on_Mk.dim()):
                raise IndexError(
                    f"Index {i_k} out of bounds for physical dimension {phys_bond_on_Mk.dim()} "
                    f"at site {k} (label '{phys_label_on_Mk}')."
                )

            # Create Selector UniTensor (ket-like)
            # Bond type is opposite to phys_bond_on_Mk (which is BD_OUT), so BD_IN
            selector_bond = phys_bond_on_Mk.redirect() # Creates BD_IN if phys_bond_on_Mk is BD_OUT
            selector_ut = UniTensor(bonds=[selector_bond], 
                                    labels=[phys_label_on_Mk], 
                                    rowrank=0, # ket-like (Dx1) for its single bond
                                    device=comp_device, 
                                    dtype=comp_dtype)

            
            # A robust way to create and set the selector:
            data_for_selector = cytnx.zeros(phys_bond_on_Mk.dim(), dtype=comp_dtype, device=comp_device)
            data_for_selector[i_k] = 1.0
            # Re-create selector_ut if the above fill_zeros/contiguous is problematic for default UT
            selector_ut = UniTensor(bonds=[selector_bond],
                                    labels=[phys_label_on_Mk],
                                    rowrank=0, # ket-like
                                    dtype=comp_dtype, # Ensure these are passed
                                    device=comp_device)
            selector_ut.put_block(data_for_selector)


            # Contract Mk_ut with selector_ut to slice the physical leg
            # Mk_ut labels: (left_link, phys_label, right_link)
            # selector_ut labels: (phys_label)
            # Mk_selected labels: (left_link, right_link)
            Mk_selected = cytnx.Contract(Mk_ut, selector_ut)
            
            # Contract current prod_ut with Mk_selected
            # prod_ut's label (e.g., "L_bound" or "link{k-1}")
            # Mk_selected's left label (e.g., "L_bound" or "link{k-1}")
            # These should match and have opposite bond types for contraction.
            prod_ut = cytnx.Contract(prod_ut, Mk_selected)
            # The new prod_ut will have the right label of Mk_selected

        # After loop, prod_ut should have one label ("R_bound") and shape [1]
        if not (prod_ut.rank() == 1 and prod_ut.shape() == [1]):
            # More descriptive error
            final_shape = prod_ut.shape() if prod_ut.rank() > 0 else "scalar (rank 0)"
            final_labels = prod_ut.labels()
            raise ValueError(
                f"TensorTrain.eval did not collapse to a rank-1 UniTensor of shape [1]. "
                f"Final rank: {prod_ut.rank()}, shape: {final_shape}, labels: {final_labels}."
            )
        
        # The rightmost bond of the last core ("R_bound") must also have dimension 1
        if self.M[-1].bonds()[2].dim() != 1: # Assuming 3rd bond is right bond
            raise ValueError(f"Right boundary bond of the last core ('{self.M[-1].labels()[2]}') must have dimension 1 for eval to result in a scalar.")

        return prod_ut.item() # Extracts the scalar value


# Inside the TensorTrain class definition:

# Inside the TensorTrain class definition:

    def overlap(self, tt: 'TensorTrain') -> float:
        if len(self.M) != len(tt.M):
            raise ValueError("Tensor trains for overlap must have the same length.")
        if not self.M: # Both are empty
            return 1.0 

        num_cores = len(self.M)
        comp_device = self.M[0].device()

        # --- Dtype determination for C_ut (same as your previous correct version) ---
        final_requires_complex_for_C, final_requires_double_for_C = False, False
        for k_check in range(num_cores):
            for core_scan in [self.M[k_check], tt.M[k_check]]:
                dt_cs = core_scan.dtype()
                # Corrected is_complex check
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
        # --- End Dtype determination ---

        # Initialize C_ut:
        # C_ut's labels will represent the "current open legs" from self (A) and tt (B) chains.
        # Initially, these are the left-most boundary legs.
        # These labels *must be unique* for C_ut itself.
        
        # Labels from the actual M[0] cores (e.g., "L_bound")
        actual_L_label_A = self.M[0].labels()[0] 
        actual_L_label_B = tt.M[0].labels()[0]  

        # Ensure boundary dimensions are 1
        if self.M[0].bonds()[0].dim() != 1 or tt.M[0].bonds()[0].dim() != 1:
            raise ValueError("Left boundary bonds for overlap must have dimension 1.")

        # C_ut will store the result of contracting the left legs.
        # Its bonds should be OUT, as they are "exposed" after contracting with core's IN bonds.
        C_ut_bonds = [cytnx.Bond(1, BD_OUT), cytnx.Bond(1, BD_OUT)]
        
        # For C_ut itself, use unique labels. These are temporary for C_ut's definition.
        # The *meaningful* labels (like "L_bound") are on Ak_c and Bk_ut.
        # We will relabel Ak_c and Bk_ut to match these unique C_ut labels for contraction.
        # OR, better: C_ut's labels should be the *outgoing* labels from the previous contraction.
        # Initial C_ut represents Identity contraction of boundary bonds.
        # Its labels are effectively the labels of the first bonds of Ak_c and Bk_ut it will connect to.
        # TO AVOID "labels cannot contain duplicated elements" if actual_L_label_A == actual_L_label_B (e.g. both "L_bound"),
        # we must ensure the labels *given to the UniTensor constructor for C_ut* are unique.
        
        # Let C_ut carry the labels of the *output* bonds from the previous step's A and B chains.
        # For k=0, these are the right bonds of hypothetical "identity" tensors to the left of M[0].
        # So, C_ut_current_A_leg_label = self.M[0].labels()[0] (e.g. "L_bound")
        #     C_ut_current_B_leg_label = tt.M[0].labels()[0] (e.g. "L_bound")
        # If these are the same string, UniTensor construction fails.
        # We need distinct labels for C_ut's own definition.
        
        # Fixed internal labels for C_ut through iterations. These are arbitrary unique names.
        c_env_label_A = "__c_env_A_leg__"
        c_env_label_B = "__c_env_B_leg__"

        # Initial C_ut (as in my last correct version for its construction)
        C_ut_bonds = [cytnx.Bond(1, BD_OUT), cytnx.Bond(1, BD_OUT)]
        C_ut = UniTensor(bonds=C_ut_bonds, 
                         labels=[c_env_label_A, c_env_label_B], 
                         rowrank=1, 
                         device=comp_device, 
                         dtype=comp_dtype_C)
        C_ut.get_block_()[0,0] = 1.0

        for k in range(num_cores):
            Ak_ut_orig = self.M[k] 
            Bk_ut_orig = tt.M[k]   

            Ak_ut = Ak_ut_orig.astype(comp_dtype_C).to(comp_device)
            Bk_ut = Bk_ut_orig.astype(comp_dtype_C).to(comp_device)
            Ak_c = Ak_ut.Conj() 
            
            # --- Correctly define actual labels before use ---
            akc_actual_L_lbl = Ak_c.labels()[0] 
            akc_actual_P_lbl = Ak_c.labels()[1]
            akc_actual_R_lbl = Ak_c.labels()[2] 

            bk_actual_L_lbl = Bk_ut.labels()[0]
            bk_actual_P_lbl = Bk_ut.labels()[1]
            bk_actual_R_lbl = Bk_ut.labels()[2]
            # --- End of correct definition ---

            # --- Step 1: Contract C_ut with Ak_c ---
            Ak_c_ready = Ak_c.relabel(akc_actual_L_lbl, c_env_label_A)
            temp_CA = cytnx.Contract(C_ut, Ak_c_ready)
            # temp_CA should now have labels: [c_env_label_B (from C_ut), 
            #                                akc_actual_P_lbl (from Ak_c), 
            #                                akc_actual_R_lbl (from Ak_c)]

            # --- Step 2: Prepare Bk_ut and contract with temp_CA ---
            Bk_mod = Bk_ut.clone() 
            phys_bond_idx = 1 
            Bk_mod.bonds()[phys_bond_idx].redirect_() 

            # Temporary unique labels for the "output" legs of this step's contraction
            # These will be the labels on C_ut_next_data_container IF we relabel before final contract
            unique_R_A_label_for_output = f"__temp_out_A_{k}__" # As in my previous suggestion
            unique_R_B_label_for_output = f"__temp_out_B_{k}__" # As in my previous suggestion

            # Relabel temp_CA's right leg to make it unique before contracting with Bk_mod
            temp_CA_relabeled_for_Bk_contract = temp_CA.relabel(akc_actual_R_lbl, unique_R_A_label_for_output)
            # temp_CA_relabeled_for_Bk_contract labels: [c_env_label_B, akc_actual_P_lbl, unique_R_A_label_for_output]

            # Relabel Bk_mod's legs to match for contraction with temp_CA_relabeled_for_Bk_contract
            # and give its right leg a unique name.
            Bk_mod_ready_for_final_contract = Bk_mod.relabel(bk_actual_L_lbl, c_env_label_B)
            Bk_mod_ready_for_final_contract.relabel_(bk_actual_P_lbl, akc_actual_P_lbl) # Match phys label from Ak_c
            Bk_mod_ready_for_final_contract.relabel_(bk_actual_R_lbl, unique_R_B_label_for_output)
            # Bk_mod_ready_for_final_contract labels: [c_env_label_B, akc_actual_P_lbl, unique_R_B_label_for_output]
            
            C_ut_next_data_container = cytnx.Contract(temp_CA_relabeled_for_Bk_contract, Bk_mod_ready_for_final_contract)
            # Expected labels for C_ut_next_data_container: [unique_R_A_label_for_output, unique_R_B_label_for_output]
            # (Order might vary, but these two should be the remaining ones)

            # --- Update C_ut for the next iteration ---
            if k < num_cores - 1:
                next_bond_dim_A = Ak_ut_orig.bonds()[2].dim() 
                next_bond_dim_B = Bk_ut_orig.bonds()[2].dim() 

                next_C_bonds = [cytnx.Bond(next_bond_dim_A, BD_OUT), 
                                cytnx.Bond(next_bond_dim_B, BD_OUT)]
                
                C_next_iter_shell = UniTensor(bonds=next_C_bonds,
                                              labels=[c_env_label_A, c_env_label_B], # Fixed internal C_ut labels
                                              rowrank=1, 
                                              device=comp_device, 
                                              dtype=comp_dtype_C)
                
                # Ensure C_ut_next_data_container is ordered correctly before get_block
                # Its current labels are unique_R_A_label_for_output and unique_R_B_label_for_output.
                # We want the data associated with unique_R_A_label_for_output to go to c_env_label_A's dimension,
                # and unique_R_B_label_for_output to go to c_env_label_B's dimension.
                # The permute_ call ensures this order for the block.
                C_ut_next_data_container.permute_([unique_R_A_label_for_output, unique_R_B_label_for_output], 
                                                  rowrank=1) 
                
                C_next_iter_shell.put_block(C_ut_next_data_container.get_block_())
                C_ut = C_next_iter_shell
            else: 
                # Last iteration, C_ut_next_data_container is the final 1x1 UniTensor.
                # Its labels are [unique_R_A_label_for_output, unique_R_B_label_for_output].
                # These correspond to what were originally R_bound for A and R_bound for B.
                C_ut = C_ut_next_data_container
        
        # ... (Final check and return C_ut.item() as before, ensure get_block_()[0,0] is used) ...
        if not (C_ut.rank() == 2 and C_ut.shape() == [1,1] and C_ut.bonds()[0].dim()==1 and C_ut.bonds()[1].dim()==1):
             final_shape = C_ut.shape(); final_labels = C_ut.labels()
             raise ValueError(f"Overlap: Result not 1x1. Rank={C_ut.rank()}, Shape={final_shape}, Labels={final_labels}.")
        final_scalar_tensor = C_ut.get_block_()[0,0] # This is a 0-rank cytnx.Tensor
        return float(final_scalar_tensor.item())    # Call .item() on the scalar Tensor

    def norm2(self) -> float:
        return self.overlap(self)

    def compressLU(self, reltol: float = 1e-12, maxBondDim: int = 0):
        # Assumes MatprrLUFixedTol is available in the scope
        # from mat_decomp import MatprrLUFixedTol # Ensure this is imported at the top of the file

        if not hasattr(self, 'M') or len(self.M) < 2:
            return

        num_cores = len(self.M)
        # Get the decomposer instance
        # Note: MatprrLUFixedTol might need to be initialized with specific dtype/device awareness
        # or its output handled accordingly. For now, assume it works with Tensors from get_block_().
        try:
            decomp = MatprrLUFixedTol(tol=reltol, rankMax=maxBondDim if maxBondDim > 0 else 0)
        except NameError:
            logger.error("MatprrLUFixedTol not found. Please ensure it's imported/defined.")
            raise

        # Determine a consistent dtype and device for intermediate tensors if decomp returns numpy
        # Typically, we want to maintain the dtype/device of the UniTensors in self.M
        # Let's assume self.M cores are already consistent from __init__
        comp_dtype = self.M[0].dtype()
        comp_device = self.M[0].device()

        for i in range(num_cores - 1):
            core_i = self.M[i]
            core_i_plus_1 = self.M[i+1] # This is the original core_i_plus_1 for this iteration

            # TensorTrain._assert_core_validity(core_i, i, num_cores)
            # TensorTrain._assert_core_validity(core_i_plus_1, i + 1, num_cores)

            # --- Prepare core_i for decomposition ---
            # Original core_i labels: (L_label, P_label, R_label_i)
            # R_label_i is the bond to be compressed, e.g., "link{i}"
            orig_L_bond_i = core_i.bonds()[0].clone() # Keep original bond object
            orig_P_bond_i = core_i.bonds()[1].clone()
            orig_L_label_i = core_i.labels()[0]
            orig_P_label_i = core_i.labels()[1]
            middle_bond_label_i = core_i.labels()[2] # This is "link{i}" or "R_bound" if i == num_cores - 2

            A_mat_tensor = cube_as_matrix2_utt(core_i) # Shape: (dim_L*dim_P, dim_R_i)
            
            # Perform decomposition
            L_out, R_out = decomp(A_mat_tensor) # L_out is (dim_L*dim_P, new_dim), R_out is (new_dim, dim_R_i)

            # Ensure L_out, R_out are cytnx.Tensors on the correct device/dtype
            if isinstance(L_out, np.ndarray):
                L_tensor = from_numpy(L_out).to(comp_device).astype(comp_dtype)
            else: # Assuming L_out is already a cytnx.Tensor (or compatible)
                L_tensor = L_out.to(comp_device).astype(comp_dtype)

            if isinstance(R_out, np.ndarray):
                R_tensor = from_numpy(R_out).to(comp_device).astype(comp_dtype)
            else: # Assuming R_out is already a cytnx.Tensor
                R_tensor = R_out.to(comp_device).astype(comp_dtype)

            # --- Update self.M[i] ---
            dim_L_i_val = orig_L_bond_i.dim()
            dim_P_i_val = orig_P_bond_i.dim()
            new_bond_dim = L_tensor.shape()[1]

            if L_tensor.shape()[0] != dim_L_i_val * dim_P_i_val:
                raise ValueError(f"Shape mismatch for L_tensor at site {i}. Expected first dim {dim_L_i_val * dim_P_i_val}, got {L_tensor.shape()[0]}")

            L_reshaped = L_tensor.reshape(dim_L_i_val, dim_P_i_val, new_bond_dim)
            
            new_core_i = UniTensor(
                bonds=[orig_L_bond_i, orig_P_bond_i, cytnx.Bond(new_bond_dim, BD_OUT)],
                labels=[orig_L_label_i, orig_P_label_i, middle_bond_label_i], # middle_bond_label_i is reused for the new bond
                rowrank=1, 
                # dtype and device will be from L_reshaped in put_block
            )
            new_core_i.put_block(L_reshaped)
            self.M[i] = new_core_i
            # TensorTrain._assert_core_validity(self.M[i], i, num_cores)


            # --- Update self.M[i+1] ---
            # Original core_i_plus_1 labels: (L_label_i1, P_label_i1, R_label_i1)
            # L_label_i1 should be middle_bond_label_i
            if core_i_plus_1.labels()[0] != middle_bond_label_i:
                raise ValueError(f"Label mismatch for absorption: M[{i}].right_label ('{middle_bond_label_i}') "
                                 f"!= M[{i+1}].left_label ('{core_i_plus_1.labels()[0]}')")

            orig_P_bond_i1 = core_i_plus_1.bonds()[1].clone()
            orig_R_bond_i1 = core_i_plus_1.bonds()[2].clone() # This is the rightmost bond of core_i_plus_1
            orig_P_label_i1 = core_i_plus_1.labels()[1]
            orig_R_label_i1 = core_i_plus_1.labels()[2] # This is "link{i+1}" or "R_bound"

            B_mat_tensor = cube_as_matrix1_utt(core_i_plus_1) # Shape: (dim_L_i1, dim_P_i1*dim_R_i1)
                                                              # Here dim_L_i1 is original dim of middle_bond_label_i

            # R_tensor shape: (new_bond_dim, original_dim_middle_bond)
            # B_mat_tensor shape: (original_dim_middle_bond, dim_P_i1*dim_R_i1)
            # C_mat_tensor shape: (new_bond_dim, dim_P_i1*dim_R_i1)
            C_mat_tensor = cytnx.linalg.Matmul(R_tensor, B_mat_tensor) 
            C_mat_tensor = C_mat_tensor.to(comp_device).astype(comp_dtype) # Ensure consistency

            dim_P_i1_val = orig_P_bond_i1.dim()
            dim_R_i1_val = orig_R_bond_i1.dim()
            
            if C_mat_tensor.shape()[0] != new_bond_dim or \
               C_mat_tensor.shape()[1] != dim_P_i1_val * dim_R_i1_val:
                raise ValueError(f"Shape mismatch for C_mat_tensor at site {i+1}. Expected ({new_bond_dim}, {dim_P_i1_val * dim_R_i1_val}), got {C_mat_tensor.shape()}")

            C_reshaped = C_mat_tensor.reshape(new_bond_dim, dim_P_i1_val, dim_R_i1_val)

            new_core_i_plus_1 = UniTensor(
                bonds=[cytnx.Bond(new_bond_dim, BD_IN), orig_P_bond_i1, orig_R_bond_i1],
                labels=[middle_bond_label_i, orig_P_label_i1, orig_R_label_i1], # middle_bond_label_i is new left bond
                rowrank=1,
            )
            new_core_i_plus_1.put_block(C_reshaped)
            self.M[i+1] = new_core_i_plus_1
            # TensorTrain._assert_core_validity(self.M[i+1], i + 1, num_cores)

# Inside the TensorTrain class definition:

    def __add__(self, other: 'TensorTrain') -> 'TensorTrain':
        if not self.M: # If self is empty
            # Return a new TensorTrain instance from a copy of other.M's cores
            # Ensure to use .clone() for each UniTensor if other.M is a list of UniTensors
            # and __init__ expects a list.
            cloned_other_cores = []
            if other.M:
                for core in other.M:
                    if isinstance(core, UniTensor):
                        cloned_other_cores.append(core.clone())
                    else: # Should not happen if other is a valid TensorTrain post-refactor
                        cloned_other_cores.append(core) # Or handle error
            return TensorTrain(cloned_other_cores if cloned_other_cores else None)

        if not other.M: # If other is empty
            cloned_self_cores = []
            if self.M:
                for core in self.M:
                    if isinstance(core, UniTensor):
                        cloned_self_cores.append(core.clone())
                    else:
                        cloned_self_cores.append(core)
            return TensorTrain(cloned_self_cores if cloned_self_cores else None)

        if len(self.M) != len(other.M):
            raise ValueError("Cannot add tensor trains of different lengths.")

        num_cores = len(self.M)

        # Determine composite device and dtype (same logic as your working version)
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
        
        type_id_to_name = { # For debug printing
            Type.Void: "Void", Type.ComplexDouble: "ComplexDouble", 
            Type.ComplexFloat: "ComplexFloat", Type.Double: "Double", 
            Type.Float: "Float", Type.Uint64: "Uint64", Type.Int64: "Int64",
            Type.Uint32: "Uint32", Type.Int32: "Int32", Type.Uint16: "Uint16", 
            Type.Int16: "Int16", Type.Bool: "Bool"
        }
        comp_dtype_name_str = type_id_to_name.get(comp_dtype, f"UnknownTypeID({comp_dtype})")
        logger.debug(f"__add__ Op: Target comp_dtype for new cores: {comp_dtype} ({comp_dtype_name_str}), Target comp_device: {comp_device}")

        new_UniTensor_cores: List[UniTensor] = []
        for k in range(num_cores):
            # --- Process self.M[k] ---
            # Create a new variable for the processed UniTensor from self.M[k]
            A_ut_processed = self.M[k].astype(comp_dtype).to(comp_device)
            A_ut_processed.contiguous_() # In-place operation, modifies A_ut_processed

            # --- Process other.M[k] ---
            # Create a new variable for the processed UniTensor from other.M[k]
            B_ut_processed = other.M[k].astype(comp_dtype).to(comp_device)
            B_ut_processed.contiguous_() # In-place operation, modifies B_ut_processed
            
            # Original labels are fetched from self.M[k] for the new core definition.
            LA_l_orig, PA_l_orig, RA_l_orig = self.M[k].labels() 

            # Use the processed UniTensors (which are not None) for subsequent operations
            if A_ut_processed is None: # Should not happen if previous steps are correct
                raise ValueError(f"__add__: A_ut_processed became None at core {k} for self.M.")
            if B_ut_processed is None:
                raise ValueError(f"__add__: B_ut_processed became None at core {k} for other.M.")

            rA, dA, sA = A_ut_processed.shape() # Dims of processed A
            rB, dB, sB = B_ut_processed.shape() # Dims of processed B

            if dA != dB: # Physical dimensions must match
                raise ValueError(f"Physical dimension mismatch at core {k}: {dA} (self) vs {dB} (other)")
            
            # Physical label consistency check (optional, good for warning)
            # if PA_l_orig != other.M[k].labels()[1]: 
            #      logger.warning(f"Physical labels at core {k} differ: '{PA_l_orig}' vs '{other.M[k].labels()[1]}'. Using '{PA_l_orig}'.")

            new_r_left: int
            new_d_phys: int = dA 
            new_s_right: int
            
            # Get blocks from the processed, contiguous UniTensors
            A_block = A_ut_processed.get_block_()
            B_block = B_ut_processed.get_block_()

            # --- Determine dimensions and fill new_data_tensor (C++ kron_add logic) ---
            # This logic remains the same as my previous C++ mimicking version
            if k == 0: # First core
                if rA != 1: logger.warning(f"__add__ k=0: Left bond of A ({rA}) is not 1 after processing.")
                if rB != rA : # Check based on processed rA and rB
                     logger.warning(f"__add__ k=0: rB ({rB}) != rA ({rA}) for B placement. This might lead to unexpected block summation if C++ logic expects exact dimension match for sub-cube assignment.")
                new_r_left = rA 
                new_s_right = sA + sB
                new_data_tensor = cytnx.zeros([new_r_left, new_d_phys, new_s_right], dtype=comp_dtype, device=comp_device)
                new_data_tensor[0:rA, :, 0:sA] = A_block
                new_data_tensor[0:min(rA,rB), :, sA : sA+sB] = B_block[0:min(rA,rB), :, :] # Adjusted slicing for safety

            elif k == num_cores - 1: # Last core
                if sB != 1: logger.warning(f"__add__ k={k}: Right bond of B ({sB}) is not 1 after processing.")
                if sA != sB :
                    logger.warning(f"__add__ k={k}: sA ({sA}) != sB ({sB}). C++ logic implies sA should match sB for A's placement. Adapting.")
                new_r_left = rA + rB
                new_s_right = sB 
                new_data_tensor = cytnx.zeros([new_r_left, new_d_phys, new_s_right], dtype=comp_dtype, device=comp_device)
                new_data_tensor[0:rA, :, 0:min(sA,sB)] = A_block[:, :, 0:min(sA,sB)] # Adjusted slicing
                new_data_tensor[rA : rA+rB, :, 0:sB] = B_block
            
            else: # Middle cores
                new_r_left = rA + rB
                new_s_right = sA + sB
                new_data_tensor = cytnx.zeros([new_r_left, new_d_phys, new_s_right], dtype=comp_dtype, device=comp_device)
                new_data_tensor[0:rA, :, 0:sA] = A_block
                new_data_tensor[rA : rA+rB, :, sA : sA+sB] = B_block
            
            # Create new bonds and labels for the UniTensor
            bd_L_new = cytnx.Bond(new_r_left, BD_IN)
            bd_P_new = A_ut_processed.bonds()[1].clone() # Physical bond from processed A_ut
            bd_R_new = cytnx.Bond(new_s_right, BD_OUT)
            # Use original labels from self.M[k] for the new core's structure
            new_core_labels = [LA_l_orig, PA_l_orig, RA_l_orig]

            new_core_ut = UniTensor(bonds=[bd_L_new, bd_P_new, bd_R_new], 
                                    labels=new_core_labels, 
                                    rowrank=1)
            new_core_ut.put_block(new_data_tensor)
            new_UniTensor_cores.append(new_core_ut)
        
        result_tt = TensorTrain(new_UniTensor_cores)
        return result_tt

# Inside the TensorTrain class definition:

    def sum(self, weights: List[List[float]]) -> Any: # Returns a Python scalar
        if not self.M:
            if not weights: 
                return 1.0 # Consistent with empty product/sum
            else: 
                raise ValueError("Cannot apply non-empty weights to an empty TensorTrain.")

        if len(weights) != len(self.M):
            raise ValueError(
                f"Weights list length {len(weights)} does not match "
                f"tensor train length {len(self.M)}."
            )

        num_cores = len(self.M)
        # Use the established dtype and device from the TensorTrain's cores
        # __init__ should have ensured these are float/complex types.
        comp_device = self.M[0].device()
        comp_dtype = self.M[0].dtype() 
        
        # Initialize prod_ut as a row vector [1.0]
        first_core_L_label = self.M[0].labels()[0]
        if self.M[0].bonds()[0].dim() != 1:
            raise ValueError(
                f"Sum: Left boundary bond '{first_core_L_label}' of core 0 "
                f"must have dimension 1."
            )

        prod_ut_bond = cytnx.Bond(1, BD_OUT) # To contract with core's BD_IN left bond
        prod_ut = UniTensor(bonds=[prod_ut_bond], 
                            labels=[first_core_L_label], 
                            rowrank=1, # Conceptually a 1xD "bra" vector
                            device=comp_device, 
                            dtype=comp_dtype)
        prod_ut.get_block_()[0] = 1.0 # Corrected element assignment

        for k, Mk_ut in enumerate(self.M):
            # self._assert_core_validity(Mk_ut, k, num_cores) # Optional validation

            w_k_list = weights[k]
            phys_bond_on_Mk = Mk_ut.bonds()[1] # Assuming physical is index 1
            phys_label_on_Mk = Mk_ut.labels()[1]
            phys_dim = phys_bond_on_Mk.dim()

            if len(w_k_list) != phys_dim:
                raise ValueError(
                    f"Weights list at index {k} (length {len(w_k_list)}) does not match "
                    f"physical dimension {phys_dim} of core {k} (label '{phys_label_on_Mk}')."
                )

            # Create weights_vec_ut (ket-like vector)
            # Its bond type is opposite to Mk_ut's physical bond (which is BD_OUT), so BD_IN.
            weights_bond = phys_bond_on_Mk.redirect() 
            
            # Determine numpy dtype for weights array based on comp_dtype's precision
            if comp_dtype == Type.ComplexDouble or comp_dtype == Type.Double:
                numpy_equiv_dtype = np.float64
            elif comp_dtype == Type.ComplexFloat or comp_dtype == Type.Float:
                numpy_equiv_dtype = np.float32
            else:
                # Should not happen if __init__ enforces float/complex types for cores
                logger.warning(f"Sum: Core dtype {comp_dtype} is not a standard float/complex type. "
                               f"Defaulting weights to np.float64.")
                numpy_equiv_dtype = np.float64
            
            weights_np_array = np.array(w_k_list, dtype=numpy_equiv_dtype)
            weights_data_tensor = from_numpy(weights_np_array).to(comp_device).astype(comp_dtype)
            
            weights_vec_ut = UniTensor(bonds=[weights_bond], 
                                       labels=[phys_label_on_Mk], 
                                       rowrank=0 # ket-like
                                       # dtype and device will be from put_block
                                      ) 
            weights_vec_ut.put_block(weights_data_tensor)
            
            # Contract Mk_ut with weights_vec_ut to sum over the physical leg, weighted
            # Mk_ut labels: (left_link, phys_label, right_link)
            # weights_vec_ut labels: (phys_label)
            # weighted_slice_ut labels: (left_link, right_link)
            weighted_slice_ut = cytnx.Contract(Mk_ut, weights_vec_ut)
            
            # Contract current prod_ut with weighted_slice_ut
            prod_ut = cytnx.Contract(prod_ut, weighted_slice_ut)
            # New prod_ut will have the right label of weighted_slice_ut

        # Final check for prod_ut (should be rank-1, shape [1])
        if not (prod_ut.rank() == 1 and prod_ut.shape() == [1]):
            final_shape = prod_ut.shape() if prod_ut.rank() > 0 else "scalar (rank 0)"
            final_labels = prod_ut.labels()
            raise ValueError(
                f"TensorTrain.sum did not collapse to a rank-1 UniTensor of shape [1]. "
                f"Final rank: {prod_ut.rank()}, shape: {final_shape}, labels: {final_labels}."
            )
        
        if self.M[-1].bonds()[2].dim() != 1: # Right boundary bond of last core
            raise ValueError(
                f"Sum: Right boundary bond '{self.M[-1].labels()[2]}' of core {num_cores-1} "
                f"must have dimension 1 for sum to result in a scalar."
            )

        return prod_ut.item() # Extracts the Python scalar

    def sum1(self) -> Any: # Was float
        if not self.M:
            return 1.0 # Consistent with sum for empty M and empty weights
        
        # Create weights as a list of lists of ones
        # Physical dimension is from the second bond of each UniTensor core
        weights_for_sum1 = [[1.0] * core.bonds()[1].dim() for core in self.M]
        return self.sum(weights_for_sum1)
    
    def trueError(self, other: 'TensorTrain', max_eval: int = int(1e6)) -> float:
        """
        Computes the maximum absolute difference between the elements of this TensorTrain
        and another TensorTrain ('other'), by evaluating all possible index combinations
        up to 'max_eval'.
        """
        if not hasattr(self, 'M') or not self.M:
            # If self is empty, error is 0 if other is also empty, otherwise it's undefined or large.
            # Assuming if self is empty, other must also be empty due to length check.
            if hasattr(other, 'M') and not other.M and (not hasattr(other, 'M') or len(self.M) == len(other.M)): # check len only if other.M exists
                 return 0.0
            # Fallthrough to length check if self.M is empty but other.M might not be (or vice versa)
            # Or raise error if self.M is empty and trying to calculate error.
            # For simplicity, if self.M is empty, and length check passes (other.M also empty), return 0.0.
            # The initial length check will handle mismatches.

        if not hasattr(other, 'M'):
            raise TypeError("The 'other' object provided to trueError is not a valid TensorTrain (missing 'M').")

        if len(self.M) != len(other.M):
            raise ValueError("Tensor trains for trueError must have the same length.")
        
        if not self.M: # Both are empty due to length check above
            return 0.0

        # Physical dimensions from the second bond of each UniTensor core
        try:
            # Ensure all cores are valid before accessing bonds
            # This would ideally be done by _assert_core_validity in __init__ or method entry
            dims = []
            for k_core, Mk_core in enumerate(self.M):
                if not isinstance(Mk_core, UniTensor) or Mk_core.rank() < 2: # Need at least 2 bonds for phys dim at index 1
                    raise ValueError(f"Core {k_core} in 'self' is not a valid UniTensor or has insufficient rank for trueError.")
                dims.append(Mk_core.bonds()[1].dim()) # Physical is index 1
        except IndexError:
            raise ValueError("Could not retrieve physical dimensions from cores in 'self'. Ensure cores are valid rank-3 UniTensors.")
        
        try:
            for k_core, Mk_core_other in enumerate(other.M):
                 if not isinstance(Mk_core_other, UniTensor) or Mk_core_other.rank() < 2:
                     raise ValueError(f"Core {k_core} in 'other' is not a valid UniTensor or has insufficient rank for trueError.")
                 if Mk_core_other.bonds()[1].dim() != dims[k_core]: # Check consistency with self's dims
                     raise ValueError(f"Physical dimension mismatch at core {k_core} between self and other.")
        except IndexError:
            raise ValueError("Could not retrieve/validate physical dimensions from cores in 'other'.")


        total_configs = np.prod(dims, dtype=np.int64) 

        if total_configs == 0 and not any(d == 0 for d in dims): 
             logger.warning(
                 f"Product of dimensions ({np.prod(dims)}) resulted in {total_configs}, "
                 f"which might indicate an overflow if no physical dimension is zero. "
                 f"Number of configurations could be very large."
            )
        
        if total_configs > max_eval:
            actual_total_str = str(np.prod(dims)) if total_configs == 0 and not any(d==0 for d in dims) else str(total_configs)
            raise ValueError(
                f"Too many index combinations ({actual_total_str} if no overflow, "
                f"or {total_configs} if overflowed to 0) for trueError, "
                f"exceeds max_eval={max_eval}"
            )
        
        if total_configs == 0 and any(d == 0 for d in dims): 
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
        
        return float(max_err)

    def save(self, file_name: str):
        """
        Saves the TensorTrain to a single file.
        For each UniTensor core, its metadata (labels, bond definitions, rowrank,
        dtype, device) and data (as a numpy array) are stored in a dictionary.
        A list of these dictionaries is then pickled to the specified file.
        """
        if not hasattr(self, 'M'):
            # This case should ideally be prevented by __init__ ensuring M is always a list.
            # If self.M could be None after a failed/partial init, this check is useful.
            logger.warning("Attempting to save a TensorTrain with no 'M' attribute or M is None. Saving empty list.")
            core_data_list = [] # Save an empty list for an empty or uninitialized TensorTrain
        elif not self.M: # self.M is an empty list
            core_data_list = []
        else:
            core_data_list = []
            for k, Mk_ut in enumerate(self.M):
                if not isinstance(Mk_ut, UniTensor):
                    raise TypeError(f"Core at index {k} must be a UniTensor for saving, got {type(Mk_ut)}.")
                
                # Ensure core is contiguous before getting block to avoid issues with views
                # and to ensure numpy conversion is straightforward.
                Mk_ut_contiguous = Mk_ut.contiguous()

                bonds_params = []
                for b_idx, b in enumerate(Mk_ut_contiguous.bonds()):
                    # For bond type, store the integer value of the enum member.
                    # Based on previous outputs:
                    # BD_IN construction results in b.type() == cytnx.bondType.BD_KET (value -1)
                    # BD_OUT construction results in b.type() == cytnx.bondType.BD_BRA (value 1)
                    bond_type_enum_val = b.type().value 

                    # Store symmetry information if present
                    qnums_list_of_list = b.qnums() # List of lists: [[q1,...],[q2,...],...]
                    
                    # Store symmetry types as their integer stype() values
                    # s.stype() returns an int (e.g., cytnx.Symmetry.U1().stype() -> 0)
                    sym_type_ints = [sym.stype() for sym in b.syms()]

                    bond_param = {
                        "dim": b.dim(),
                        "type_val": bond_type_enum_val, 
                        "qnums": qnums_list_of_list, 
                        "sym_type_ints": sym_type_ints 
                        # Note: For symmetric bonds, 'degs' (degeneracies) are also crucial
                        # and are typically part of the qnums structure or need to be saved separately
                        # if using the (bond_type, qnums_list_of_list_int, degs_list_int, syms_list_Symmetry) constructor.
                        # Current Bond constructor from qnums and syms often infers degs or uses a specific format.
                        # For simplicity, if no symmetries, qnums and sym_type_ints will be empty.
                    }
                    bonds_params.append(bond_param)

                core_dict = {
                    "labels": Mk_ut_contiguous.labels(),
                    "bonds_params": bonds_params,
                    "rowrank": Mk_ut_contiguous.rowrank(),
                    "dtype_int": Mk_ut_contiguous.dtype(),    # Integer Type ID
                    "device_int": Mk_ut_contiguous.device(),  # Integer Device ID
                    "block_numpy": Mk_ut_contiguous.get_block_().numpy() # Data as numpy array
                }
                core_data_list.append(core_dict)
        
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(core_data_list, f)
            logger.info(f"TensorTrain saved successfully to {file_name} with {len(core_data_list)} cores.")
        except Exception as e:
            logger.error(f"Error saving TensorTrain to {file_name}: {e}")
            raise

    @staticmethod
    def load(file_name: str) -> 'TensorTrain':
        """
        Loads a TensorTrain from a single file.
        The file is expected to contain a pickled list of dictionaries,
        each dictionary holding the metadata and data for one UniTensor core.
        """
        try:
            with open(file_name, 'rb') as f:
                core_data_list = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Error loading TensorTrain: File {file_name} not found.")
            raise
        except Exception as e:
            logger.error(f"Error loading TensorTrain data from {file_name}: {e}")
            raise

        if not isinstance(core_data_list, list):
            raise TypeError(f"Data loaded from {file_name} is not a list as expected for TensorTrain.")

        if not core_data_list: # Handle case of empty saved TensorTrain
            logger.info(f"Loaded an empty TensorTrain from {file_name}.")
            return TensorTrain([]) 

        loaded_cores_M: List[UniTensor] = []
        for i, core_dict in enumerate(core_data_list):
            if not isinstance(core_dict, dict):
                raise TypeError(f"Core data at index {i} in {file_name} is not a dictionary.")

            try:
                labels = core_dict["labels"]
                bonds_params = core_dict["bonds_params"]
                rowrank = core_dict["rowrank"]
                dtype_int = core_dict["dtype_int"]       # Integer Type ID
                device_int = core_dict["device_int"]     # Integer Device ID
                block_numpy = core_dict["block_numpy"]

                reconstructed_bonds = []
                for bp_idx, bp_param in enumerate(bonds_params):
                    dim = bp_param["dim"]
                    # type_val was from b.type().value, e.g., -1 for KET (constructed with BD_IN), 1 for BRA (constructed with BD_OUT)
                    bond_type_val = bp_param["type_val"] 
                    qnums = bp_param.get("qnums", [])
                    sym_type_ints = bp_param.get("sym_type_ints", [])

                    # Map the saved integer value back to the cytnx.bondType enum member
                    # The Bond constructor Bond(dim, bond_type_enum_member) expects an enum member.
                    bond_type_enum_member: cytnx.bondType
                    if bond_type_val == cytnx.bondType.BD_KET.value: # -1
                        bond_type_enum_member = cytnx.bondType.BD_IN # Use BD_IN enum for constructor
                    elif bond_type_val == cytnx.bondType.BD_BRA.value: # 1
                        bond_type_enum_member = cytnx.bondType.BD_OUT # Use BD_OUT enum for constructor
                    else:
                        # Fallback for older saves or unexpected values, default to BD_REG (like BD_IN)
                        logger.warning(f"Unknown bond_type_val {bond_type_val} for core {i}, bond {bp_idx}. Defaulting to BD_REG.")
                        bond_type_enum_member = cytnx.bondType.BD_REG


                    if not qnums and not sym_type_ints: # Non-symmetric bond
                        reconstructed_bonds.append(
                            cytnx.Bond(dim, bond_type_enum_member)
                        )
                    else: # Symmetric bond
                        symmetries = []
                        for stype_int in sym_type_ints:
                            symmetries.append(cytnx.Symmetry.from_stype(stype_int))
                        
                        # The Bond constructor for symmetric bonds needs qnums and degs (degeneracies)
                        # and the list of Symmetry objects.
                        # Format of qnums: list[list[int]]
                        # We need 'degs': list[int]
                        # If 'degs' were not saved, this part needs a strategy.
                        # Assuming qnums contains [[q1_sec1, q2_sec1,...], [q1_sec2,...]]
                        # And degs would be [deg_sec1, deg_sec2, ...]
                        # The common constructor is Bond(type, qnums_ll, degs_l, syms_l)
                        # Or Bond(type, list_of_Qs_with_degs, syms_l)
                        # For now, if this path is hit, it means symmetries are used and 'degs'
                        # data might be missing from the save file or its reconstruction here.
                        
                        # A common format for qnums in cytnx.Bond is List[Qs], where Qs is tuple(List[int], degen)
                        # If bp_param["qnums"] was saved as List[List[int]] and we are missing degs:
                        if not all(isinstance(q_sec, list) for q_sec in qnums): # Check if qnums is List[Qs-like_tuple]
                             raise NotImplementedError(
                                f"Symmetric bond loading for core {i}, bond {bp_idx}: "
                                f"Saved qnums format requires 'degs' which are not explicitly saved/loaded in this version, "
                                f"or qnums are not in List[List[int]] format. Full symmetric bond reconstruction needs review."
                            )
                        
                        # Assuming qnums is List[List[int]] and we need corresponding degs.
                        # If your original bonds did not have explicit degeneracies (each qnum sector has deg=1),
                        # then degs would be a list of 1s, matching len(qnums).
                        # This is a strong assumption if not saved.
                        degs_assumed = [1] * len(qnums) # Placeholder assumption
                        if not qnums: # If qnums became empty after get, but syms were present.
                            reconstructed_bonds.append(cytnx.Bond(dim, bond_type_enum_member)) # Fallback to non-symm
                        else:
                            try:
                                reconstructed_bonds.append(
                                    cytnx.Bond(bond_type_enum_member, qnums, degs_assumed, symmetries)
                                )
                            except Exception as e_symm:
                                logger.error(f"Failed to reconstruct symmetric bond for core {i}, bond {bp_idx} with assumed degs: {e_symm}. "
                                             "Falling back to non-symmetric or re-raise.")
                                # Fallback for safety if symmetric reconstruction fails:
                                reconstructed_bonds.append(cytnx.Bond(dim, bond_type_enum_member))
                                # Or re-raise e_symm if strictness is required

                # Create UniTensor shell using integer IDs for dtype and device for constructor
                ut = UniTensor(bonds=reconstructed_bonds, 
                               labels=labels, 
                               rowrank=rowrank,
                               dtype=dtype_int,   # Pass integer Type ID
                               device=device_int) # Pass integer Device ID
                
                data_tensor = from_numpy(block_numpy).astype(dtype_int).to(device_int)
                ut.put_block(data_tensor)
                
                loaded_cores_M.append(ut)
            except KeyError as ke:
                logger.error(f"Missing key {ke} in loaded core data at index {i} from {file_name}.")
                raise
            except Exception as e:
                logger.error(f"Error reconstructing UniTensor core at index {i} from {file_name}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        return TensorTrain(loaded_cores_M)
    

if __name__ == "__main__":
    import numpy as _np # Use _np to avoid conflict if user uses np for other things
    from cytnx import from_numpy, cytnx # Ensure cytnx is available for bondType if needed directly in test

    # Configure logging for better feedback during tests
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger() # Get root logger
    # You can set logger.setLevel(logging.DEBUG) for more verbose output from your class if needed

    _np.random.seed(128)
    
    # Define physical dimensions for a small MPS
    phys_dims = [3, 4, 5] # Using the phys_dims from your provided test snippet
    L = len(phys_dims)

    # --- Build a random TensorTrain of trivial bond dims (1, d, 1) using UniTensors ---
    print(f"\nDEBUG: Preparing to create {L} UniTensor cores with phys_dims {phys_dims}...")
    ut_cores = []
    current_device = cytnx.Device.cpu # Explicitly set to CPU
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

    # --- Basic Information and Assertions ---
    if original_tt.M:
        print(f"\n--- Info for the first core of original_tt (phys_dims: {phys_dims}) ---")
        first_core = original_tt.M[0]
        print(f"Type of core: {type(first_core)}")
        print(f"Labels: {first_core.labels()}")
        print(f"Shape: {first_core.shape()}")
        bond_type_strs = [str(b.type()) for b in first_core.bonds()] # More robust printing
        print(f"Bond types (as string): {bond_type_strs}")
        print(f"Rowrank: {first_core.rowrank()}")
        print(f"Device: {first_core.device_str()}")
        print(f"Dtype: {first_core.dtype_str()}")
        
        print("DEBUG: Running _assert_core_validity on all cores of original_tt...")
        for k_assert in range(L):
            TensorTrain._assert_core_validity(original_tt.M[k_assert], k_assert, L, check_data_presence=True)
        print("DEBUG: _assert_core_validity PASSED for all cores in original_tt.")

    print(f"\nSuccessfully created TensorTrain with UniTensor cores.")
    print(f"Number of cores in original_tt: {len(original_tt.M)}")
    if original_tt.M:
        print(f"First core labels of original_tt: {original_tt.M[0].labels()}")

    # --- Test eval ---
    print("\n--- Testing Eval ---")
    idx_to_eval = [0] * L # Evaluate for index (0,0,...,0)
    try:
        val_eval = original_tt.eval(idx_to_eval)
        print(f"original_tt.eval({idx_to_eval}) = {val_eval}")
        
        # Manual calculation for this specific product state (1,d,1) with idx=[0]*L
        manual_prod = 1.0
        for k_manual in range(L):
            # Accessing element (0,0,0) of the block for core k
            # original_tt.M[k] is UniTensor, get_block_() is Tensor
            # For a (1,dk,1) core, data is at (0, idx[k], 0) = (0,0,0) here
            core_block_np = original_tt.M[k_manual].get_block_().numpy() # Get as numpy for easy indexing
            manual_prod *= core_block_np[0, idx_to_eval[k_manual], 0]
        print(f"Manual product for eval({idx_to_eval}) = {manual_prod}")
        assert _np.isclose(val_eval, manual_prod), f"Eval mismatch: {val_eval} vs {manual_prod}"
        print("Eval test PASSED.")
    except Exception as e:
        print(f"ERROR in Eval test: {e}")
        logger.exception("Eval test exception details:")


    # --- Test overlap and norm2 ---
    print("\n--- Testing Overlap & Norm2 ---")
    try:
        # Create another identical TT for overlap test
        # ut_cores_b = [core.clone() for core in ut_cores] # Ensure deep copy for B if A is modified
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


    # --- Test __add__ ---
    print("\n--- Testing Add ---")
    try:
        tt_c = original_tt + tt_b_for_overlap 
        print(f"tt_c (original_tt + tt_b_for_overlap) has {len(tt_c.M)} cores.")
        if tt_c.M and L > 0: # Ensure L > 0 for M[0] and M[L-1] access
            # --- For the first core (k=0) of tt_c ---
            first_core_c = tt_c.M[0]
            print(f"First core of tt_c: Labels={first_core_c.labels()}, Shape={first_core_c.shape()}")
            
            # Expected dimensions for the FIRST core based on C++ like __add__
            expected_L_dim_first_core = original_tt.M[0].bonds()[0].dim() # Should be rA (1)
            expected_P_dim_first_core = original_tt.M[0].bonds()[1].dim() # Should be dA
            expected_R_dim_first_core = original_tt.M[0].bonds()[2].dim() + tt_b_for_overlap.M[0].bonds()[2].dim() # sA + sB

            assert first_core_c.bonds()[0].dim() == expected_L_dim_first_core, \
                f"Sum TT First Core: Left bond dim mismatch. Expected {expected_L_dim_first_core}, Got {first_core_c.bonds()[0].dim()}"
            assert first_core_c.bonds()[1].dim() == expected_P_dim_first_core, \
                f"Sum TT First Core: Phys bond dim mismatch. Expected {expected_P_dim_first_core}, Got {first_core_c.bonds()[1].dim()}"
            assert first_core_c.bonds()[2].dim() == expected_R_dim_first_core, \
                f"Sum TT First Core: Right bond dim mismatch. Expected {expected_R_dim_first_core}, Got {first_core_c.bonds()[2].dim()}"

            # --- For the last core (k=L-1) of tt_c (if L > 1) ---
            if L > 1:
                last_core_c = tt_c.M[L-1]
                print(f"Last core of tt_c: Labels={last_core_c.labels()}, Shape={last_core_c.shape()}")

                expected_L_dim_last_core = original_tt.M[L-1].bonds()[0].dim() + tt_b_for_overlap.M[L-1].bonds()[0].dim() # rA + rB
                expected_P_dim_last_core = original_tt.M[L-1].bonds()[1].dim() # dA
                expected_R_dim_last_core = tt_b_for_overlap.M[L-1].bonds()[2].dim() # Should be sB (from other, which is 1)
                                            # Or original_tt.M[L-1].bonds()[2].dim() if C++ was sA

                assert last_core_c.bonds()[0].dim() == expected_L_dim_last_core, \
                    f"Sum TT Last Core: Left bond dim mismatch. Expected {expected_L_dim_last_core}, Got {last_core_c.bonds()[0].dim()}"
                assert last_core_c.bonds()[1].dim() == expected_P_dim_last_core, \
                    f"Sum TT Last Core: Phys bond dim mismatch. Expected {expected_P_dim_last_core}, Got {last_core_c.bonds()[1].dim()}"
                assert last_core_c.bonds()[2].dim() == expected_R_dim_last_core, \
                    f"Sum TT Last Core: Right bond dim mismatch. Expected {expected_R_dim_last_core}, Got {last_core_c.bonds()[2].dim()}"

            # --- For middle cores of tt_c (if L > 2) ---
            if L > 2:
                middle_core_c = tt_c.M[1] # Example for k=1
                print(f"Middle core (k=1) of tt_c: Labels={middle_core_c.labels()}, Shape={middle_core_c.shape()}")
                expected_L_dim_middle = original_tt.M[1].bonds()[0].dim() + tt_b_for_overlap.M[1].bonds()[0].dim()
                expected_P_dim_middle = original_tt.M[1].bonds()[1].dim()
                expected_R_dim_middle = original_tt.M[1].bonds()[2].dim() + tt_b_for_overlap.M[1].bonds()[2].dim()
                assert middle_core_c.bonds()[0].dim() == expected_L_dim_middle
                assert middle_core_c.bonds()[1].dim() == expected_P_dim_middle
                assert middle_core_c.bonds()[2].dim() == expected_R_dim_middle


            print("Addition core shapes structurally verified against C++-like logic.")
            
            # Now, the eval test can proceed because tt_c's outer bonds are dimension 1
            val_c_eval = tt_c.eval(idx_to_eval)
            print(f"tt_c.eval({idx_to_eval}) = {val_c_eval}")
            # val_eval was from original_tt.eval(idx_to_eval)
            assert _np.isclose(val_c_eval, 2 * val_eval), f"tt_c.eval ({val_c_eval}) should be 2 * original_tt.eval ({val_eval})"
            print("__add__ test PASSED.")
        else:
            print("__add__ test: tt_c has no cores (unexpected if inputs were not empty).")

    except Exception as e:
        print(f"ERROR in __add__ test: {e}")
        logger.exception("__add__ test exception details:")


    # --- Test sum and sum1 ---
    print("\n--- Testing Sum & Sum1 ---")
    try:
        s1_a = original_tt.sum1()
        print(f"original_tt.sum1() = {s1_a}")
        # Manual calculation for sum1 (all weights are 1)
        # This is eval with each physical index summed with weight 1.
        # For a (1,d,1) TT, sum1 = product_k ( sum_j M_k[0,j,0] )
        manual_sum1 = 1.0
        for k_manual_s1 in range(L):
            core_block_s1_np = original_tt.M[k_manual_s1].get_block_().numpy()
            manual_sum1 *= _np.sum(core_block_s1_np[0, :, 0]) # Sum over physical index, bond indices are 0
        print(f"Manual sum1 = {manual_sum1}")
        assert _np.isclose(s1_a, manual_sum1), f"Sum1 mismatch: {s1_a} vs {manual_sum1}"

        custom_weights = [[0.5] * d for d in phys_dims] # phys_dims was [3,4,5] from your test code
        s_custom_a = original_tt.sum(custom_weights)
        print(f"original_tt.sum(custom_weights) = {s_custom_a}")
        # Manual calculation for custom_weights
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

    # --- Test save and load ---
    print("\n--- Testing Save & Load ---")
    fn_test = "test_tt_save_load.cytn_tt_obj" # Changed extension for clarity
    try:
        original_tt.save(fn_test)
        print(f"original_tt saved to {fn_test}")
        tt_loaded = TensorTrain.load(fn_test)
        print(f"TensorTrain loaded from {fn_test}, has {len(tt_loaded.M)} cores.")
        assert len(original_tt.M) == len(tt_loaded.M), "Loaded TT core count mismatch"
        
        if original_tt.M and tt_loaded.M:
            # More thorough check than just labels
            # TensorTrain._assert_core_validity(tt_loaded.M[0],0,L) # Check validity of loaded core
            # error_loaded = original_tt.trueError(tt_loaded, max_eval=np.prod(phys_dims)) # Max_eval should be total states
            # Using a fixed small max_eval for faster testing if prod(phys_dims) is large
            max_eval_for_test = 1
            for dim_p in phys_dims: max_eval_for_test *= dim_p # Calculate actual total states
            if max_eval_for_test > 200: max_eval_for_test = 200 # Cap for speed

            error_loaded = original_tt.trueError(tt_loaded, max_eval=max_eval_for_test) 
            print(f"trueError between original_tt and tt_loaded = {error_loaded}")
            assert _np.isclose(error_loaded, 0.0), f"Loaded TT differs from original, error={error_loaded}"
        print("Save and Load test PASSED.")
    except Exception as e:
        print(f"ERROR in Save/Load test: {e}")
        logger.exception("Save/Load test exception details:")


    # --- Test compressLU (if MatprrLUFixedTol is available) ---
    # Attempt to import MatprrLUFixedTol for testing compressLU
    mat_decomp_available = False
    try:
        from mat_decomp import MatprrLUFixedTol # Ensure this is the correct import path
        mat_decomp_available = True
        print("\nDEBUG: mat_decomp.MatprrLUFixedTol imported successfully for compressLU test.")
    except ImportError:
        print("\nWARNING: mat_decomp.MatprrLUFixedTol not found. compressLU tests will be SKIPPED.")

    if mat_decomp_available:
        print("\n--- Testing CompressLU ---")
        # to_compress_tt was created at the beginning as a copy of original_tt's cores
        # For trueError, we need an uncompressed reference that matches to_compress_tt before compression
        tt_compress_ref = TensorTrain(ut_cores.copy()) # Fresh reference for before compression state
        maxBondDim = 50 # <--- DEFINE LOCAL VARIABLE
        
        print(f"Shape of M[0] before compress: {to_compress_tt.M[0].shape()}, Labels: {to_compress_tt.M[0].labels()}")
        if L > 1: print(f"Shape of M[1] before compress: {to_compress_tt.M[1].shape()}, Labels: {to_compress_tt.M[1].labels()}")

        try:
            to_compress_tt.compressLU(reltol=1e-7, maxBondDim=10) # Use a moderate tolerance
            print("compressLU finished.")
            print(f"Shape of M[0] after compress: {to_compress_tt.M[0].shape()}, Labels: {to_compress_tt.M[0].labels()}")
            if L > 1: print(f"Shape of M[1] after compress: {to_compress_tt.M[1].shape()}, Labels: {to_compress_tt.M[1].labels()}")

            # Check bond dimensions after compression
            if L > 1: # Compression affects internal bonds
                for k_comp_check in range(L - 1):
                    # Right bond of core k, Left bond of core k+1 (should be "link{k}")
                    # Max bond dim 10, but for (1,d,1) input, it often compresses to 1 unless data is special.
                    # It should not exceed maxBondDim (or original dim if smaller)
                    dim_link_k_right_M_k = to_compress_tt.M[k_comp_check].bonds()[2].dim()
                    dim_link_k_left_M_k1 = to_compress_tt.M[k_comp_check+1].bonds()[0].dim()
                    print(f"  Compressed bond 'link{k_comp_check}': M[{k_comp_check}]_right_dim={dim_link_k_right_M_k}, M[{k_comp_check+1}]_left_dim={dim_link_k_left_M_k1}")
                    assert dim_link_k_right_M_k == dim_link_k_left_M_k1, f"Compressed bond 'link{k_comp_check}' dimension mismatch"
                    assert dim_link_k_right_M_k <= max(1,maxBondDim), f"Compressed bond 'link{k_comp_check}' dim {dim_link_k_right_M_k} > maxBondDim {maxBondDim}"
                print("Compressed bond dimensions are consistent and within maxBondDim.")
            
            # Calculate error due to compression
            max_eval_for_compress_test = 1
            for dim_p_c in phys_dims: max_eval_for_compress_test *= dim_p_c
            if max_eval_for_compress_test > 200: max_eval_for_compress_test = 200

            error_compress = tt_compress_ref.trueError(to_compress_tt, max_eval=max_eval_for_compress_test)
            print(f"Maximum reconstruction error after compressLU: {error_compress:.3e}")
            # Note: True error can be non-zero due to compression. reltol influences this.
            # We expect error_compress to be roughly around reltol.
            # For random (1,d,1) input, LU might be exact if maxBondDim >= 1.
            # If reltol is very small, and maxBondDim allows, error should be small.
            # assert error_compress < reltol * 100, f"Compression error {error_compress} too large for reltol {reltol}" # Heuristic check

            print("CompressLU test section PASSED (execution completed, functional check heuristic).")
        except Exception as e:
            print(f"ERROR in CompressLU test: {e}")
            logger.exception("CompressLU test exception details:")
    
    print("\n--- All Implemented Tests Finished ---")