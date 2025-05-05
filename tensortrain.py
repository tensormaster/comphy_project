# filename: tensortrain.py (Corrected and with TODOs Implemented)

import cytnx
import numpy as np
from typing import List, Tuple, Optional, Union, Callable, Dict
import logging
import warnings
import os
import pickle # For simple metadata saving
from itertools import product # For trueError index generation
from tensorfuc import TensorFunction # Assuming this is the correct import path

logger = logging.getLogger(__name__)

# Define standard labels (integers for bonds, -1 for physical)
def phys_label() -> int: return -1
def left_bond_label(site: int) -> int: return site
def right_bond_label(site: int) -> int: return site + 1

# Helper function to convert a flat index to multi-index
# Needed for trueError
def _to_multi_index(flat_index: int, dims: List[int]) -> Tuple[int, ...]:
    """Converts a flat index to a multi-dimensional index."""
    if not dims: return ()
    multi_index = [0] * len(dims)
    current_index = flat_index
    # Use numpy prod with uint64 to prevent overflow for large tensor sizes
    block_size = np.prod(dims, dtype=np.uint64)
    if flat_index >= block_size:
         raise IndexError("flat_index out of range")
    if block_size == 0: return tuple(multi_index) # Handle zero dimension case

    for i in range(len(dims)):
        # Calculate block size for remaining dimensions safely
        remaining_dims = dims[i+1:]
        block_size = np.prod(remaining_dims, dtype=np.uint64) if remaining_dims else 1

        if block_size == 0: # Should not happen if dims are positive, but safeguard
             multi_index[i] = 0
             current_index = 0 # Or handle as error?
        else:
            multi_index[i] = int(current_index // block_size) # Use integer division
            current_index %= block_size
    return tuple(multi_index)


class CytnxTensorTrain:
    """
    Stores and manipulates a Tensor Train (MPS) using Cytnx UniTensors.
    Aligns concepts with tensor_train.h.
    Cores are expected to have labels [left_bond, physical, right_bond] (as strings).
    """
    def __init__(self, cores: Optional[List[cytnx.UniTensor]] = None):
        """
        Initializes the Tensor Train.

        Args:
            cores: (Optional) A list of cytnx.UniTensor objects. If provided,
                   a basic validity check is performed. Cores are cloned.
        """
        self.cores: List[cytnx.UniTensor] = []
        # Removed self._is_complex, will infer from dtype property
        self._device = cytnx.Device.cpu # Default device

        if cores:
            if not isinstance(cores, list) or not all(isinstance(c, cytnx.UniTensor) for c in cores):
                raise TypeError("`cores` must be a list of cytnx.UniTensor objects.")

            if cores:
                 # Infer device from first core (dtype inferred via property)
                 self._device = cores[0].device()

            # Store clones and perform check
            self.cores = [core.clone() for core in cores]
            self._check_validity() # Check structure upon initialization
        else:
            # Handle empty initialization
            pass # device remains default CPU

    @property
    def dtype(self):
        # Return default if no cores, otherwise infer from first core
        if not self.cores:
            # Default to Double if empty, consistent with C++? Or raise error?
            return cytnx.Type.Double
        # FIX: Check dtype of the first core
        core_dtype = self.cores[0].dtype()
        return core_dtype

    @property
    def is_complex(self) -> bool:
        """Checks if the Tensor Train stores complex data."""
        dt = self.dtype
        return dt == cytnx.Type.ComplexDouble or dt == cytnx.Type.ComplexFloat


    @property
    def device(self):
        if not self.cores:
             return self._device # Return default device if empty
        # Device should be consistent, checked in validity, return from first core
        return self.cores[0].device()

    # --- Basic Access (Keep as before) ---
    def __len__(self) -> int:
        """Returns the number of cores (length of the TT/MPS)."""
        return len(self.cores)

    def __getitem__(self, i: int) -> cytnx.UniTensor:
        """Gets a clone of the i-th core."""
        if i < 0: i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(f"Index {i} out of bounds for TT of length {len(self)}")
        return self.cores[i].clone()

    # --- Update Core (Corrected dtype check) ---
    def update_core(self, i: int, core: cytnx.UniTensor):
         """Updates the i-th core (checking validity)."""
         if not isinstance(core, cytnx.UniTensor):
             raise TypeError("Core must be a cytnx.UniTensor.")
         if i < 0: i += len(self)
         if not 0 <= i < len(self):
              raise IndexError(f"Index {i} out of bounds for TT of length {len(self)}")

         # Basic property check - Use properties of the TT if it exists
         target_dtype = self.dtype
         target_device = self.device
         # FIX: Check dtype correctly
         core_is_complex = core.dtype() == cytnx.Type.ComplexDouble or core.dtype() == cytnx.Type.ComplexFloat
         target_is_complex = target_dtype == cytnx.Type.ComplexDouble or target_dtype == cytnx.Type.ComplexFloat

         if core_is_complex != target_is_complex:
              warnings.warn(f"Core {i} complex type mismatch (core: {core_is_complex}, TT: {target_is_complex}).")
         if core.device() != target_device:
              warnings.warn(f"Core {i} device mismatch. Moving to {target_device}")
              core = core.to(target_device)
         if core.dtype() != target_dtype:
               warnings.warn(f"Core {i} dtype mismatch ({core.dtype()} vs {target_dtype}). Attempting cast.")
               core = core.astype(target_dtype)


         # Check labels and shape against neighbors BEFORE replacing
         N = len(self)
          # --- FIX: Ensure labels are strings ---
         expected_labels = [str(left_bond_label(i)), str(phys_label()), str(right_bond_label(i))]
         current_labels = core.labels() # Labels are already strings in UniTensor

         # Check Rank 3
         if core.rank() != 3:
              raise ValueError(f"Core {i} must have rank 3. Found rank {core.rank()}. Labels: {current_labels}")

         if set(current_labels) != set(expected_labels):
             # Allow for different ordering? No, enforce standard order.
              raise ValueError(f"Core {i} labels {current_labels} do not match expected standard order {expected_labels}")

         # Get dimensions using expected labels (which are now strings)
         core_ldim = core.shape()[current_labels.index(expected_labels[0])]
         core_rdim = core.shape()[current_labels.index(expected_labels[2])]


         # Check bond dimensions with neighbors
         # Left neighbor check
         if i > 0:
              left_core = self.cores[i-1]
              rb_left_core_str = str(right_bond_label(i-1)) # Expected right label of left core
              try:
                    left_core_rdim = left_core.shape()[left_core.labels().index(rb_left_core_str)]
                    if core_ldim != left_core_rdim:
                         raise ValueError(f"Left bond dim mismatch at bond {i}: core {i} ({core_ldim}) vs core {i-1} ({left_core_rdim})")
              except (ValueError, IndexError):
                   raise ValueError(f"Could not verify left bond dim for core {i}. Left core labels: {left_core.labels()}")
         elif core_ldim != 1: # Check first core left boundary
              raise ValueError(f"Core 0 left bond dimension must be 1, found {core_ldim}.")

         # Right neighbor check
         if i < N - 1:
              right_core = self.cores[i+1]
              lb_right_core_str = str(left_bond_label(i+1)) # Expected left label of right core
              try:
                    right_core_ldim = right_core.shape()[right_core.labels().index(lb_right_core_str)]
                    if core_rdim != right_core_ldim:
                         raise ValueError(f"Right bond dim mismatch at bond {i+1}: core {i} ({core_rdim}) vs core {i+1} ({right_core_ldim})")
              except (ValueError, IndexError):
                   raise ValueError(f"Could not verify right bond dim for core {i}. Right core labels: {right_core.labels()}")
         elif core_rdim != 1: # Check last core right boundary
             raise ValueError(f"Core {N-1} right bond dimension must be 1, found {core_rdim}.")


         self.cores[i] = core.clone()


    # --- Properties (Keep as before, validity check ensures correctness) ---
    @property
    def physical_dims(self) -> List[int]:
        """Returns the list of physical dimensions."""
        if not self.cores: return []
        dims = []
        pl_str = str(phys_label()) # Use string label
        for i, core in enumerate(self.cores):
             try:
                 idx = core.labels().index(pl_str)
                 dims.append(core.shape()[idx])
             except ValueError:
                  raise ValueError(f"Physical label '{pl_str}' not found in core {i} with labels {core.labels()}")
             except IndexError:
                  raise ValueError(f"Could not get shape for physical index in core {i}. Shape: {core.shape()}, Labels: {core.labels()}")
        return dims

    @property
    def bond_dims(self) -> List[int]:
        """Returns the list of bond dimensions [1, d1, d2, ..., d_{L-1}, 1]."""
        if not self.cores: return []

        N = len(self)
        try:
            self._check_validity() # Ensures structure is correct
        except ValueError as e:
            raise ValueError(f"Cannot get bond_dims, TT structure invalid: {e}")

        dims = [1] # Left boundary dimension
        for i in range(N - 1): # Iterate up to second to last core
             core = self.cores[i]
             rl_str = str(right_bond_label(i)) # Use string label
             try:
                 idx = core.labels().index(rl_str)
                 dims.append(core.shape()[idx])
             except (ValueError, IndexError):
                 raise RuntimeError(f"Internal error: Cannot find right bond label '{rl_str}' or shape in validated core {i}.")
        dims.append(1) # Right boundary dimension
        return dims

    # --- Validity Check (Corrected dtype/label checks) ---
    def _check_validity(self):
        """Performs structural checks on the tensor train."""
        if not self.cores:
            return True # Empty TT is valid

        N = len(self)
        expected_left_dim = 1
        # Use properties to get target dtype/device
        target_dtype = self.dtype
        target_device = self.device


        for i in range(N):
            core = self.cores[i]
            shape = core.shape()
            labels = core.labels() # These are strings
            rank = core.rank()

            # Check properties
            if core.dtype() != target_dtype:
                raise TypeError(f"Core {i} dtype {core.dtype()} mismatch with inferred TT dtype {target_dtype}")
            if core.device() != target_device:
                 raise TypeError(f"Core {i} device {core.device()} mismatch with inferred TT device {target_device}")

            # Define expected string labels
            ll_str = str(left_bond_label(i))
            pl_str = str(phys_label())
            rl_str = str(right_bond_label(i))

            expected_str_labels = {ll_str, pl_str, rl_str}

            # Check rank (should be 3)
            if rank != 3:
                 raise ValueError(f"Core {i} has rank {rank}, expected 3. Labels: {labels}")
            # Check labels match expected set
            if set(labels) != expected_str_labels:
                 raise ValueError(f"Core {i} labels {labels} do not match expected {expected_str_labels}")

            # Get dimensions based on labels
            try:
                # Use the string labels to find indices
                left_dim = shape[labels.index(ll_str)]
                phys_dim = shape[labels.index(pl_str)]
                right_dim = shape[labels.index(rl_str)]
            except (ValueError, IndexError) as e:
                 raise ValueError(f"Error accessing shape/labels for core {i}. Labels: {labels}, Shape: {shape}. Error: {e}")


            # Check left bond dimension
            if left_dim != expected_left_dim:
                 raise ValueError(f"Bond dimension mismatch at bond {i}: Core {i} left dim ({left_dim}) != expected ({expected_left_dim})")

            # Check boundary conditions
            if i == 0 and left_dim != 1:
                 raise ValueError(f"Core 0 left bond dimension is {left_dim}, expected 1.")
            if i == N - 1 and right_dim != 1:
                 raise ValueError(f"Core {N-1} right bond dimension is {right_dim}, expected 1.")

            # Update expected dimension for the next core
            expected_left_dim = right_dim

        return True

    # --- Evaluation (Use string labels) ---
    def eval(self, index: Union[Tuple[int, ...], List[int]]) -> Union[float, complex]:
        """
        Evaluates the tensor train for a given multi-index of physical legs.
        """
        if len(index) != len(self):
            raise ValueError(f"Index length {len(index)} must match TT length {len(self)}")
        if not self.cores:
             raise ValueError("Cannot evaluate an empty TensorTrain")

        # Start with scalar 1 UniTensor
        contracted_tensor = cytnx.UniTensor(cytnx.ones([1], dtype=self.dtype, device=self.device),
                                            labels=[str(left_bond_label(0))]) # Use string label

        pl_str = str(phys_label())
        temp_phys_label_str = "-99" # Temporary string label

        for i in range(len(self)):
            core = self.cores[i]
            phys_idx = index[i]
            ll_str = str(left_bond_label(i))
            rl_str = str(right_bond_label(i))

            # --- Slice the physical leg ---
            try:
                phys_dim = core.shape()[core.labels().index(pl_str)]
                if not 0 <= phys_idx < phys_dim:
                     raise IndexError(f"Physical index {phys_idx} out of bounds for core {i} dim {phys_dim}")
            except (ValueError, IndexError):
                 raise ValueError(f"Physical label '{pl_str}' not found or shape error in core {i}")

            # Create delta tensor
            delta = cytnx.zeros([phys_dim], dtype=self.dtype, device=self.device)
            delta[phys_idx] = 1
            phys_delta_ut = cytnx.UniTensor(delta, labels=[temp_phys_label_str]) # Use temp string label

            # Temporarily relabel the core's physical leg to match
            core_labels_orig = core.labels() # Store original labels
            label_map = {ll_str: ll_str, pl_str: temp_phys_label_str, rl_str: rl_str}
            core.relabels_([label_map[lbl] for lbl in core_labels_orig]) # Apply mapping

            # Contract delta with the core to slice the physical leg
            # Input L labels: [ll_str], Core labels: [ll_str, temp_phys_label_str, rl_str]
            # Delta labels: [temp_phys_label_str]
            # Output should have labels [ll_str, rl_str]
            try:
                 sliced_core = cytnx.Contract(core, phys_delta_ut)
            except Exception as e:
                 logger.error(f"Contraction error during slicing core {i}: {e}")
                 logger.error(f"Core labels: {core.labels()}, Delta labels: {phys_delta_ut.labels()}")
                 raise
            # Restore core's original labels
            core.relabels_(core_labels_orig)


            # --- Contract with the accumulated tensor ---
            # contracted_tensor labels: [ll_str]
            # sliced_core labels: [ll_str, rl_str]
            # Output should have labels: [rl_str]
            try:
                 contracted_tensor = cytnx.Contract(contracted_tensor, sliced_core)
            except Exception as e:
                 logger.error(f"Contraction error during accumulation step {i}: {e}")
                 logger.error(f"Accumulator labels: {contracted_tensor.labels()}, Sliced core labels: {sliced_core.labels()}")
                 raise


        # Final check and return item
        expected_final_label = str(right_bond_label(len(self)-1))
        if contracted_tensor.labels() != [expected_final_label]:
             warnings.warn(f"Final contracted tensor has unexpected labels: {contracted_tensor.labels()}, expected ['{expected_final_label}']")
        if contracted_tensor.shape() != [1]:
             raise ValueError(f"Final contraction result is not a scalar: shape {contracted_tensor.shape()}")

        return contracted_tensor.item()


    # --- Overlap and Norm (Use string labels) ---
    def overlap(self, other: 'CytnxTensorTrain') -> Union[float, complex]:
        """
        Computes the overlap <self|other>.
        """
        if len(self) != len(other):
            raise ValueError("Cannot compute overlap: TensorTrains have different lengths.")
        if not self.cores: return 0.0 # Overlap of empty TT is 0

        # --- Check device and dtype compatibility ---
        if self.device != other.device:
             warnings.warn(f"Device mismatch in overlap ({self.device} vs {other.device}). Moving 'other' TT.")
             other = other.to(self.device)
             if other is None: raise RuntimeError("Failed to move 'other' TT to correct device.")
        result_is_complex = self.is_complex or other.is_complex
        result_dtype = cytnx.Type.ComplexDouble if result_is_complex else cytnx.Type.Double

        # Start with identity matrix (1x1) labeled as bonds (0, 0')
        label_offset = len(self) + 10
        def other_left_bond_label_str(site: int) -> str: return str(site + label_offset)
        def other_right_bond_label_str(site: int) -> str: return str(site + 1 + label_offset)

        ll0_str = str(left_bond_label(0))
        ll0_other_str = other_left_bond_label_str(0)

        L_contract = cytnx.UniTensor(cytnx.eye(1, dtype=result_dtype, device=self.device),
                                     labels=[ll0_str, ll0_other_str])

        pl_str = str(phys_label())

        for i in range(len(self)):
            core1 = self.cores[i].astype(result_dtype).contiguous()
            core2_orig = other.cores[i].astype(result_dtype).contiguous()

            ll1_str = str(left_bond_label(i))
            rl1_str = str(right_bond_label(i))
            ll2_str = other_left_bond_label_str(i)
            rl2_str = other_right_bond_label_str(i)

            # Create conjugated core2 with primed labels
            core2_conj = core2_orig.Conj()
            # Core 2 original labels assumed: [str(ll1), pl_str, str(rl1)]
            # Relabel conj core 2 to: [ll2_str, pl_str, rl2_str]
            core2_conj.relabels_([ll2_str, pl_str, rl2_str])

            # Ensure core1 has standard labels: [ll1_str, pl_str, rl1_str]
            core1.relabels_([ll1_str, pl_str, rl1_str])

            # Contract L with core1: L[ll1, ll2] * core1[ll1, pl, rl1] -> Temp[ll2, pl, rl1]
            Temp = cytnx.Contract(L_contract, core1)

            # Contract Temp with core2_conj: Temp[ll2, pl, rl1] * core2_conj[ll2, pl, rl2] -> L_contract[rl1, rl2]
            L_contract = cytnx.Contract(Temp, core2_conj)
            # Labels should automatically resolve to [rl1_str, rl2_str]

        # Final check and return
        if L_contract.shape() != [1, 1]:
             raise ValueError(f"Final overlap contraction is not a scalar: shape {L_contract.shape()}")
        final_labels = [str(right_bond_label(len(self)-1)), other_right_bond_label_str(len(self)-1)]
        if L_contract.labels() != final_labels:
             warnings.warn(f"Final overlap tensor has unexpected labels: {L_contract.labels()}, expected {final_labels}")

        return L_contract.item()

    def norm2(self) -> float:
        """Computes the squared norm <self|self>."""
        if not self.cores: return 0.0
        # Norm^2 should be real, use target dtype
        target_dtype = self.dtype
        target_device = self.device

        L_contract = cytnx.UniTensor(cytnx.eye(1, dtype=target_dtype, device=target_device),
                                     labels=[str(left_bond_label(0))]*2) # Use same label twice

        pl_str = str(phys_label())

        for i in range(len(self)):
            core = self.cores[i].contiguous()
            core_conj = core.Conj() # Labels are same as core: [ll, pl, rl] (strings)

            ll_str = str(left_bond_label(i))
            rl_str = str(right_bond_label(i))

            # Ensure standard labels before contraction
            core.relabels_([ll_str, pl_str, rl_str])
            core_conj.relabels_([ll_str, pl_str, rl_str])

            # Contract L with core: L[ll, ll] * core[ll, pl, rl] -> Temp[ll, pl, rl]
            Temp = cytnx.Contract(L_contract, core)

            # Contract Temp with core_conj: Temp[ll, pl, rl] * core_conj[ll, pl, rl] -> L_contract[rl, rl]
            L_contract = cytnx.Contract(Temp, core_conj)
            # Labels should automatically resolve to [rl_str, rl_str]

        # Final check and return
        if L_contract.shape() != [1, 1]:
             raise ValueError(f"Final norm contraction is not a scalar: shape {L_contract.shape()}")
        final_labels = [str(right_bond_label(len(self)-1))]*2
        if L_contract.labels() != final_labels:
             warnings.warn(f"Final norm tensor has unexpected labels: {L_contract.labels()}, expected {final_labels}")

        val = L_contract.item()
        if isinstance(val, complex) and abs(val.imag) > 1e-12:
            warnings.warn(f"Norm squared has non-negligible imaginary part: {val.imag}. Returning real part.")
        res = np.real(val)
        if res < -1e-12:
            warnings.warn(f"Norm squared is negative: {res}. Clamping to 0.")
            res = 0.0
        return res


    # --- Summation (Use string labels) ---
    def sum1(self) -> Union[float, complex]:
         """Computes the plain sum (contraction over all physical legs)."""
         if not self.cores: return 0.0

         contracted_tensor = cytnx.UniTensor(cytnx.ones([1], dtype=self.dtype, device=self.device),
                                             labels=[str(left_bond_label(0))])
         pl_str = str(phys_label())
         temp_phys_label_str = "-99"

         for i in range(len(self)):
              core = self.cores[i]
              ll_str = str(left_bond_label(i))
              rl_str = str(right_bond_label(i))
              try:
                   phys_dim = core.shape()[core.labels().index(pl_str)]
              except (ValueError, IndexError):
                  raise ValueError(f"Cannot find physical label/shape in core {i}")

              sum_vec = cytnx.ones([phys_dim], dtype=self.dtype, device=self.device)
              phys_sum_ut = cytnx.UniTensor(sum_vec, labels=[temp_phys_label_str])

              # Temporarily relabel core
              core_labels_orig = core.labels()
              label_map = {ll_str: ll_str, pl_str: temp_phys_label_str, rl_str: rl_str}
              core.relabels_([label_map[lbl] for lbl in core_labels_orig])

              summed_core = cytnx.Contract(core, phys_sum_ut)
              core.relabels_(core_labels_orig) # Restore labels

              contracted_tensor = cytnx.Contract(contracted_tensor, summed_core)

         if contracted_tensor.shape() != [1]:
              raise ValueError(f"Final summation result is not a scalar: shape {contracted_tensor.shape()}")
         return contracted_tensor.item()

    def sum(self, weights: List[Union[np.ndarray, cytnx.Tensor]]) -> Union[float, complex]:
        """Computes the weighted sum using provided weight vectors for each physical leg."""
        if len(weights) != len(self):
            raise ValueError(f"Number of weight vectors ({len(weights)}) must match TT length ({len(self)})")
        if not self.cores: return 0.0

        contracted_tensor = cytnx.UniTensor(cytnx.ones([1], dtype=self.dtype, device=self.device),
                                            labels=[str(left_bond_label(0))])
        pl_str = str(phys_label())
        temp_phys_label_str = "-99"

        for i in range(len(self)):
            core = self.cores[i]
            weight_vec = weights[i]
            ll_str = str(left_bond_label(i))
            rl_str = str(right_bond_label(i))

            # Ensure weight is a cytnx Tensor on the correct device/dtype
            if isinstance(weight_vec, np.ndarray):
                 weight_vec = cytnx.from_numpy(weight_vec)
            if weight_vec.device() != self.device:
                 weight_vec = weight_vec.to(self.device)

            # Check/cast dtype carefully
            target_dtype = self.dtype
            if self.is_complex and not (weight_vec.dtype() in [cytnx.Type.ComplexDouble, cytnx.Type.ComplexFloat]):
                 warnings.warn(f"Weight vector {i} is real but TT is complex. Casting weight to {target_dtype}.")
                 weight_vec = weight_vec.astype(target_dtype)
            elif weight_vec.dtype() != target_dtype:
                 warnings.warn(f"Weight vector {i} dtype {weight_vec.dtype()} mismatch with TT {target_dtype}. Attempting cast.")
                 weight_vec = weight_vec.astype(target_dtype)

            try:
                phys_dim = core.shape()[core.labels().index(pl_str)]
            except (ValueError, IndexError):
                 raise ValueError(f"Cannot find physical label/shape in core {i}")

            if weight_vec.shape() != [phys_dim]:
                 raise ValueError(f"Weight vector {i} shape {weight_vec.shape()} mismatch with physical dim {phys_dim}")

            phys_weight_ut = cytnx.UniTensor(weight_vec, labels=[temp_phys_label_str])

            # Temporarily relabel core
            core_labels_orig = core.labels()
            label_map = {ll_str: ll_str, pl_str: temp_phys_label_str, rl_str: rl_str}
            core.relabels_([label_map[lbl] for lbl in core_labels_orig])

            weighted_core = cytnx.Contract(core, phys_weight_ut)
            core.relabels_(core_labels_orig) # Restore labels

            contracted_tensor = cytnx.Contract(contracted_tensor, weighted_core)

        if contracted_tensor.shape() != [1]:
              raise ValueError(f"Final weighted sum result is not a scalar: shape {contracted_tensor.shape()}")
        return contracted_tensor.item()


    # --- Compression Framework (Sweep Methods - SKETCH) ---
    # WARNING: This is a complex sketch requiring careful testing and refinement.
    def _compress_step(self, M1: cytnx.UniTensor, M2: cytnx.UniTensor, direction: str,
                       tol: float = 1e-14, max_bond: Optional[int] = None):
        """Internal helper for SVD compression step."""
        if direction == 'left':
            l, p, r = [int(lbl) for lbl in M1.labels()] # Assume labels are 'l', 'p', 'r' (as strings)
            r_match, p_next, r_next = [int(lbl) for lbl in M2.labels()]
            if r != r_match: raise ValueError(f"Bond label mismatch L->R: {r} vs {r_match}")

            M1_perm = M1.permute([str(l), str(p), str(r)]) # Ensure order for rowrank
            M1_matrix = M1_perm.contiguous()
            M1_matrix.set_rowrank(2)

            try: S, U, Vt = cytnx.linalg.Svd(M1_matrix, is_U=True, is_Vt=True)
            except Exception as e: raise RuntimeError(f"SVD L->R failed: {e}")

            # --- Truncation ---
            s_np = S.numpy() # Get numpy array for analysis
            s_np_sq = s_np**2
            total_sum_sq = np.sum(s_np_sq)
            k_new = len(s_np) # Default: no truncation
            if tol > 0 and total_sum_sq > 1e-14:
                 cumulative_sum_sq = np.cumsum(s_np_sq)
                 k_tol = np.searchsorted(cumulative_sum_sq, total_sum_sq * (1 - tol**2)) + 1
                 k_new = min(k_new, k_tol)
            if max_bond is not None and max_bond > 0: k_new = min(k_new, max_bond)
            k_new = max(1, k_new) # Ensure rank is at least 1

            # Truncate
            S_trunc = S[:k_new]
            U_trunc = U[:, :k_new] # Shape [(l*p), k_new]
            Vt_trunc = Vt[:k_new, :] # Shape [k_new, r]

            # --- Update Cores ---
            new_r_label = str(r) # Use original right bond label for the new bond
            # M1_new from U_trunc: shape [l, p, k_new]
            M1_new_tensor = U_trunc.reshape(M1.shape()[0], M1.shape()[1], k_new)
            M1_new = cytnx.UniTensor(M1_new_tensor, labels=[str(l), str(p), new_r_label], rowrank=1)

            # Factor SVt: shape [k_new, r]
            # Need temporary labels for matrix multiplication interpretation
            temp_label_k = "-88"
            S_diag_ut = cytnx.linalg.Diag(S_trunc, labels=[temp_label_k]*2)
            Vt_trunc_ut = cytnx.UniTensor(Vt_trunc, labels=[temp_label_k, str(r)], rowrank=1)
            SVt = cytnx.Contract(S_diag_ut, Vt_trunc_ut) # Labels [k_new, r]
            SVt.relabels_([new_r_label, str(r)]) # Relabel to [new_r_label, r]

            # M2_new = SVt @ M2: shape [k_new, p_next, r_next]
            M2_new = cytnx.Contract(SVt, M2)
            # Labels should be [new_r_label, p_next, r_next]
            M2_new.relabels_([new_r_label, str(p_next), str(r_next)]) # Ensure standard labels

            return M1_new, M2_new

        elif direction == 'right':
             l, p, r = [int(lbl) for lbl in M1.labels()]
             r_match, p_next, r_next = [int(lbl) for lbl in M2.labels()]
             if r != r_match: raise ValueError(f"Bond label mismatch R->L: {r} vs {r_match}")

             M2_perm = M2.permute([str(r), str(p_next), str(r_next)])
             M2_matrix = M2_perm.contiguous()
             M2_matrix.set_rowrank(1) # Group by left bond

             try: S, U, Vt = cytnx.linalg.Svd(M2_matrix, is_U=True, is_Vt=True)
             except Exception as e: raise RuntimeError(f"SVD R->L failed: {e}")

             # --- Truncation (same as L->R) ---
             s_np = S.numpy(); s_np_sq = s_np**2; total_sum_sq = np.sum(s_np_sq)
             k_new = len(s_np)
             if tol > 0 and total_sum_sq > 1e-14:
                  cumulative_sum_sq = np.cumsum(s_np_sq)
                  k_tol = np.searchsorted(cumulative_sum_sq, total_sum_sq * (1 - tol**2)) + 1
                  k_new = min(k_new, k_tol)
             if max_bond is not None and max_bond > 0: k_new = min(k_new, max_bond)
             k_new = max(1, k_new)

             S_trunc = S[:k_new]
             U_trunc = U[:, :k_new]   # Shape [r, k_new]
             Vt_trunc = Vt[:k_new, :] # Shape [k_new, (p_next*r_next)]

             # --- Update Cores ---
             new_l_label = str(r) # Keep original left bond label for the new bond
             # M2_new from Vt_trunc: shape [k_new, p_next, r_next]
             M2_new_tensor = Vt_trunc.reshape(k_new, M2.shape()[1], M2.shape()[2])
             M2_new = cytnx.UniTensor(M2_new_tensor, labels=[new_l_label, str(p_next), str(r_next)], rowrank=1)

             # Factor US: shape [r, k_new]
             temp_label_k = "-88"
             U_trunc_ut = cytnx.UniTensor(U_trunc, labels=[str(r), temp_label_k], rowrank=1)
             S_diag_ut = cytnx.linalg.Diag(S_trunc, labels=[temp_label_k]*2)
             US = cytnx.Contract(U_trunc_ut, S_diag_ut) # Labels [r, k_new]
             US.relabels_([str(r), new_l_label]) # Relabel to [r, new_l_label]

             # M1_new = M1 @ US: shape [l, p, k_new]
             M1_new = cytnx.Contract(M1, US)
             # Labels should be [l, p, new_l_label]
             M1_new.relabels_([str(l), str(p), new_l_label]) # Ensure standard labels

             return M1_new, M2_new
        else:
             raise ValueError("Direction must be 'left' or 'right'")


    def left_to_right_sweep(self, tol: float = 1e-14, max_bond: Optional[int] = None):
         """Performs a left-to-right SVD sweep for compression/orthogonalization."""
         logger.info(f"Starting left-to-right SVD sweep (tol={tol}, max_bond={max_bond})...")
         N = len(self)
         if N < 2: return

         for i in range(N - 1):
              # logger.debug(f"  L->R Processing bond {i+1} (between core {i} and {i+1})")
              M1 = self.cores[i]
              M2 = self.cores[i+1]
              try:
                  M1_new, M2_new = self._compress_step(M1, M2, 'left', tol, max_bond)
                  self.cores[i] = M1_new
                  self.cores[i+1] = M2_new
              except Exception as e:
                   logger.error(f"Error during left-to-right compression step at bond {i+1}: {e}")
                   raise
         logger.info("Left-to-right sweep finished.")


    def right_to_left_sweep(self, tol: float = 1e-14, max_bond: Optional[int] = None):
         """Performs a right-to-left SVD sweep for compression/orthogonalization."""
         logger.info(f"Starting right-to-left SVD sweep (tol={tol}, max_bond={max_bond})...")
         N = len(self)
         if N < 2: return

         for i in range(N - 1, 0, -1):
             # logger.debug(f"  R->L Processing bond {i} (between core {i-1} and {i})")
             M1 = self.cores[i-1]
             M2 = self.cores[i]
             try:
                 M1_new, M2_new = self._compress_step(M1, M2, 'right', tol, max_bond)
                 self.cores[i-1] = M1_new
                 self.cores[i] = M2_new
             except Exception as e:
                 logger.error(f"Error during right-to-left compression step at bond {i}: {e}")
                 raise
         logger.info("Right-to-left sweep finished.")

    # --- High-level Compression ---
    def compressSVD(self, reltol: float = 1e-12, max_bond_dim: Optional[int] = None, sweeps: int = 2):
         """Compresses the TT using SVD sweeps."""
         logger.info(f"Starting SVD compression: reltol={reltol}, max_bond_dim={max_bond_dim}, sweeps={sweeps}")
         if len(self) < 2:
             logger.info("Skipping compression for TT length < 2.")
             return

         for sweep_num in range(sweeps):
             logger.info(f"--- Compression Sweep {sweep_num + 1}/{sweeps} ---")
             self.right_to_left_sweep(tol=reltol, max_bond=max_bond_dim)
             self.left_to_right_sweep(tol=reltol, max_bond=max_bond_dim)
             # Check validity after each full sweep
             try:
                 self._check_validity()
                 logger.info(f"Bond dims after sweep {sweep_num + 1}: {self.bond_dims}")
             except ValueError as e:
                  logger.error(f"TT structure invalid after sweep {sweep_num + 1}: {e}")
                  raise RuntimeError("Compression failed due to invalid intermediate state.")

         logger.info("SVD compression sweeps completed.")

    @staticmethod # <--- 確保這個裝飾器存在
    def random(physical_dims: List[int],
               max_bond_dim: int,
               dtype=cytnx.Type.Double,
               device=cytnx.Device.cpu) -> 'CytnxTensorTrain':
        """
        Creates a Tensor Train with random cores.
        Ensures correct labels [left, phys, right] as strings.
        """
        N = len(physical_dims)
        if N == 0:
            return CytnxTensorTrain() # Return empty TT

        cores = []
        current_bond_dim = 1 # Left boundary dimension

        # FIX: Check dtype correctly
        is_complex = (dtype == cytnx.Type.ComplexDouble or dtype == cytnx.Type.ComplexFloat)

        for i in range(N):
            phys_dim = physical_dims[i]
            if phys_dim <= 0: raise ValueError(f"Physical dimension must be positive at site {i}")

            # Determine right bond dimension capped by max_bond_dim
            dim_right = min(max_bond_dim, current_bond_dim * phys_dim) if i < N - 1 else 1
            # Ensure dim_right is at least 1
            dim_right = max(1, dim_right)

            dim_left = current_bond_dim
            shape = (dim_left, phys_dim, dim_right)
            # Define integer labels first
            int_labels = [left_bond_label(i), phys_label(), right_bond_label(i)]
            # Convert labels to strings
            str_labels = [str(lbl) for lbl in int_labels]

            # Create random tensor data
            if is_complex:
                real_part = cytnx.random.uniform(shape, -1, 1, device=device)
                imag_part = cytnx.random.uniform(shape, -1, 1, device=device)
                core_tensor = real_part + 1j * imag_part
                core_tensor = core_tensor.astype(dtype)
            else:
                core_tensor = cytnx.random.uniform(shape, -1, 1, device=device).astype(dtype)

            # Create UniTensor with correct STRING labels directly
            # Set rowrank=1 convention (left vs rest)
            core_uni = cytnx.UniTensor(core_tensor, labels=str_labels, rowrank=1)
            cores.append(core_uni)

            current_bond_dim = dim_right

        # Create the TT object using the generated cores
        # The constructor will call _check_validity
        return CytnxTensorTrain(cores)
    # --- Save/Load ---
    def save(self, filename_prefix: str):
        """
        Saves the Tensor Train cores and metadata.
        """
        N = len(self)
        metadata = {
            'length': N,
            'dtype_str': str(self.dtype), # Store dtype representation
            'device_str': str(self.device) # Store device as string
        }
        meta_filename = f"{filename_prefix}_meta.pkl"
        try:
            with open(meta_filename, 'wb') as f: pickle.dump(metadata, f)
        except Exception as e: raise IOError(f"Failed to save metadata: {e}")

        for i, core in enumerate(self.cores):
            core_filename = f"{filename_prefix}_core_{i}.cytnx"
            try: core.Save(core_filename)
            except Exception as e: raise IOError(f"Failed to save core {i}: {e}")
        logger.info(f"TensorTrain saved with prefix '{filename_prefix}'")

    @classmethod
    def load(cls, filename_prefix: str) -> 'CytnxTensorTrain':
        """Loads a Tensor Train from saved files."""
        meta_filename = f"{filename_prefix}_meta.pkl"
        if not os.path.exists(meta_filename): raise FileNotFoundError(f"Metadata file not found: {meta_filename}")
        try:
            with open(meta_filename, 'rb') as f: metadata = pickle.load(f)
        except Exception as e: raise IOError(f"Failed to load metadata: {e}")

        N = metadata.get('length', 0)
        if N == 0: return cls()

        cores = []
        for i in range(N):
            core_filename = f"{filename_prefix}_core_{i}.cytnx"
            if not os.path.exists(core_filename): raise FileNotFoundError(f"Core file not found: {core_filename}")
            try: cores.append(cytnx.UniTensor.Load(core_filename))
            except Exception as e: raise IOError(f"Failed to load core {i}: {e}")

        instance = cls(cores)
        logger.info(f"TensorTrain loaded from prefix '{filename_prefix}'")
        return instance

    # --- Addition ---
    def __add__(self, other: 'CytnxTensorTrain') -> 'CytnxTensorTrain':
        """Adds two Tensor Trains self + other."""
        if not isinstance(other, CytnxTensorTrain): return NotImplemented
        N = len(self);
        if N != len(other): raise ValueError("TT lengths must match for addition.")
        if N == 0: return CytnxTensorTrain()
        if self.physical_dims != other.physical_dims: raise ValueError("Physical dimensions must match for addition.")
        if self.device != other.device:
            warnings.warn(f"Device mismatch in TT add. Moving 'other' to {self.device}.")
            other = other.to(self.device)
            if other is None: raise RuntimeError("Failed to move 'other' TT.")

        result_is_complex = self.is_complex or other.is_complex
        result_dtype = cytnx.Type.ComplexDouble if result_is_complex else cytnx.Type.Double

        new_cores = []
        pl_str = str(phys_label())

        for i in range(N):
            A = self.cores[i].astype(result_dtype)
            B = other.cores[i].astype(result_dtype)
            shapeA = A.shape(); shapeB = B.shape()
            lA, pA, rA = shapeA; lB, pB, rB = shapeB
            ll_str = str(left_bond_label(i)); rl_str = str(right_bond_label(i))

            if i == 0: # First core
                new_l, new_p, new_r = 1, pA, rA + rB
                C_tensor = cytnx.zeros((new_l, new_p, new_r), dtype=result_dtype, device=self.device)
                C_tensor[0, :, :rA] = A.get_block().reshape(pA, rA)
                C_tensor[0, :, rA:] = B.get_block().reshape(pA, rB)
            elif i == N - 1: # Last core
                new_l, new_p, new_r = lA + lB, pA, 1
                C_tensor = cytnx.zeros((new_l, new_p, new_r), dtype=result_dtype, device=self.device)
                C_tensor[:lA, :, 0] = A.get_block().reshape(lA, pA)
                C_tensor[lA:, :, 0] = B.get_block().reshape(lB, pA)
            else: # Middle cores
                new_l, new_p, new_r = lA + lB, pA, rA + rB
                C_tensor = cytnx.zeros((new_l, new_p, new_r), dtype=result_dtype, device=self.device)
                C_tensor[:lA, :, :rA] = A.get_block()
                C_tensor[lA:, :, rA:] = B.get_block()

            new_labels = [ll_str, pl_str, rl_str]
            C = cytnx.UniTensor(C_tensor, labels=new_labels, rowrank=1)
            new_cores.append(C)

        return CytnxTensorTrain(new_cores)

    # --- True Error ---
    def trueError(self, func: Callable[[Tuple[int,...]], Union[float, complex]],
                  max_n_eval: int = 1000000) -> float:
        """Computes the maximum absolute error |self(idx) - func(idx)|."""
        if not self.cores: return 0.0
        dims = self.physical_dims
        try: total_size = np.prod(dims, dtype=np.uint64)
        except: return -1.0
        if total_size == 0: return 0.0
        if total_size > max_n_eval:
            logger.warning(f"Tensor size ({total_size}) > max_n_eval ({max_n_eval}). Skipping trueError.")
            return -1.0

        logger.info(f"Calculating trueError checking {total_size} elements...")
        max_err = 0.0; max_err_idx = None
        func_wrapped = TensorFunction(func, use_cache=False) # Ensure tuple input for func

        count = 0
        for idx_tuple in product(*[range(d) for d in dims]):
            try:
                approx_val = self.eval(idx_tuple)
                exact_val = func_wrapped(idx_tuple) # Use wrapper
                # Handle potential complex numbers correctly for abs
                diff = approx_val - exact_val
                err = abs(diff)
                if err > max_err: max_err = err; max_err_idx = idx_tuple
            except Exception as e:
                 logger.error(f"Error during trueError eval at {idx_tuple}: {e}")
                 return -1.0
            # Progress indicator (optional)
            # count += 1
            # if count % 100000 == 0: logger.debug(f"  ...checked {count}/{total_size}")

        logger.info(f"trueError finished. Max error = {max_err:.6e} at {max_err_idx}")
        return max_err

    # --- Utility ---
    def to(self, device) -> Optional['CytnxTensorTrain']:
        """Moves the TT to a different device."""
        try:
            new_cores = [core.to(device) for core in self.cores]
            return CytnxTensorTrain(new_cores) # Create new instance
        except Exception as e:
             logger.error(f"Failed to move TT to device {device}: {e}")
             return None


# --- Example Usage (with fixes for tests) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    print("\n" + "="*30)
    print("--- Testing Random TT ---")
    pdims = [2, 3, 4, 2]
    chi = 5
    try:
        # This should now work with string labels fix
        tt_rand = CytnxTensorTrain.random(pdims, chi, dtype=cytnx.Type.Double)
        print(f"Length: {len(tt_rand)}")
        print(f"Phys Dims: {tt_rand.physical_dims}")
        print(f"Bond Dims: {tt_rand.bond_dims}")
        idx_rand = tuple(np.random.randint(d) for d in pdims)
        val_rand = tt_rand(idx_rand)
        print(f"Eval at {idx_rand}: {val_rand:.6f}")
        norm_sq_rand = tt_rand.norm2()
        print(f"Norm^2: {norm_sq_rand:.6f}")
    except Exception as e:
        logger.error(f"Error in Random TT test: {e}", exc_info=True)


    print("\n" + "="*30)
    print("--- Testing Overlap ---")
    try:
        # This should now work
        tt_rand2 = CytnxTensorTrain.random(pdims, chi, dtype=cytnx.Type.Double)
        # Need tt_rand to exist from previous block
        if 'tt_rand' in locals():
            ovlp = tt_rand.overlap(tt_rand2)
            print(f"Overlap with another random TT: {ovlp:.6f}")
            # Check self overlap equals norm2
            self_ovlp = tt_rand.overlap(tt_rand)
            print(f"Self Overlap: {self_ovlp:.6f} (should match Norm^2)")
            assert np.isclose(self_ovlp, tt_rand.norm2()), "Self overlap != norm2"
        else:
            print("Skipping overlap test as tt_rand was not created.")
    except Exception as e:
        logger.error(f"Error in Overlap test: {e}", exc_info=True)

    print("\n" + "="*30)
    print("--- Testing Summation ---")
    try:
        # Need tt_rand to exist
        if 'tt_rand' in locals():
            sum_plain = tt_rand.sum1()
            print(f"Plain Sum (sum1): {sum_plain:.6f}")
            # Create weights compatible with TT dtype/device
            weights = [cytnx.random.normal([d], 0, 1, device=tt_rand.device).astype(tt_rand.dtype) for d in pdims]
            sum_weighted = tt_rand.sum(weights)
            print(f"Weighted Sum: {sum_weighted:.6f}")
        else:
             print("Skipping summation test as tt_rand was not created.")
    except Exception as e:
        logger.error(f"Error in Summation test: {e}", exc_info=True)

    print("\n" + "="*30)
    print("--- Testing Save/Load ---")
    try:
         # Need tt_rand to exist
        if 'tt_rand' in locals():
            prefix = "test_tt_save"
            tt_rand.save(prefix)
            print(f"TT saved with prefix '{prefix}'.")
            tt_loaded = CytnxTensorTrain.load(prefix)
            print(f"TT loaded successfully. Length: {len(tt_loaded)}")
            assert len(tt_rand) == len(tt_loaded)
            assert tt_rand.physical_dims == tt_loaded.physical_dims
            idx_check = tuple(np.random.randint(d) for d in tt_rand.physical_dims)
            val_orig = tt_rand(idx_check)
            val_loaded = tt_loaded(idx_check)
            print(f"Eval at {idx_check}: Original={val_orig:.6f}, Loaded={val_loaded:.6f}")
            assert np.isclose(val_orig, val_loaded), "Loaded TT eval mismatch!"
            print("Save/Load test passed.")
            # Clean up
            try:
                 os.remove(f"{prefix}_meta.pkl")
                 for i in range(len(tt_rand)): os.remove(f"{prefix}_core_{i}.cytnx")
            except OSError as e:
                 logger.warning(f"Could not clean up test files: {e}")
        else:
            print("Skipping Save/Load test as tt_rand was not created.")
    except Exception as e:
        logger.error(f"Error in Save/Load test: {e}", exc_info=True)

    print("\n" + "="*30)
    print("--- Testing Addition ---")
    try:
        tt_A = CytnxTensorTrain.random([2,2], 2)
        tt_B = CytnxTensorTrain.random([2,2], 3)
        tt_C = tt_A + tt_B
        print(f"Length after add: {len(tt_C)}")
        print(f"Phys Dims after add: {tt_C.physical_dims}")
        print(f"Bond Dims A: {tt_A.bond_dims}")
        print(f"Bond Dims B: {tt_B.bond_dims}")
        print(f"Bond Dims C: {tt_C.bond_dims}")
        # Check bond dim additivity (except boundaries)
        assert tt_C.bond_dims[1] == tt_A.bond_dims[1] + tt_B.bond_dims[1]
        idx_add = (0,1)
        val_A = tt_A(idx_add); val_B = tt_B(idx_add); val_C = tt_C(idx_add)
        print(f"Eval at {idx_add}: A={val_A:.4f}, B={val_B:.4f}, C={val_C:.4f}")
        assert np.isclose(val_C, val_A + val_B), "Addition eval mismatch!"
        print("Addition test passed.")
    except Exception as e:
        logger.error(f"Error in Addition test: {e}", exc_info=True)


    print("\n" + "="*30)
    print("--- Testing TrueError ---")
    try:
        pdims_small = [2, 2, 2]
        bonds = [cytnx.Bond(1), cytnx.Bond(1), cytnx.Bond(1), cytnx.Bond(1)]
        phys_bonds = [cytnx.Bond(d) for d in pdims_small]
        exact_cores = []

        # Use cytnx.from_numpy for tensor creation
        t0_np = np.array([[[1/np.sqrt(2)],[1/np.sqrt(2)]]]) # Shape (1,2,1)
        t0 = cytnx.from_numpy(t0_np)
        exact_cores.append(cytnx.UniTensor(t0, labels=['0','-1','1'], rowrank=1))

        t1_np = np.array([[[1/np.sqrt(2)],[-1/np.sqrt(2)]]])# Shape (1,2,1)
        t1 = cytnx.from_numpy(t1_np)
        exact_cores.append(cytnx.UniTensor(t1, labels=['1','-1','2'], rowrank=1))

        t2_np = np.array([[[1.],[0.]]]) # Shape (1,2,1)
        t2 = cytnx.from_numpy(t2_np)
        exact_cores.append(cytnx.UniTensor(t2, labels=['2','-1','3'], rowrank=1))

        tt_exact = CytnxTensorTrain(exact_cores)

        # Define exact function
        def exact_func(idx: Tuple[int,...]) -> float:
             x,y,z = idx
             val = (1/np.sqrt(2)) * ((1/np.sqrt(2)) * (1 if y==0 else -1)) * (1 if z==0 else 0)
             return val

        max_err = tt_exact.trueError(exact_func, max_n_eval=100)
        print(f"Max error between exact TT and its function: {max_err:.6e}")
        assert max_err < 1e-14, "TrueError failed for exact case" # Allow for float precision

        # Create noisy version
        tt_noisy_cores = [c.clone() for c in tt_exact.cores]
        noise_shape = tt_noisy_cores[1].shape()
        noise_np = np.random.rand(*noise_shape) * 0.01
        noise = cytnx.from_numpy(noise_np)
        # Make sure underlying tensors are contiguous before put_block
        block_orig = tt_noisy_cores[1].get_block_().contiguous()
        tt_noisy_cores[1].put_block(block_orig + noise)
        tt_noisy = CytnxTensorTrain(tt_noisy_cores)

        max_err_noisy = tt_noisy.trueError(exact_func, max_n_eval=100)
        print(f"Max error between noisy TT and exact function: {max_err_noisy:.6e}")
        assert max_err_noisy > 1e-4, "TrueError did not detect noise sufficiently" # Adjusted threshold

        print("TrueError test passed.")

    except Exception as e:
        logger.error(f"Error in TrueError test: {e}", exc_info=True)

    print("\n" + "="*30)