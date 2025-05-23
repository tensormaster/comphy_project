import cytnx
from cytnx import UniTensor, Type, Device, BD_IN, BD_OUT,Bond
import numpy as np
import logging
from typing import Dict, Optional, List
from tensor_train import TensorTrain # Assume TensorTrain class is defined in tensor_train.py


logger = logging.getLogger(__name__)

class ProdOp:
    one: cytnx.Tensor

    @classmethod
    def set_identity(cls, d: int, dtype: cytnx.Type = cytnx.Type.Double, device: cytnx.Device = cytnx.Device.cpu) -> None:
        cls.one = cytnx.eye(d, dtype=dtype, device=device)
        logger.info(f"ProdOp.one set to {d}x{d} identity matrix. Dtype: {cls.one.dtype_str()}, Device: {cls.one.device_str()}")

    def __init__(self, ops: Optional[Dict[int, cytnx.Tensor]] = None) -> None:
        self.ops: Dict[int, cytnx.Tensor] = {}
        if ops is not None:
            if not hasattr(ProdOp, 'one') or ProdOp.one is None:
                raise RuntimeError("ProdOp.one is not set. Call ProdOp.set_identity(d) before initializing ProdOp with operators.")
            
            # These are correct as they come from an instance
            ref_dtype_enum = ProdOp.one.dtype()
            ref_device_enum = ProdOp.one.device()
            ref_dim = ProdOp.one.shape()[0]

            # For logging, get string names from the instance if possible, or map from enum
            ref_dtype_str = ProdOp.one.dtype_str() # Use instance method
            ref_device_str = ProdOp.one.device_str() # Use instance method

            for site, op_input in ops.items():
                if not (isinstance(site, int) and site >= 0):
                    raise ValueError(f"Site index must be a non-negative int, got {site}")
                if not isinstance(op_input, cytnx.Tensor):
                    raise TypeError(f"Operator at site {site} must be a cytnx.Tensor, got {type(op_input)}")
                
                op_shape = op_input.shape()
                if len(op_shape) != 2 or op_shape[0] != op_shape[1]:
                    raise ValueError(f"Operator at site {site} must be a square 2D tensor, got shape {op_shape}")
                if op_shape[0] != ref_dim:
                    raise ValueError(
                        f"Operator at site {site} has dimension {op_shape[0]}, "
                        f"but ProdOp.one has dimension {ref_dim}."
                    )

                processed_op = op_input.clone()
                op_dtype = processed_op.dtype()
                op_device = processed_op.device()

                op_is_complex = (op_dtype == Type.ComplexDouble or op_dtype == Type.ComplexFloat)
                ref_is_real = (ref_dtype_enum == Type.Double or ref_dtype_enum == Type.Float)

                if op_is_complex and ref_is_real:
                    # Log with instance's dtype_str() method
                    logger.debug(f"Site {site}: Op is complex ({processed_op.dtype_str()}), "
                                 f"ProdOp.one is real ({ref_dtype_str}). Extracting real part.")
                    real_part_op = processed_op.real()
                    if real_part_op.dtype() != ref_dtype_enum:
                        logger.debug(f"Site {site}: Real part is {real_part_op.dtype_str()}, "
                                     f"converting to ProdOp.one's real dtype {ref_dtype_str}.")
                        processed_op = real_part_op.astype(ref_dtype_enum)
                    else:
                        processed_op = real_part_op
                elif op_dtype != ref_dtype_enum: # Handles all other astype-able cases
                    logger.debug(f"Site {site}: Op dtype {processed_op.dtype_str()} differs from ProdOp.one dtype {ref_dtype_str}. Performing astype.")
                    processed_op = processed_op.astype(ref_dtype_enum)
                
                if processed_op.device() != ref_device_enum: # Check device after dtype conversion
                    logger.debug(f"Site {site}: Op device {processed_op.device_str()} differs from ProdOp.one device {ref_device_str}. Moving to target device.")
                    processed_op = processed_op.to(ref_device_enum)
                
                self.ops[site] = processed_op
        
        # Use strings obtained from ProdOp.one instance for the final log
        final_log_dtype_str = ProdOp.one.dtype_str() if hasattr(ProdOp, 'one') and ProdOp.one else "Unknown (ProdOp.one not set)"
        final_log_device_str = ProdOp.one.device_str() if hasattr(ProdOp, 'one') and ProdOp.one else "Unknown"
        logger.debug(f"ProdOp initialized with {len(self.ops)} operators using ProdOp.one (dtype: {final_log_dtype_str}, device: {final_log_device_str}).")

    def length(self) -> int:
        return 0 if not self.ops else max(self.ops.keys()) + 1

    def to_tensorTrain(self, length_: int) -> TensorTrain:
        if length_ < 0:
            raise ValueError(f"length_ must be non-negative, got {length_}")
        if not hasattr(ProdOp, 'one') or ProdOp.one is None:
            raise RuntimeError("ProdOp.one is not set. Call ProdOp.set_identity(d) first.")
        
        d_phys = ProdOp.one.shape()[0] 
        phys_dim_merged = d_phys * d_phys

        target_dtype = ProdOp.one.dtype()
        target_device = ProdOp.one.device()

        ut_cores: List[cytnx.UniTensor] = []
        for i in range(length_):
            op_matrix_dd = self.ops.get(i, ProdOp.one) 
            
            core_data_tensor = op_matrix_dd.reshape(1, phys_dim_merged, 1)
            
            bd_left = cytnx.Bond(1, BD_IN)
            bd_phys_merged = cytnx.Bond(phys_dim_merged, BD_OUT) 
            bd_right = cytnx.Bond(1, BD_OUT)

            label_left = "L_bound" if i == 0 else f"link{i-1}"
            label_phys = f"p_mrg{i}" 
            label_right = "R_bound" if i == length_ - 1 else f"link{i}"
            core_labels = [label_left, label_phys, label_right]
            
            ut_core = UniTensor(bonds=[bd_left, bd_phys_merged, bd_right], 
                                labels=core_labels, 
                                rowrank=1,
                                dtype=target_dtype, 
                                device=target_device)
            ut_core.put_block(core_data_tensor.contiguous_()) 


            ut_cores.append(ut_core)
            
        return TensorTrain(ut_cores)
    

    def overlap(self, mps: TensorTrain) -> float:
        if not isinstance(mps, TensorTrain):
            raise TypeError("Input 'mps' for ProdOp.overlap must be a TensorTrain instance.")

        L_mps = len(mps.M)

        if not self.ops: 
            if L_mps == 0: 
                logger.info("ProdOp.overlap: Both ProdOp and MPS are empty. Returning 0.0.")
                return 0.0 
            # If ProdOp is empty but MPS is not, all ops are identity.
            # This will be handled by self.ops.get(k, ProdOp.one) in the loop.
            logger.info("ProdOp.overlap: ProdOp is empty, treating all operators as identity.")

        if L_mps == 0 and self.ops: # Non-empty ProdOp, empty MPS
            raise ValueError("Cannot compute overlap of non-empty ProdOp with an empty MPS.")
        
        if L_mps > 0 and not mps.M[0]: # Check if mps.M exists and is not empty before accessing mps.M[0]
             raise ValueError("Input MPS has non-zero length but its core list appears empty or first core is None.")


        if not hasattr(ProdOp, 'one') or ProdOp.one is None:
            raise RuntimeError("ProdOp.one is not set. Call ProdOp.set_identity(d) first.")

        d_phys = ProdOp.one.shape()[0]
        phys_dim_merged_expected = d_phys * d_phys
        
        op_ref_dtype = ProdOp.one.dtype()
        # op_ref_device = ProdOp.one.device() # Less critical for numpy conversion to float list

        weights_for_sum: List[List[float]] = [] 

        for k in range(L_mps):
            if not mps.M[k]: 
                raise ValueError(f"Core {k} in the input mps is None.")
            
            # Assuming physical bond is at index 1 for the UniTensor core in mps
            # and its label corresponds to the physical index.
            try:
                mps_core_phys_bond = mps.M[k].bonds()[1]
                mps_core_phys_dim = mps_core_phys_bond.dim()
            except IndexError:
                raise ValueError(f"Core {k} in MPS does not have at least 2 bonds. Shape: {mps.M[k].shape()}, Labels: {mps.M[k].labels()}")


            if mps_core_phys_dim != phys_dim_merged_expected:
                raise ValueError(
                    f"Physical dimension of input mps core {k} (label: '{mps.M[k].labels()[1]}') is {mps_core_phys_dim}, "
                    f"but ProdOp requires a merged physical dimension of {phys_dim_merged_expected} (from d={d_phys})."
                )

            op_matrix_dd = self.ops.get(k, ProdOp.one) 
            
            # Ensure op_matrix_dd is compatible for numpy conversion (correct dtype for data, on CPU)
            # ProdOp.one and self.ops elements are already on consistent dtype/device from __init__/set_identity
            op_matrix_dd_for_numpy = op_matrix_dd.to(Device.cpu) 
            
            if op_matrix_dd_for_numpy.shape() != [d_phys, d_phys]:
                 raise ValueError(
                    f"Internal operator at site {k} has shape {op_matrix_dd_for_numpy.shape()} "
                    f"after .to(cpu), expected {[d_phys, d_phys]}."
                )
            
            # Flatten the dxd operator. C++ {op.begin(), op.end()} implies memory order.
            # For a cytnx.Tensor from numpy (row-major default) or direct init (row-major),
            # .flatten() or .numpy().flatten() (default 'C') should be row-major.
            op_flat_numpy = op_matrix_dd_for_numpy.numpy().flatten(order='C') # Explicitly row-major
            
            # Convert to List[float], taking .real if elements are complex,
            # as TensorTrain.sum expects List[List[float]].
            current_weight_vector = []
            for x_val in op_flat_numpy:
                if isinstance(x_val, complex):
                    current_weight_vector.append(float(x_val.real))
                else:
                    current_weight_vector.append(float(x_val))
            weights_for_sum.append(current_weight_vector)

        result_from_sum = mps.sum(weights_for_sum)
        
        logger.debug(f"ProdOp.overlap (via mps.sum): L_mps={L_mps}, result_from_sum={result_from_sum}, type={type(result_from_sum)}")

        final_result: float
        if isinstance(result_from_sum, complex):
            if abs(result_from_sum.imag) > 1e-12: 
                logger.warning(
                    f"ProdOp.overlap (via mps.sum) received complex result {result_from_sum}. "
                    "Returning its real part."
                )
            final_result = result_from_sum.real
        elif isinstance(result_from_sum, (int, float, np.number)): # Handle numpy scalar types
            final_result = float(result_from_sum)
        else:
            logger.error(f"ProdOp.overlap: mps.sum returned unexpected type {type(result_from_sum)}. Value: {result_from_sum}")
            try:
                final_result = float(result_from_sum) # Attempt conversion
            except (TypeError, ValueError) as e:
                 raise TypeError(f"Cannot convert result of mps.sum (type {type(result_from_sum)}, value {result_from_sum}) to float for ProdOp.overlap. Original error: {e}")
        
        return final_result


    def overlap_mps(self, psi: TensorTrain) -> float:
        if not isinstance(psi, TensorTrain):
            raise TypeError("Input 'psi' for ProdOp.overlap_mps must be a TensorTrain instance.")
        
        # Handle empty psi
        if not psi.M: 
            if not self.ops: # Both ProdOp and psi are empty
                logger.info("ProdOp.overlap_mps: Both ProdOp and psi are empty. Returning 0.0")
                return 0.0
            else: # Non-empty ProdOp on empty MPS
                raise ValueError("Cannot compute expectation value with non-empty ProdOp on an empty MPS psi.")

        L_psi = len(psi.M)
        # If self.ops is empty here, it implies ProdOp is the Identity operator.
        # The loop with self.ops.get(k, ProdOp.one) will correctly use ProdOp.one.
        
        max_op_site = max(self.ops.keys(), default=-1)
        # max_op_site can be -1 if self.ops is empty. L_psi > 0 is always true if psi.M is not empty.
        if max_op_site >= L_psi : # This condition is fine even if max_op_site is -1.
            raise ValueError(
                f"Operator site index {max_op_site} is out of bounds for MPS of length {L_psi}."
            )
        
        if not hasattr(ProdOp, 'one') or ProdOp.one is None:
            raise RuntimeError("ProdOp.one is not set. Call ProdOp.set_identity(d) first.")
        
        d_phys = ProdOp.one.shape()[0]

        # --- Determine effective_dtype and effective_device ---
        op_dtypes = [op.dtype() for op in self.ops.values()]
        psi_core_dtypes = [core.dtype() for core in psi.M if core is not None] # psi.M checked not empty
        all_involved_dtypes = [ProdOp.one.dtype()] + op_dtypes + psi_core_dtypes
        is_complex = any(dt in [Type.ComplexDouble, Type.ComplexFloat] for dt in all_involved_dtypes)
        is_double = any(dt in [Type.Double, Type.ComplexDouble] for dt in all_involved_dtypes)
        
        effective_dtype = Type.Void 
        if is_complex:
            effective_dtype = Type.ComplexDouble if is_double else Type.ComplexFloat
        else:
            effective_dtype = Type.Double if is_double else Type.Float
        
        effective_device = ProdOp.one.device()
        
        _dummyt = cytnx.zeros(1,dtype=effective_dtype,device=effective_device) # For logging
        logger.debug(f"Overlap_mps: EffectiveDtype={_dummyt.dtype_str()}, Device={_dummyt.device_str()}")

        # --- Initialize Environment Tensor L_env ---
        env_LpsiC_lbl, env_Lop_lbl, env_LpsiK_lbl = "envLpsiC", "envLop", "envLpsiK"
        L_env = UniTensor(
            bonds=[Bond(1, BD_OUT), Bond(1, BD_OUT), Bond(1, BD_OUT)],
            labels=[env_LpsiC_lbl, env_Lop_lbl, env_LpsiK_lbl],
            rowrank=1, dtype=effective_dtype, device=effective_device
        )
        L_env.get_block_()[0,0,0] = 1.0

        # --- Iteration over MPS sites ---
        for k in range(L_psi):
            psi_k = psi.M[k].astype(effective_dtype).to(effective_device)
            psi_k_conj = psi_k.Conj()
            op_k_dd = self.ops.get(k, ProdOp.one).astype(effective_dtype).to(effective_device)

            op_vL_lbl, op_pIn_lbl, op_pOut_lbl, op_vR_lbl = f"opvL{k}",f"opPIn{k}",f"opPOut{k}",f"opvR{k}"
            Op_k = UniTensor(
                bonds=[Bond(1,BD_IN), Bond(d_phys,BD_OUT), Bond(d_phys,BD_IN), Bond(1,BD_OUT)],
                labels=[op_vL_lbl, op_pIn_lbl, op_pOut_lbl, op_vR_lbl], 
                rowrank=2, dtype=effective_dtype, device=effective_device
            )
            Op_k.put_block(op_k_dd.reshape(1,d_phys,d_phys,1).contiguous_())

            psi_k_L_orig_lbl, psi_k_P_orig_lbl, psi_k_R_orig_lbl = psi_k.labels()
            psi_k_conj_L_orig_lbl, psi_k_conj_P_orig_lbl, psi_k_conj_R_orig_lbl = psi_k_conj.labels()
            
            L_env_ctr = L_env.clone()
            psi_k_conj_ctr = psi_k_conj.clone() 
            Op_k_ctr = Op_k.clone()
            
            # --- Contract T1 ---
            ct_lbl_L_psiC = f"__ct_LpsiC_{k}__" 
            L_env_ctr.relabel_(env_LpsiC_lbl, ct_lbl_L_psiC)
            
            idx_redirect_psi_conj_L = psi_k_conj_ctr.labels().index(psi_k_conj_L_orig_lbl)
            bond_to_check_psi_conj_L = psi_k_conj_ctr.bonds()[idx_redirect_psi_conj_L]
            original_type_name_T1 = bond_to_check_psi_conj_L.type().name
            if bond_to_check_psi_conj_L.type() != BD_IN: 
                 bond_to_check_psi_conj_L.redirect_()
                 logger.debug(f"Site {k} T1-redirect: psi_k_conj_ctr leg '{psi_k_conj_L_orig_lbl}' was {original_type_name_T1}, redirected to {bond_to_check_psi_conj_L.type().name}")
            else:
                 logger.debug(f"Site {k} T1-redirect: psi_k_conj_ctr leg '{psi_k_conj_L_orig_lbl}' is already {original_type_name_T1} (BD_IN/KET).")
            psi_k_conj_ctr.relabel_(psi_k_conj_L_orig_lbl, ct_lbl_L_psiC)
            T1 = cytnx.Contract(L_env_ctr, psi_k_conj_ctr)

            # --- Contract T2 ---
            ct_lbl_Lop_opvL = f"__ct_Lop_opvL_{k}__"
            ct_lbl_psiCP_opPIn = f"__ct_psiCP_opPIn_{k}__"
            T1_ctr = T1.clone()
            T1_ctr.relabel_(env_Lop_lbl, ct_lbl_Lop_opvL)
            T1_ctr.relabel_(psi_k_conj_P_orig_lbl, ct_lbl_psiCP_opPIn)
            
            Op_k_for_T2 = Op_k_ctr.clone() 
            Op_k_for_T2.relabel_(op_vL_lbl, ct_lbl_Lop_opvL)
            Op_k_for_T2.relabel_(op_pIn_lbl, ct_lbl_psiCP_opPIn)
            
            idx_T1_psiCP_leg = T1_ctr.labels().index(ct_lbl_psiCP_opPIn)
            bond_T1_psiCP_leg = T1_ctr.bonds()[idx_T1_psiCP_leg]
            original_type_name_T2 = bond_T1_psiCP_leg.type().name
            if bond_T1_psiCP_leg.type() != BD_IN:
                 bond_T1_psiCP_leg.redirect_()
                 logger.debug(f"Site {k} T2-redirect: T1_ctr leg '{ct_lbl_psiCP_opPIn}' was {original_type_name_T2}, redirected to {bond_T1_psiCP_leg.type().name}")
            else:
                 logger.debug(f"Site {k} T2-redirect: T1_ctr leg '{ct_lbl_psiCP_opPIn}' is already {original_type_name_T2} (BD_IN/KET).")
            T2 = cytnx.Contract(T1_ctr, Op_k_for_T2)
            
            # --- Contract L_env_updated (final for site k) ---
            T2_for_final = T2.clone()
            psi_k_for_final = psi_k.clone() # Use original psi_k (already cloned as psi_k_ctr, but use fresh name)

            # Define unique names for the "dangling" legs that will form the next L_env
            unique_dangling_T2_psiCR = psi_k_conj_R_orig_lbl + f"_d1_{k}"
            if psi_k_conj_R_orig_lbl in T2_for_final.labels():
                T2_for_final.relabel_(psi_k_conj_R_orig_lbl, unique_dangling_T2_psiCR)
            
            unique_dangling_T2_opR = op_vR_lbl + f"_d2_{k}"
            if op_vR_lbl in T2_for_final.labels():
                T2_for_final.relabel_(op_vR_lbl, unique_dangling_T2_opR)

            unique_dangling_psiK_R = psi_k_R_orig_lbl + f"_d3_{k}"
            if psi_k_R_orig_lbl in psi_k_for_final.labels():
                psi_k_for_final.relabel_(psi_k_R_orig_lbl, unique_dangling_psiK_R)
            
            ct_lbl_LpsiK_psiKL = f"__ct_LpsiK_psiKL_{k}__"
            ct_lbl_opPOut_psiKP = f"__ct_opPOut_psiKP_{k}__"

            T2_for_final.relabel_(env_LpsiK_lbl, ct_lbl_LpsiK_psiKL)
            T2_for_final.relabel_(op_pOut_lbl, ct_lbl_opPOut_psiKP)   
            psi_k_for_final.relabel_(psi_k_L_orig_lbl, ct_lbl_LpsiK_psiKL)
            psi_k_for_final.relabel_(psi_k_P_orig_lbl, ct_lbl_opPOut_psiKP)
            
            logger.debug(f"Site {k} Pre-FinalContract: T2_for_final labels: {T2_for_final.labels()}")
            logger.debug(f"  Bond types for T2_for_final['{ct_lbl_LpsiK_psiKL}']: {T2_for_final.bond(ct_lbl_LpsiK_psiKL).type().name}")
            logger.debug(f"  Bond types for T2_for_final['{ct_lbl_opPOut_psiKP}']: {T2_for_final.bond(ct_lbl_opPOut_psiKP).type().name}")
            logger.debug(f"Site {k} Pre-FinalContract: psi_k_for_final labels: {psi_k_for_final.labels()}")
            logger.debug(f"  Bond types for psi_k_for_final['{ct_lbl_LpsiK_psiKL}']: {psi_k_for_final.bond(ct_lbl_LpsiK_psiKL).type().name}")
            logger.debug(f"  Bond types for psi_k_for_final['{ct_lbl_opPOut_psiKP}']: {psi_k_for_final.bond(ct_lbl_opPOut_psiKP).type().name}")
            
            L_env_updated = cytnx.Contract(T2_for_final, psi_k_for_final)
            
            if k < L_psi - 1:
                permute_target_order_on_L_env_updated = [unique_dangling_T2_psiCR, unique_dangling_T2_opR, unique_dangling_psiK_R]
                current_L_updated_labels = L_env_updated.labels()
                if not all(pt_lbl in current_L_updated_labels for pt_lbl in permute_target_order_on_L_env_updated):
                    logger.error(f"Site {k} L_env_updated permute error: Expected labels {permute_target_order_on_L_env_updated} not all found in {current_L_updated_labels}.")
                
                L_env_perm = L_env_updated.permute(permute_target_order_on_L_env_updated, rowrank=1)
                
                bonds_for_next_L_env = []
                for i_next_leg, next_leg_lbl_permuted in enumerate(L_env_perm.labels()):
                    temp_bond = L_env_perm.bonds()[i_next_leg].clone()
                    if temp_bond.type() != BD_OUT: # Ensure outgoing for next L_env
                        temp_bond.redirect_()
                    bonds_for_next_L_env.append(temp_bond)

                L_env = UniTensor(
                    bonds=bonds_for_next_L_env,
                    labels=[env_LpsiC_lbl, env_Lop_lbl, env_LpsiK_lbl], 
                    rowrank=1, dtype=effective_dtype, device=effective_device
                )
                L_env.put_block(L_env_perm.get_block_())
            else:
                L_env = L_env_updated
        
        if not (L_env.rank() == 3 and L_env.shape() == [1,1,1] or L_env.is_scalar()):
            logger.error(f"Final L_env error. Rank:{L_env.rank()},Shape:{L_env.shape()},Labels:{L_env.labels()}")
            raise ValueError("Final L_env structure incorrect.")
        
        final_val = L_env.item()
        # For logging dtype of final_val, which might be a Python scalar now
        final_val_dtype_str = ""
        if isinstance(final_val, complex):
            final_val_dtype_str = "Python complex"
        elif isinstance(final_val, float):
            final_val_dtype_str = "Python float"
        elif isinstance(final_val, int):
            final_val_dtype_str = "Python int"
        else: # Fallback for other types like numpy scalars if L_env.item() returns them
            final_val_dtype_str = str(type(final_val))

        logger.debug(f"Overlap_mps: L_psi={L_psi}, final val={final_val}, val_type={final_val_dtype_str}")

        # Corrected return logic for final_val
        if isinstance(final_val, complex):
            if abs(final_val.imag) > 1e-9: 
                logger.warning(f"Overlap_mps: Result {final_val} has a significant imaginary part. Returning .real part: {final_val.real}")
            return final_val.real # Always return .real if it's a complex type
        elif isinstance(final_val, (float, int, np.number)): # Handles float, int, and numpy real numeric types
            return float(final_val)
        else:
            # This case should ideally not be reached if L_env.item() returns standard numerics
            logger.error(f"Overlap_mps: final_val is of unexpected type {type(final_val)}. Value: {final_val}")
            # Attempt to convert, or raise a more specific error
            try:
                return float(final_val) 
            except TypeError as e:
                raise TypeError(f"Cannot convert final_val (type {type(final_val)}, value {final_val}) to float for return. Original error: {e}")




def build_physical_mps(d: int, L: int, dtype: cytnx.Type = cytnx.Type.Double, device: cytnx.Device = cytnx.Device.cpu) -> TensorTrain:
    if L == 0:
        return TensorTrain([]) # Return a TensorTrain with an empty list of cores
    
    # Create a list of cytnx.Tensor objects first
    # These will be (1,d,1)
    tensor_cores = []
    for _ in range(L):
        # Create a 1D tensor of ones, then reshape to (1,d,1)
        # Ensure it's on the specified dtype and device
        core_data_1d = cytnx.ones([d], dtype=dtype, device=device)
        core_tensor_3d = core_data_1d.reshape(1, d, 1)
        tensor_cores.append(core_tensor_3d)
        
    # TensorTrain constructor will convert these to UniTensors
    return TensorTrain(tensor_cores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG to see more detailed logs from ProdOp methods
    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG) # Alternatively

    # Standard test parameters
    d_std, L_std = 3, 4
    # Edge case test parameters
    d_edge, L_edge = 2, 1

    # Helper function for expected value in overlap_mps (simplified for product state MPS)
    def calculate_expected_overlap_mps_simple(prod_op: ProdOp, mps_len: int, mps_phys_dim: int) -> float:
        # This helper calculates Product_k [ Sum_{elements of Op_k} ]
        # Assumes MPS is a product state of all |+> states (sum of basis states)
        # which, when contracted with <+|Op|+>, gives Sum_{elements of Op}
        
        if not hasattr(ProdOp, 'one') or ProdOp.one is None: # Guard clause
            ProdOp.set_identity(mps_phys_dim) # Default init if not set

        expected_val_complex = complex(1.0, 0.0)
        
        for i in range(mps_len):
            local_op_tensor = prod_op.ops.get(i, ProdOp.one) # Get dxd Tensor
            
            # Sum elements of the local operator
            # Ensure it's on CPU for numpy().sum() and handle complex types
            local_op_for_sum = local_op_tensor.to(Device.cpu)
            sum_elements_local_op_np = local_op_for_sum.numpy().sum()
            
            current_op_sum_complex: complex
            if isinstance(sum_elements_local_op_np, complex):
                current_op_sum_complex = complex(sum_elements_local_op_np)
            else: # It's a real number (float or int from numpy sum)
                current_op_sum_complex = complex(float(sum_elements_local_op_np))
            
            expected_val_complex *= current_op_sum_complex
            
        return expected_val_complex.real


    print(f"--- Standard Tests (d={d_std}, L={L_std}) for overlap_mps ---")
    ProdOp.set_identity(d_std, dtype=cytnx.Type.Double, device=Device.cpu) # Ensure type for these tests
    mps_std = build_physical_mps(d_std, L_std, dtype=cytnx.Type.Double, device=Device.cpu)

    print(f"Test 1 (d={d_std}, L={L_std}): Empty ProdOp overlap_mps")
    op_empty_std = ProdOp()
    res_test1 = op_empty_std.overlap_mps(mps_std)
    print(f"  Calculated: {res_test1}, Expected: 81.0")
    assert np.isclose(res_test1, 81.0), "Test Case 1 overlap_mps Failed"

    print(f"Test 2 (d={d_std}, L={L_std}): Identity MPO overlap_mps")
    ops_id_std = {i: ProdOp.one.clone() for i in range(L_std)} # Clone to be safe
    op_id_std = ProdOp(ops_id_std)
    id_res_std = op_id_std.overlap_mps(mps_std)
    expected_id_std = calculate_expected_overlap_mps_simple(op_id_std, L_std, d_std)
    print(f"  Calculated: {id_res_std}, Expected: {expected_id_std}")
    assert np.isclose(id_res_std, expected_id_std), f"Test Case 2 overlap_mps Failed: {id_res_std} vs {expected_id_std}"

    print(f"Test 3 (d={d_std}, L={L_std}): Single‐site scaled MPO overlap_mps")
    op_scaled_local_std = cytnx.eye(d_std, dtype=ProdOp.one.dtype(), device=ProdOp.one.device()) * 2.0
    site_for_scaled_op_std = L_std // 2
    ops_scaled_std = {site_for_scaled_op_std: op_scaled_local_std}
    op_scaled_std = ProdOp(ops_scaled_std)
    scaled_res_std = op_scaled_std.overlap_mps(mps_std)
    expected_scaled_std = calculate_expected_overlap_mps_simple(op_scaled_std, L_std, d_std)
    print(f"  (Site {site_for_scaled_op_std}) Calculated: {scaled_res_std}, Expected: {expected_scaled_std}")
    assert np.isclose(scaled_res_std, expected_scaled_std), f"Test Case 3 overlap_mps Failed: {scaled_res_std} vs {expected_scaled_std}"
    
    print(f"Test 4 (d={d_std}, L={L_std}): Out-of-bounds operator test overlap_mps")
    try:
        op_oob_std = ProdOp({L_std: ProdOp.one.clone()})
        op_oob_std.overlap_mps(mps_std) # This should raise ValueError due to site index
        assert False, "Test Case 4 overlap_mps Failed: Expected ValueError for out-of-bounds site"
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")
    
    print("\n--- Standard Spin Operators (d=2, L=4) for overlap_mps ---")
    d_spin, L_spin = 2, 4 # Using d=2 for standard spin operators
    ProdOp.set_identity(d_spin, dtype=cytnx.Type.Double, device=Device.cpu)
    mps_spin_d2 = build_physical_mps(d_spin, L_spin, dtype=cytnx.Type.Double, device=Device.cpu)
    test_site_spin = L_spin // 2

    Sx_op_d2 = cytnx.zeros([d_spin, d_spin], dtype=cytnx.Type.Double, device=Device.cpu)
    Sx_op_d2[0,1] = 0.5; Sx_op_d2[1,0] = 0.5
    op_sx_d2 = ProdOp({test_site_spin: Sx_op_d2})
    sx_res_d2 = op_sx_d2.overlap_mps(mps_spin_d2)
    expected_sx_d2 = calculate_expected_overlap_mps_simple(op_sx_d2, L_spin, d_spin)
    print(f"Sx operator at site {test_site_spin} (d={d_spin}): Sum_el_local={Sx_op_d2.numpy().sum():.1f}")
    print(f"  Calculated: {sx_res_d2:.8f}, Expected: {expected_sx_d2:.8f}")
    assert np.isclose(sx_res_d2, expected_sx_d2), "Sx Test Failed (d=2)"

    Sy_op_d2 = cytnx.zeros([d_spin, d_spin], dtype=cytnx.Type.ComplexDouble, device=Device.cpu)
    Sy_op_d2[0,1] = -0.5j; Sy_op_d2[1,0] = 0.5j
    op_sy_d2 = ProdOp({test_site_spin: Sy_op_d2})
    sy_res_d2 = op_sy_d2.overlap_mps(mps_spin_d2)
    expected_sy_d2 = calculate_expected_overlap_mps_simple(op_sy_d2, L_spin, d_spin)
    print(f"Sy operator at site {test_site_spin} (d={d_spin}): Sum_el_local={Sy_op_d2.numpy().sum()}") # Should be 0
    print(f"  Calculated: {sy_res_d2:.8f}, Expected: {expected_sy_d2:.8f}") # Expect 0
    assert np.isclose(sy_res_d2, expected_sy_d2, atol=1e-12), "Sy Test Failed (d=2)"

    Sz_op_d2 = cytnx.zeros([d_spin, d_spin], dtype=cytnx.Type.Double, device=Device.cpu)
    Sz_op_d2[0,0] = 0.5; Sz_op_d2[1,1] = -0.5
    op_sz_d2 = ProdOp({test_site_spin: Sz_op_d2})
    sz_res_d2 = op_sz_d2.overlap_mps(mps_spin_d2)
    expected_sz_d2 = calculate_expected_overlap_mps_simple(op_sz_d2, L_spin, d_spin)
    print(f"Sz operator at site {test_site_spin} (d={d_spin}): Sum_el_local={Sz_op_d2.numpy().sum():.1f}") # Should be 0
    print(f"  Calculated: {sz_res_d2:.8f}, Expected: {expected_sz_d2:.8f}") # Expect 0
    assert np.isclose(sz_res_d2, expected_sz_d2, atol=1e-12), "Sz Test Failed (d=2)"

    print("\n--- Edge Case Tests (d_edge, L_edge) for overlap_mps ---")
    ProdOp.set_identity(d_edge, dtype=cytnx.Type.Double, device=Device.cpu)
    mps_edge = build_physical_mps(d_edge, L_edge, dtype=cytnx.Type.Double, device=Device.cpu)

    print("Test 5: L=1, Identity operator overlap_mps")
    ops_id_L1 = {0: ProdOp.one.clone()}
    op_id_L1 = ProdOp(ops_id_L1)
    res_id_L1 = op_id_L1.overlap_mps(mps_edge)
    expected_id_L1 = calculate_expected_overlap_mps_simple(op_id_L1, L_edge, d_edge) 
    print(f"  Calculated: {res_id_L1}, Expected: {expected_id_L1}")
    assert np.isclose(res_id_L1, expected_id_L1), "Test Case 5 overlap_mps Failed"

    print("Test 7: L=1, Complex operator overlap_mps (custom complex op for d_edge)")
    custom_complex_op_L1_np = np.array([[1+1j, 2j], [3+0j, 4-2j]], dtype=np.complex128) if d_edge==2 else np.array([[1+1j]], dtype=np.complex128)
    custom_complex_op_L1 = cytnx.from_numpy(custom_complex_op_L1_np).to(Device.cpu) # Ensure CPU
    
    op_complex_L1 = ProdOp({0: custom_complex_op_L1})
    res_complex_L1 = op_complex_L1.overlap_mps(mps_edge)
    expected_complex_L1 = calculate_expected_overlap_mps_simple(op_complex_L1, L_edge, d_edge)
    print(f"  Sum of local op elements: {custom_complex_op_L1.numpy().sum()}")
    print(f"  Calculated: {res_complex_L1:.8f}, Expected: {expected_complex_L1:.8f}")
    assert np.isclose(res_complex_L1, expected_complex_L1, atol=1e-12), "Test Case 7 overlap_mps Failed"

    print("\n--- ProdOp.overlap specific tests (using mps.sum logic) ---")
    # Test 14: ProdOp.overlap method (using merged phys_dim for mps)
    # This tests the self.overlap which calls mps.sum()
    d_for_overlap_method = 2 # Example physical dim d for local ops
    L_for_overlap_method = 1 # Example length
    
    ProdOp.set_identity(d_for_overlap_method, dtype=Type.Double, device=Device.cpu)
    
    # Create an MPS with merged physical dimension d*d
    # For L=1, mps_merged.M[0] has shape (1, d*d, 1)
    merged_phys_dim_mps = d_for_overlap_method * d_for_overlap_method
    
    # Core for mps_merged: make it an "all ones" state in the merged basis for simplicity
    # This means its sum over weights_from_ProdOp will be Sum_k (w_k * 1.0)
    # And if weights_from_ProdOp is flattened Identity, result should be d*d.
    # If weights_from_ProdOp is flattened X, result should be Sum_elements(X).

    # Create the mps with UniTensor cores directly
    mps_merged_core_data = cytnx.ones([merged_phys_dim_mps], dtype=Type.Double, device=Device.cpu).reshape(1, merged_phys_dim_mps, 1)
    
    label_L_mrg = "L_bound"
    label_P_mrg = "p_mrg0" # Merged physical index for mps
    label_R_mrg = "R_bound"
    
    bd_L_mrg = Bond(1, BD_IN)
    bd_P_mrg = Bond(merged_phys_dim_mps, BD_OUT) # This is the bond ProdOp.overlap expects for its weights
    bd_R_mrg = Bond(1, BD_OUT)
    
    mps_merged_ut_core = UniTensor(bonds=[bd_L_mrg, bd_P_mrg, bd_R_mrg],
                                   labels=[label_L_mrg, label_P_mrg, label_R_mrg],
                                   rowrank=1)
    mps_merged_ut_core.put_block(mps_merged_core_data)
    
    mps_merged = TensorTrain([mps_merged_ut_core])

    print(f"Test 14 (ProdOp.overlap with d={d_for_overlap_method}, L={L_for_overlap_method}): Op is Identity")
    op_for_overlap_id = ProdOp({0: ProdOp.one.clone()}) 
    overlap_res_id = op_for_overlap_id.overlap(mps_merged)
    # Expected: mps.sum([flattened_identity_dxd])
    # mps.sum with all-ones state and weights W is Sum(W).
    # flattened_identity_dxd has d ones and d*d - d zeros. Sum = d.
    expected_overlap_res_id = float(d_for_overlap_method) 
    print(f"  Calculated: {overlap_res_id}, Expected: {expected_overlap_res_id}")
    assert np.isclose(overlap_res_id, expected_overlap_res_id), "Test 14a (Identity Op) Failed"

    print(f"Test 14b (ProdOp.overlap with d={d_for_overlap_method}, L={L_for_overlap_method}): Op is all 2s")
    op_all_twos_dd = cytnx.ones([d_for_overlap_method,d_for_overlap_method], dtype=Type.Double, device=Device.cpu) * 2.0
    op_for_overlap_custom = ProdOp({0: op_all_twos_dd})
    overlap_res_custom = op_for_overlap_custom.overlap(mps_merged)
    # Expected: Sum(flattened_op_all_twos) = (d*d) * 2.0
    expected_overlap_res_custom = float( (d_for_overlap_method**2) * 2.0 )
    print(f"  Calculated: {overlap_res_custom}, Expected: {expected_overlap_res_custom}")
    assert np.isclose(overlap_res_custom, expected_overlap_res_custom), "Test 14b (All Twos Op) Failed"

    print(f"Test 14c (ProdOp.overlap with d={d_for_overlap_method}, L={L_for_overlap_method}): Complex Op")
    complex_op_dd_np = np.array([[1+1j, 2j],[0, 3-1j]], dtype=np.complex128) if d_for_overlap_method==2 else np.ones((d_for_overlap_method,d_for_overlap_method),dtype=np.complex128)*(1+1j)
    complex_op_dd = cytnx.from_numpy(complex_op_dd_np).to(Device.cpu) # ensure on CPU
    ProdOp.set_identity(d_for_overlap_method, dtype=Type.ComplexDouble, device=Device.cpu) # Match op type for ProdOp.one if needed
    op_for_overlap_complex = ProdOp({0: complex_op_dd})
    overlap_res_complex = op_for_overlap_complex.overlap(mps_merged)
    # Expected: Sum of real parts of flattened_complex_op
    expected_overlap_res_complex = float(complex_op_dd_np.flatten().real.sum())
    print(f"  Op elements sum real part: {expected_overlap_res_complex}")
    print(f"  Calculated: {overlap_res_complex}, Expected: {expected_overlap_res_complex}")
    assert np.isclose(overlap_res_complex, expected_overlap_res_complex, atol=1e-12), "Test 14c (Complex Op) Failed"

    print("\nAll ProdOp tests completed.")