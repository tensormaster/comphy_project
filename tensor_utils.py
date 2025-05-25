# tensor_utils.py
import cytnx
from cytnx import UniTensor, Tensor, BD_IN, BD_OUT, linalg, Type # Ensure Type is imported
from typing import Tuple
import logging
import numpy as np # Import numpy for manual inversion

logger = logging.getLogger(__name__)
# 這裡不需要重複設置logger.setLevel，因為在 tensor_ci.py 中已經通過 basicConfig 設置了 DEBUG 級別
# 並且每個模組的 logger 默認會繼承 root logger 的級別。
# 如果想確保，可以在主腳本或入口點統一設置 root logger。

def cube_as_matrix1(core: UniTensor) -> Tensor:
    """
    Reshapes a rank-3 UniTensor core (chi_left, d, chi_right) 
    into a matrix B(chi_left, (d*chi_right)).
    Assumes core bonds are ordered (left_virtual, physical, right_virtual)
    and core.rowrank = 1 (meaning the first bond index is the row index for the
    UniTensor's default matrix view, and data is stored [left, phys, right]
    when block is retrieved).
    """
    if not isinstance(core, UniTensor):
        raise TypeError(f"Input 'core' must be a cytnx.UniTensor, got {type(core)}")
    if core.rank() != 3:
        raise ValueError(f"Core must be rank 3 for cube_as_matrix1. Got rank {core.rank()}")
        
    block = core.contiguous().get_block_() 
    
    dim_left = block.shape()[0]
    dim_phys = block.shape()[1]
    dim_right = block.shape()[2]
    
    return block.reshape(dim_left, dim_phys * dim_right)

def cube_as_matrix2(core: UniTensor) -> Tensor:
    """
    Reshapes a rank-3 UniTensor core (chi_left, d, chi_right) 
    into a matrix B((chi_left*d), chi_right).
    Assumes core bonds are ordered (left_virtual, physical, right_virtual)
    and core.rowrank = 1 (or similar logic that makes get_block() ordered).
    """
    if not isinstance(core, UniTensor):
        raise TypeError(f"Input 'core' must be a cytnx.UniTensor, got {type(core)}")
    if core.rank() != 3:
        raise ValueError(f"Core must be rank 3 for cube_as_matrix2. Got rank {core.rank()}")
        
    block = core.contiguous().get_block_()
    
    dim_left = block.shape()[0]
    dim_phys = block.shape()[1]
    dim_right = block.shape()[2]
    
    return block.reshape(dim_left * dim_phys, dim_right)

def mat_AB1(A: Tensor, B: Tensor) -> Tensor:
    """
    Computes A @ B^-1 in a stable way using QR decomposition, mimicking C++ xfac::mat_AB1.
    A: Tensor of shape (m, n)
    B: Tensor of shape (n, n) (must be square and invertible for this QR method)
    Returns A @ B^-1, shape (m, n)
    """
    logger.debug(f"mat_AB1 called: A shape {A.shape()}, B shape {B.shape()}") # DEBUG
    if not (isinstance(A, Tensor) and isinstance(B, Tensor)):
        raise TypeError("Inputs A and B must be cytnx.Tensors.")
    if A.shape()[1] != B.shape()[0]: 
        raise ValueError(
            f"A.shape[1] ({A.shape()[1]}) must match B.shape[0] ({B.shape()[0]}) for A @ B^-1."
        )
    if B.shape()[0] != B.shape()[1]:
        logger.error(f"mat_AB1: Matrix B (shape {B.shape()}) is not square. "
                       "QR method for A @ B^-1 typically assumes B is square and invertible. "
                       "Numerical stability issues or incorrect results may occur. Consider linalg.Lstsq for non-square B.")

    m_A, n_A = A.shape() 
    n_B_rows, n_B_cols = B.shape() 

    if A.shape()[1] != B.shape()[1]:
         raise ValueError(f"Column count of A ({A.shape()[1]}) must match column count of B ({B.shape()[1]}) for C++ like stacking in mat_AB1.")

    stacked_rows = m_A + n_B_rows
    stacked_cols = n_A 
    
    common_dtype = A.dtype()
    if A.dtype() != B.dtype(): 
        if A.dtype() %2 == 1 or B.dtype() %2 == 1: 
            common_dtype = Type.ComplexDouble if A.dtype() > Type.ComplexFloat or B.dtype() > Type.ComplexFloat else Type.ComplexFloat
        elif A.dtype() == Type.Double or B.dtype() == Type.Double:
            common_dtype = Type.Double
            
    AB_stacked_data = cytnx.zeros((stacked_rows, stacked_cols), dtype=common_dtype, device=A.device())
    AB_stacked_data[:m_A, :] = A.astype(common_dtype)
    AB_stacked_data[m_A:, :] = B.astype(common_dtype)
    
    try:
        Q, R_qr = linalg.Qr(AB_stacked_data)
        Qa = Q[:m_A, :]      
        Qb = Q[m_A:m_A+n_B_rows, :]

        logger.debug(f"mat_AB1 (QR): Qb shape: {Qb.shape()}, content:\n{Qb.numpy()}") # DEBUG added
        if Qb.shape()[0] != Qb.shape()[1] or Qb.shape()[0] == 0 : 
             raise RuntimeError(f"Qb (shape {Qb.shape()}) is not square or is empty, cannot invert.")
        
        Qb_inv: Tensor
        # --- START TEMPORARY WORKAROUND for cytnx.linalg.Inv issue ---
        # Original algorithm: Qb_inv = linalg.Inv(Qb, clip=1e-14)
        # Reason for workaround: cytnx.linalg.Inv was observed to return zero for non-zero [1,1] tensors.
        if Qb.shape() == [1, 1]:
            # 手動求逆 1x1 矩陣以避免潛在的 cytnx.linalg.Inv 問題。
            # 如果 cytnx.linalg.Inv 函式庫修復了對 [1,1] 矩陣的正確求逆，此處應改回原始算法。
            val = Qb.item() # Extract scalar value
            if abs(val) < 1e-14: # Check for near-zero value, consistent with clip tolerance
                logger.warning(f"mat_AB1 (QR): Qb value {val} is near zero. Returning zero inverse for this element.")
                Qb_inv = cytnx.zeros((1,1), dtype=Qb.dtype(), device=Qb.device())
            else:
                # Create a new 1x1 tensor with the inverse value
                Qb_inv = cytnx.from_numpy(np.array([[1.0 / val]], dtype=Qb.numpy().dtype)).to(Qb.device()).astype(Qb.dtype())
            logger.debug(f"mat_AB1 (QR - Manual Qb_inv): Qb_inv shape: {Qb_inv.shape()}, content:\n{Qb_inv.numpy()}") # DEBUG
        else:
            # Fallback to cytnx.linalg.Inv for larger matrices or if direct inversion is not for [1,1]
            Qb_inv = linalg.Inv(Qb, clip=1e-14) 
            logger.debug(f"mat_AB1 (QR - cytnx.linalg.Inv): Qb_inv shape: {Qb_inv.shape()}, content:\n{Qb_inv.numpy()}") # DEBUG
        # --- END TEMPORARY WORKAROUND ---

        result = Qa @ Qb_inv
        logger.debug(f"mat_AB1 (QR): Result shape {result.shape()}, content:\n{result.numpy()}") # DEBUG
        return result.astype(A.dtype()) 
        
    except RuntimeError as e:
        logger.warning(f"mat_AB1: QR-based method failed ({e}). Falling back to A @ Inv(B) or pseudo-inverse.")
        try:
            # Original algorithm: return A @ linalg.Inv(B, clip=1e-14)
            # Reason for workaround: 同上，若 B 為 [1,1] 且有 cytnx.linalg.Inv 問題，此處需注意。
            B_inv: Tensor
            if B.shape() == [1,1]:
                val_B = B.item()
                if abs(val_B) < 1e-14:
                    B_inv = cytnx.zeros((1,1), dtype=B.dtype(), device=B.device())
                else:
                    B_inv = cytnx.from_numpy(np.array([[1.0 / val_B]], dtype=B.numpy().dtype)).to(B.device()).astype(B.dtype())
            else:
                B_inv = linalg.Inv(B, clip=1e-14)
            return A @ B_inv

        except RuntimeError:
            logger.warning("mat_AB1: Direct Inv(B) failed. Attempting pseudo-inverse via Svd.")
            Ub, Sb_vec, Vhb = linalg.Svd(B.contiguous(), is_U=True, is_Vt=True)
            Sb_pinv_vec = cytnx.zeros_like(Sb_vec)
            tol_pinv = 1e-14 * Sb_vec[0].item() if Sb_vec.shape()[0] > 0 else 1e-14
            for i in range(Sb_vec.shape()[0]):
                if abs(Sb_vec[i].item()) > tol_pinv:
                    Sb_pinv_vec[i] = 1.0 / Sb_vec[i].item()
            
            S_pinv_mat = cytnx.zeros((Vhb.shape()[0], Ub.shape()[0]), dtype=Sb_vec.dtype(), device=Sb_vec.device())
            min_dim_s = min(S_pinv_mat.shape())
            for i in range(min_dim_s):
                 if i < Sb_pinv_vec.shape()[0]:
                    S_pinv_mat[i,i] = Sb_pinv_vec[i]
            
            # Vhb is Vt, so V = Vt.permute(1,0). For B_pinv = V S_pinv Ut
            V_ct = Vhb.contiguous().permute(1, 0) 
            U_ct_T = Ub.contiguous().permute(1,0)
            B_pinv = V_ct @ S_pinv_mat @ U_ct_T
            return A @ B_pinv

def mat_A1B(A: Tensor, B: Tensor) -> Tensor:
    """
    Computes A^-1 @ B in a stable way using QR decomposition, mimicking C++ xfac::mat_A1B.
    A: Tensor of shape (m, m) (must be square and invertible)
    B: Tensor of shape (m, n)
    Returns A^-1 @ B, shape (m, n)
    """
    logger.debug(f"mat_A1B called: A shape {A.shape()}, B shape {B.shape()}") # DEBUG
    if not (isinstance(A, Tensor) and isinstance(B, Tensor)):
        raise TypeError("Inputs A and B must be cytnx.Tensors.")
    if A.shape()[0] != A.shape()[1]:
        logger.error(f"mat_A1B: Matrix A (shape {A.shape()}) must be square and invertible for QR method.")
        return linalg.Lstsq(A.contiguous(), B.contiguous(), clip=1e-14)
        
    if A.shape()[1] != B.shape()[0]:
        raise ValueError(
            f"A.shape[1] ({A.shape()[1]}) must match B.shape[0] ({B.shape()[0]}) for A^-1 @ B."
        )

    B_contig = B.contiguous()
    B_T = B_contig.permute(1,0)  
    A_contig = A.contiguous()
    A_T = A_contig.permute(1,0)  
    
    n_BT, m_BT = B_T.shape() 
    n_AT, m_AT = A_T.shape() 

    stacked_rows = n_BT + n_AT
    stacked_cols = m_BT 
    
    common_dtype = A.dtype()
    if A.dtype() != B.dtype():
        if A.dtype() %2 == 1 or B.dtype() %2 == 1: 
            common_dtype = Type.ComplexDouble if A.dtype() > Type.ComplexFloat or B.dtype() > Type.ComplexFloat else Type.ComplexFloat
        elif A.dtype() == Type.Double or B.dtype() == Type.Double:
            common_dtype = Type.Double

    BA_T_stacked_data = cytnx.zeros((stacked_rows, stacked_cols), dtype=common_dtype, device=A.device())
    BA_T_stacked_data[:n_BT, :] = B_T.astype(common_dtype)
    BA_T_stacked_data[n_BT:, :] = A_T.astype(common_dtype)
    
    try:
        Q, R_qr = linalg.Qr(BA_T_stacked_data)
        Qb_from_Q = Q[:n_BT, :]      
        Qa_from_Q = Q[n_BT:n_BT+n_AT, :]

        logger.debug(f"mat_A1B (QR): Qa_from_Q shape: {Qa_from_Q.shape()}, content:\n{Qa_from_Q.numpy()}") # DEBUG added
        if Qa_from_Q.shape()[0] != Qa_from_Q.shape()[1] or Qa_from_Q.shape()[0] == 0:
            raise RuntimeError(f"Qa_from_Q (shape {Qa_from_Q.shape()}) is not square or is empty, cannot invert.")

        InvQa: Tensor
        # --- START TEMPORARY WORKAROUND for cytnx.linalg.Inv issue ---
        # Original algorithm: InvQa = linalg.Inv(Qa_from_Q, clip=1e-14)
        # Reason for workaround: cytnx.linalg.Inv was observed to return zero for non-zero [1,1] tensors.
        if Qa_from_Q.shape() == [1, 1]:
            # 手動求逆 1x1 矩陣以避免潛在的 cytnx.linalg.Inv 問題。
            # 如果 cytnx.linalg.Inv 函式庫修復了對 [1,1] 矩陣的正確求逆，此處應改回原始算法。
            val = Qa_from_Q.item() # Extract scalar value
            if abs(val) < 1e-14: # Check for near-zero value, consistent with clip
                logger.warning(f"mat_A1B (QR): Qa_from_Q value {val} is near zero. Returning zero inverse for this element.")
                InvQa = cytnx.zeros((1,1), dtype=Qa_from_Q.dtype(), device=Qa_from_Q.device())
            else:
                # Create a new 1x1 tensor with the inverse value
                InvQa = cytnx.from_numpy(np.array([[1.0 / val]], dtype=Qa_from_Q.numpy().dtype)).to(Qa_from_Q.device()).astype(Qa_from_Q.dtype())
            logger.debug(f"mat_A1B (QR - Manual InvQa): InvQa shape: {InvQa.shape()}, content:\n{InvQa.numpy()}") # DEBUG
        else:
            # Fallback to cytnx.linalg.Inv for larger matrices or if direct inversion is not for [1,1]
            InvQa = linalg.Inv(Qa_from_Q, clip=1e-14)
            logger.debug(f"mat_A1B (QR - cytnx.linalg.Inv): InvQa shape: {InvQa.shape()}, content:\n{InvQa.numpy()}") # DEBUG
        # --- END TEMPORARY WORKAROUND ---
        
        InvQa_T = InvQa.permute(1,0)
        Qb_from_Q_T = Qb_from_Q.permute(1,0)
        result = InvQa_T @ Qb_from_Q_T 
        logger.debug(f"mat_A1B (QR): Result shape {result.shape()}, content:\n{result.numpy()}") # DEBUG
        return result.astype(A.dtype())

    except RuntimeError as e:
        logger.warning(f"mat_A1B: QR-based method failed ({e}). Falling back to linalg.Lstsq(A, B).")
        return linalg.Lstsq(A.contiguous(), B.contiguous(), clip=1e-14)