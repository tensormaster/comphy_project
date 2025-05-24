# tensor_utils.py (或者 tensor_train.py 或 tensor_ci.py 內部)
import cytnx
from cytnx import UniTensor, Tensor, BD_IN, BD_OUT, linalg, Type # Ensure Type is imported
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

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
        
    # Ensure data is contiguous for predictable reshape after get_block()
    # get_block() returns a Tensor whose shape matches the bond dimensions in order.
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
    if not (isinstance(A, Tensor) and isinstance(B, Tensor)):
        raise TypeError("Inputs A and B must be cytnx.Tensors.")
    if A.shape()[1] != B.shape()[0]: # A_cols must match B_rows for A @ B_inv
        raise ValueError(
            f"A.shape[1] ({A.shape()[1]}) must match B.shape[0] ({B.shape()[0]}) for A @ B^-1."
        )
    if B.shape()[0] != B.shape()[1]:
        logger.error(f"mat_AB1: Matrix B (shape {B.shape()}) is not square. "
                       "QR method for A @ B^-1 typically assumes B is square and invertible. "
                       "Numerical stability issues or incorrect results may occur. Consider linalg.Lstsq for non-square B.")
        # For non-square B, A @ B_pinv would be linalg.Lstsq(B.T(), A.T()).T()
        # However, to match C++ logic which seems to expect square B for inv(Qb):
        # We will proceed but results might be unreliable if B is not invertible.
        # A robust solution might involve SVD based pseudo-inverse if Inv fails.
        # For now, let's try direct inverse if QR path fails.

    m_A, n_A = A.shape() # n_A is common dimension
    n_B_rows, n_B_cols = B.shape() # n_B_rows == n_A, n_B_cols == n_A for square B

    # Stack [A; B] vertically. AB_stacked has shape (m_A + n_B_rows, n_A)
    # This requires A and B to have the same number of columns (n_A)
    if A.shape()[1] != B.shape()[1]:
         raise ValueError(f"Column count of A ({A.shape()[1]}) must match column count of B ({B.shape()[1]}) for C++ like stacking in mat_AB1.")

    # Manual Vstack
    stacked_rows = m_A + n_B_rows
    stacked_cols = n_A # = B.shape()[1]
    
    # Ensure compatible dtypes before creating zeros
    common_dtype = A.dtype()
    if A.dtype() != B.dtype(): # Promote to complex if one is complex, or to double if one is double
        if A.dtype() %2 == 1 or B.dtype() %2 == 1: # one is complex
            common_dtype = Type.ComplexDouble if A.dtype() > Type.ComplexFloat or B.dtype() > Type.ComplexFloat else Type.ComplexFloat
        elif A.dtype() == Type.Double or B.dtype() == Type.Double:
            common_dtype = Type.Double
            
    AB_stacked_data = cytnx.zeros((stacked_rows, stacked_cols), dtype=common_dtype, device=A.device())
    AB_stacked_data[:m_A, :] = A.astype(common_dtype)
    AB_stacked_data[m_A:, :] = B.astype(common_dtype)
    
    try:
        Q, R_qr = linalg.Qr(AB_stacked_data)
        # Q shape: (stacked_rows, min(stacked_rows, stacked_cols))
        # R_qr shape: (min(stacked_rows, stacked_cols), stacked_cols)
        # For inv(Qb), Qb must be square. Qb is Q[m_A:, :n_B_rows] (using B's row dim for slicing Q)
        # If B is square (n_B_rows x n_B_rows), then stacked_cols = n_B_rows.
        # Q is (m_A + n_B_rows, n_B_rows).
        # R_qr is (n_B_rows, n_B_rows)
        
        Qa = Q[:m_A, :]      # Shape: (m_A, n_B_rows)
        Qb = Q[m_A:m_A+n_B_rows, :] # Shape: (n_B_rows, n_B_rows)

        if Qb.shape()[0] != Qb.shape()[1] or Qb.shape()[0] == 0 : # Qb must be square and non-empty for Inv
             raise RuntimeError(f"Qb (shape {Qb.shape()}) is not square or is empty, cannot invert.")
             
        Qb_inv = linalg.Inv(Qb)
        result = Qa @ Qb_inv
        return result.astype(A.dtype()) # Cast back to original type of A if common_dtype promoted it
        
    except RuntimeError as e:
        logger.warning(f"mat_AB1: QR-based method failed ({e}). Falling back to A @ Inv(B) or pseudo-inverse.")
        try:
            B_inv = linalg.Inv(B) # Try direct inverse
            return A @ B_inv
        except RuntimeError:
            logger.warning("mat_AB1: Direct Inv(B) failed. Attempting pseudo-inverse via Svd.")
            # Fallback: A @ B_pinv using SVD
            # B = U S Vh. B_pinv = V S_pinv Uh
            # Ensure B is contiguous for SVD
            Ub, Sb_vec, Vhb = linalg.Svd(B.contiguous(), is_U=True, is_Vt=True)
            Sb_pinv_vec = cytnx.zeros_like(Sb_vec)
            tol_pinv = 1e-14 * Sb_vec[0].item() if Sb_vec.shape()[0] > 0 else 1e-14
            for i in range(Sb_vec.shape()[0]):
                if abs(Sb_vec[i].item()) > tol_pinv:
                    Sb_pinv_vec[i] = 1.0 / Sb_vec[i].item()
            
            # Construct S_pinv matrix (Sb_vec.shape[0] x Ub.shape()[1] or Sb_vec.shape[0] x Sb_vec.shape[0])
            # If B is kxn, U is kxk, S is kxn (diag part), Vh is nxn
            # B_pinv should be nxk
            S_pinv_mat = cytnx.zeros((Vhb.shape()[0], Ub.shape()[0]), dtype=Sb_vec.dtype(), device=Sb_vec.device())
            min_dim_s = min(S_pinv_mat.shape())
            for i in range(min_dim_s):
                 if i < Sb_pinv_vec.shape()[0]:
                    S_pinv_mat[i,i] = Sb_pinv_vec[i]

            B_pinv = Vhb.contiguous().T() @ S_pinv_mat @ Ub.contiguous().T()
            return A @ B_pinv


def mat_A1B(A: Tensor, B: Tensor) -> Tensor:
    """
    Computes A^-1 @ B in a stable way using QR decomposition, mimicking C++ xfac::mat_A1B.
    A: Tensor of shape (m, m) (must be square and invertible)
    B: Tensor of shape (m, n)
    Returns A^-1 @ B, shape (m, n)
    """
    if not (isinstance(A, Tensor) and isinstance(B, Tensor)):
        raise TypeError("Inputs A and B must be cytnx.Tensors.")
    if A.shape()[0] != A.shape()[1]:
        logger.error(f"mat_A1B: Matrix A (shape {A.shape()}) must be square and invertible for QR method.")
        # Fallback to linalg.Lstsq which solves AX=B
        return linalg.Lstsq(A.contiguous(), B.contiguous())
        
    if A.shape()[1] != B.shape()[0]: # A_cols must match B_rows for A_inv @ B
        raise ValueError(
            f"A.shape[1] ({A.shape()[1]}) must match B.shape[0] ({B.shape()[0]}) for A^-1 @ B."
        )

    # C++ logic: BA_T_stacked = vstack(B.T(), A.T())
    B_T = B.contiguous().T()  # Shape: (n, m) where m is A's dim
    A_T = A.contiguous().T()  # Shape: (m, m)
    
    n_BT, m_BT = B_T.shape() # m_BT = m
    n_AT, m_AT = A_T.shape() # n_AT = m, m_AT = m

    # Manual Vstack
    stacked_rows = n_BT + n_AT
    stacked_cols = m_BT # = m_AT = m
    
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
        # Q shape: (stacked_rows, min(stacked_rows, stacked_cols)) -> (n+m, m)
        # R_qr shape: (min, stacked_cols) -> (m,m)
        
        # Qb_from_Q comes from B_T part of Q
        Qb_from_Q = Q[:n_BT, :]      # Shape: (n_BT, m) -> (n, m)
        # Qa_from_Q comes from A_T part of Q
        Qa_from_Q = Q[n_BT:n_BT+n_AT, :] # Shape: (n_AT, m) -> (m, m)

        if Qa_from_Q.shape()[0] != Qa_from_Q.shape()[1] or Qa_from_Q.shape()[0] == 0: # Qa must be square and non-empty for Inv
            raise RuntimeError(f"Qa_from_Q (shape {Qa_from_Q.shape()}) is not square or is empty, cannot invert.")

        InvQa = linalg.Inv(Qa_from_Q)
        result = InvQa.T() @ Qb_from_Q.T() # (m x m) @ (m x n) -> (m x n)
        return result.astype(A.dtype())

    except RuntimeError as e:
        logger.warning(f"mat_A1B: QR-based method failed ({e}). Falling back to linalg.Lstsq(A, B).")
        return linalg.Lstsq(A.contiguous(), B.contiguous()) # Solves AX = B for X