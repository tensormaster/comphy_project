import pickle
import numpy as np
from itertools import product
import logging
from typing import List, Any, Union

# Initialize logger
logger = logging.getLogger(__name__)

# Cytnx core imports
from cytnx import Tensor, UniTensor, Type, Device, from_numpy, zeros, eye, linalg
# Due to the Jupyter environment, the code can run without explicitly importing it.
from mat_decomp import MatprrLUFixedTol

# --- Helper Functions ---
def _assert_cube(A: Tensor):
    """
    确认传入的张量 A 是 3 维的“立方体”张量。
    如果维度不是 3，则抛出错误。
    """
    if len(A.shape()) != 3:
        raise ValueError(f"Expected a 3D tensor (cube), got shape {A.shape()}")


def cube_as_matrix1(A: Tensor) -> Tensor:
    _assert_cube(A)
    r, c, s = A.shape()
    return A.reshape(r, c * s)


def cube_as_matrix2(A: Tensor) -> Tensor:
    _assert_cube(A)
    r, c, s = A.shape()
    return A.reshape(r * c, s)

# --- Main TensorTrain Class ---
class TensorTrain:
    def __init__(self, M: List[Tensor] = None):
        self.M: List[Tensor] = M if M is not None else []
    
    def eval(self, idx: List[int]) -> Any:
        if len(idx) != len(self.M):
            raise ValueError(
                f"Index length {len(idx)} does not match tensor train length {len(self.M)}."
            )
        prod = from_numpy(np.eye(1))
        for k, Mk in enumerate(self.M):
            _assert_cube(Mk)
            r_k, d_k, r_k1 = Mk.shape()
            i_k = idx[k]
            if i_k < 0 or i_k >= d_k:
                raise IndexError(
                    f"Index {i_k} out of bounds for dimension {d_k} at site {k}."
                )
            col = Mk[:, i_k, :].reshape(r_k, r_k1)
            prod = prod @ col
        if list(prod.shape()) != [1, 1]:
            raise ValueError(
                f"TensorTrain.eval did not collapse to scalar, got shape {prod.shape()}"
            )
        return prod[0, 0].item()

    __call__ = eval

    def overlap(self, tt: 'TensorTrain') -> float:
        if len(self.M) != len(tt.M):
            raise ValueError("Tensor trains must have the same length.")
        C = from_numpy(np.eye(1))
        for Ak, Bk in zip(self.M, tt.M):
            _assert_cube(Ak); _assert_cube(Bk)
            r_next, s_next = Ak.shape()[2], Bk.shape()[2]
            C_next = from_numpy(np.zeros((r_next, s_next), dtype=C.numpy().dtype))
            for p in range(Ak.shape()[1]):
                A_slice_H = Ak[:, p, :].permute(1, 0).Conj()
                C_next += A_slice_H @ C @ Bk[:, p, :]
            C = C_next
        if C.shape() != [1, 1]:
            raise ValueError("Inner-product did not collapse to scalar.")
        return float(C.item())

    def norm2(self) -> float:
        return self.overlap(self)

    def compressLU(self, reltol: float = 1e-12, maxBondDim: int = 0):
        decomp = MatprrLUFixedTol(tol=reltol, rankMax=maxBondDim)
        n = len(self.M)
        if n < 2:
            return
        for i in range(n - 1):
            A_mat = cube_as_matrix2(self.M[i])
            # 這裡改成 L_out, R_out 而不先叫 L_np, R_np
            L_out, R_out = decomp(A_mat)
            # 僅在必要時轉回 cytnx.Tensor
            from cytnx import Tensor
            if isinstance(L_out, Tensor):
                L = L_out
            else:
                L = from_numpy(L_out)
            if isinstance(R_out, Tensor):
                R = R_out
            else:
                R = from_numpy(R_out)

            r, c, _ = self.M[i].shape()
            new_bond = L.shape()[1]
            self.M[i] = L.reshape(r, c, new_bond)
            B_mat = cube_as_matrix1(self.M[i + 1])
            C_mat = R @ B_mat
            _, c_next, s_next = self.M[i + 1].shape()
            self.M[i + 1] = C_mat.reshape(new_bond, c_next, s_next)

    # Placeholder for other compress methods
    def compressSVD(self, maxBondDim: int = 0, tol: float = 1e-12):
        pass

    def compressCI(self, rank: int = 0):
        pass

    def __add__(self, other: 'TensorTrain') -> 'TensorTrain':
        if not self.M:
            return other
        if not other.M:
            return self
        if len(self.M) != len(other.M):
            raise ValueError("Cannot add tensor trains of different lengths.")
        new_cores = []
        for A, B in zip(self.M, other.M):
            r1, d1, s1 = A.shape()
            r2, _, s2 = B.shape()
            new_core = zeros((r1+r2, d1, s1+s2), dtype=A.dtype(), device=A.device())
            for p in range(d1):
                new_core[0:r1, p, 0:s1] = A[:, p, :]
                new_core[r1:r1+r2, p, s1:s1+s2] = B[:, p, :]
            new_cores.append(new_core)
        return TensorTrain(new_cores)

    def trueError(self, other: 'TensorTrain', max_eval: int = int(1e6)) -> float:
        if len(self.M) != len(other.M):
            raise ValueError("Tensor trains must have the same length.")
        dims = [Mk.shape()[1] for Mk in self.M]
        total = np.prod(dims)
        if total > max_eval:
            raise ValueError(f"Too many index combinations ({total}), exceed max_eval={max_eval}")
        max_err = 0.0
        for idx in product(*[range(d) for d in dims]):
            v1 = self.eval(idx)
            v2 = other.eval(idx)
            err = abs(v1 - v2)
            if err > max_err:
                max_err = err
        return max_err

    def sum(self, weights: List[List[float]]) -> float:
        if len(weights) != len(self.M):
            raise ValueError("Weights length must match number of cores.")
        prod = from_numpy(np.eye(1))
        for k, Mk in enumerate(self.M):
            _assert_cube(Mk)
            w = weights[k]
            dim = Mk.shape()[1]
            if len(w) != dim:
                raise ValueError(f"Weights[{k}] length {len(w)} does not match physical dim {dim}.")
            weighted = Mk[:, 0, :] * w[0]
            for j in range(1, dim):
                weighted = weighted + Mk[:, j, :] * w[j]
            prod = prod @ weighted
        return prod.item()

    def sum1(self) -> float:
        weights = [[1.0] * Mk.shape()[1] for Mk in self.M]
        return self.sum(weights)

    def save(self, file_name: str):
        arrs = [Mk.numpy() for Mk in self.M]
        with open(file_name, 'wb') as f:
            pickle.dump(arrs, f)

    @staticmethod
    def load(file_name: str) -> 'TensorTrain':
        with open(file_name, 'rb') as f:
            arrs = pickle.load(f)
        M = [from_numpy(a) for a in arrs]
        return TensorTrain(M)


# --- Testing Main Program ---
if __name__ == "__main__":
    import numpy as _np
    _np.random.seed(128)
    # Define physical dimensions for a small MPS
    phys_dims = [3, 4, 5]
    # Build a random TensorTrain of trivial bond dims (1)
    cores = []
    for d in phys_dims:
        arr = _np.random.rand(1, d, 1)
        cores.append(from_numpy(arr))
    original_tt = TensorTrain(cores.copy())
    # Make a deep copy for compression
    to_compress = TensorTrain(cores.copy())
    # Compress the TT using LU-based method
    to_compress.compressLU(reltol=1e-6)
    # Compute maximum true error across all entries
    error = to_compress.trueError(original_tt, max_eval=int(1e4))
    print(f"Maximum reconstruction error after compressLU: {error:.3e}")
    # Also print norm before and after
    print(f"Original norm^2: {original_tt.norm2():.3e}")
    print(f"Compressed norm^2: {to_compress.norm2():.3e}")
