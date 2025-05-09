# filename: tensor_train.py

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
# Import mat_decomp wrapper
# from mat_decomp import MatprrLUFixedTol

# --- Helper Functions ---
def _assert_cube(A: Tensor):
    """
    确认传入的张量 A 是 3 维的“立方体”张量。
    如果维度不是 3，则抛出错误。
    """
    if len(A.shape()) != 3:
        raise ValueError(f"Expected a 3D tensor (cube), got shape {A.shape()}")


def cube_as_matrix1(A: Tensor) -> Tensor:
    """
    将 3D 张量 A(i,j,k) 展平为矩阵 B(i, j*k)：
      - 保持第一个维度 i 不动
      - 将 j 和 k 两个维度合并到一起
    常见于从左向右分解时需要将一个核心视作矩阵。
    """
    _assert_cube(A)
    r, c, s = A.shape()
    return A.reshape(r, c * s)


def cube_as_matrix2(A: Tensor) -> Tensor:
    """
    将 3D 张量 A(i,j,k) 展平为矩阵 B(i*j, k)：
      - 将第一个和第二个维度合并
      - 保持第三个物理维度 k 不动
    常见于从右向左分解时需要将一个核心视作矩阵。
    """
    _assert_cube(A)
    r, c, s = A.shape()
    return A.reshape(r * c, s)

# --- Main TensorTrain Class ---
class TensorTrain:
    def __init__(self, M: List[Tensor] = None):
        """
        初始化 TensorTrain：
          M: 核心张量列表，每个 Tensor 是一个形状为 (r_k, d_k, r_{k+1}) 的三维张量
        如果未提供 M，则默认创建一个空的 TensorTrain。
        """
        self.M: List[Tensor] = M if M is not None else []
    
    def eval(self, idx: List[int]) -> Any:
        """
        计算给定多重索引 idx 对应的标量值：
        1. 检查 idx 长度是否与核心数一致。
        2. 初始化一个 unitensor（1×1 Tensor） prod，用于累积中间结果。
        3. 依次从每个核心中抽取对应物理索引的矩阵切片 col。
        4. 使用矩阵乘 prod = prod @ col，不断更新 unitensor。
        5. 最终 prod 应为 1×1 张量，将其转换为 Python 标量返回。

        unitensor 用于封装标量运算过程中的中间结果，最后提取其唯一元素。
        """
        if len(idx) != len(self.M):
            raise ValueError(
                f"Index length {len(idx)} does not match tensor train length {len(self.M)}."
            )
        prod = from_numpy(np.eye(1))
        logger.info(f"Initial prod shape: {list(prod.shape())}")
        for k, Mk in enumerate(self.M):
            _assert_cube(Mk)
            r_k, d_k, r_k1 = Mk.shape()
            i_k = idx[k]
            logger.info(f"Core {k}: shape={Mk.shape()}, idx={i_k}")
            if i_k < 0 or i_k >= d_k:
                raise IndexError(
                    f"Index {i_k} out of bounds for dimension {d_k} at site {k}."
                )
            col = Mk[:, i_k, :].reshape(r_k, r_k1)
            logger.info(f"Slice at core {k} shape: {list(col.shape())}")
            prod = prod @ col
            logger.info(f"After multiply step {k}, prod shape: {list(prod.shape())}")
        logger.info(f"Final prod shape before scalar extraction: {list(prod.shape())}")
        if list(prod.shape()) != [1, 1]:
            raise ValueError(
                f"TensorTrain.eval did not collapse to scalar, got shape {prod.shape()}"
            )
        result = prod[0, 0].item()
        logger.info(f"Eval result: {result}")
        return result

    __call__ = eval

    def overlap(self, tt: 'TensorTrain') -> float:
        """
        计算两个 TensorTrain 的内积（overlap）：
        在每个站点做双核心张量收缩，最终得到 unitensor 并提取标量。
        """
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
        """返回自身重叠，即范数平方。"""
        return self.overlap(self)

    def compressLU(self, reltol: float = 1e-12, maxBondDim: int = 0):
        """
        使用 LU 分解方法压缩每个核心的虚拟维度。
        需要将核心视为矩阵才能进行 MatprrLUFixedTol 分解。
        """
        decomp = MatprrLUFixedTol(tol=reltol, rankMax=maxBondDim)
        n = len(self.M)
        if n < 2:
            return
        for i in range(n - 1):
            A_mat = cube_as_matrix2(self.M[i])
            L_np, R_np = decomp(A_mat)
            L = from_numpy(L_np); R = from_numpy(R_np)
            r, c, _ = self.M[i].shape()
            new_bond = L.shape()[1]
            self.M[i] = L.reshape(r, c, new_bond)
            B_mat = cube_as_matrix1(self.M[i + 1])
            C_mat = R @ B_mat
            _, c_next, s_next = self.M[i + 1].shape()
            self.M[i + 1] = C_mat.reshape(new_bond, c_next, s_next)

    def compressSVD(self, maxBondDim: int = 0, tol: float = 1e-12):
       #These tasks are of lower priority and only require attention after other tasks are completed.
       
        pass

    def compressCI(self, rank: int = 0):
       #These tasks are of lower priority and only require attention after other tasks are completed.
       
        pass

    def __add__(self, other: 'TensorTrain') -> 'TensorTrain':
        """
        定义两个 TensorTrain 在虚拟维度上的块对角拼接加法。
        """
        if not self.M:
            return other
        if not other.M:
            return self
        if len(self.M) != len(other.M):
            raise ValueError("Cannot add tensor trains of different lengths.")
        new_cores = []
        for A, B in zip(self.M, other.M):
            r1, d1, s1 = A.shape()
            r2, d2, s2 = B.shape()
            if d1 != d2:
                raise ValueError(f"Physical dimension mismatch: {d1} vs {d2}")
            dtype = A.dtype()
            device = A.device()
            new_core = zeros((r1+r2, d1, s1+s2), dtype=dtype, device=device)
            for p in range(d1):
                new_core[0:r1, p, 0:s1] = A[:, p, :]
                new_core[r1:r1+r2, p, s1:s1+s2] = B[:, p, :]
            new_cores.append(new_core)
        return TensorTrain(new_cores)

    def trueError(self, other: 'TensorTrain', max_eval: int = int(1e6)) -> float:
        """
        对比所有索引下的值，返回最大误差。
        """
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
        """
        对每个核心按物理索引进行加权，然后累积至 unitensor。
        """
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
        """
        等价于所有物理分量权重均为 1 的 sum。
        """
        weights = [[1.0] * Mk.shape()[1] for Mk in self.M]
        return self.sum(weights)

    def save(self, file_name: str):
        """Save the tensor train to a file using pickle."""
        arrs = [Mk.numpy() for Mk in self.M]
        with open(file_name, 'wb') as f:
            pickle.dump(arrs, f)

    @staticmethod
    def load(file_name: str) -> 'TensorTrain':
        """
        从 pickle 文件加载 TensorTrain。
        """
        with open(file_name, 'rb') as f:
            arrs = pickle.load(f)
        M = [from_numpy(a) for a in arrs]
        return TensorTrain(M)
