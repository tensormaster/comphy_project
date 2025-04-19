import cytnx
from cytnx import *
import numpy as np
from IndexSet import IndexSet
from Qgrid import QuanticGrid
from AdaptiveLU import *
from IndexSet import *
from itertools import *
from crossdata import *


# 假設 CrossData, IndexSet, UniTensor 均已正確匯入
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def contract_last_first(t1, t2):
    """
    將張量 t1 (形狀為 (a1, a2, ..., r)) 與 t2 (形狀為 (r, b1, b2, ..., b_k))
    沿著 t1 的最後一個軸與 t2 的第一個軸做收縮，
    回傳形狀為 (a1, a2, ..., b1, b2, ..., b_k) 的張量。
    """
    shape1 = t1.shape()  # 例如 (a1, a2, ..., r)
    shape2 = t2.shape()  # 例如 (r, b1, b2, ..., b_k)
    if shape1[-1] != shape2[0]:
        raise ValueError(f"Contract dimensions do not match: {shape1[-1]} vs {shape2[0]}")
    # 計算 t1 除最後一軸外的元素總數
    m = 1
    for s in shape1[:-1]:
        m *= s
    # t1 最後一軸長度 r
    r = shape1[-1]
    # 計算 t2 除第一軸外的元素總數
    n2 = 1
    for s in shape2[1:]:
        n2 *= s
    # 將 t1 reshape 為 (m, r)
    t1_mat = t1.reshape(m, r)
    # 將 t2 reshape 為 (r, n2)
    t2_mat = t2.reshape(r, n2)
    # 進行矩陣乘法（注意 cytnx.Tensor 支援 @ 運算符）
    res_mat = t1_mat @ t2_mat
    # 新張量形狀為 t1 除最後一軸加上 t2 除第一軸
    new_shape = list(shape1[:-1]) + list(shape2[1:])
    # 將乘法結果 reshape 回新形狀
    res_tensor = res_mat.reshape(*new_shape)
    return res_tensor

def check_nan_in_tensor(core):
    if core is None:
        raise ValueError("核心張量為 None，請檢查分解過程。")
    shape = core.shape()
    if not shape:
        return math.isnan(core.item())
    for idx in product(*[range(s) for s in shape]):
        try:
            if math.isnan(core[idx].item()):
                return True
        except Exception as e:
            raise RuntimeError(f"檢查張量元素 {idx} 時失敗：{e}")
    return False


def check_tensor_contents(t, name="Tensor"):
    try:
        arr = t.to_numpy()
    except Exception as e:
        raise ValueError(f"{name} 轉換成 numpy array 時失敗，可能內含非數值元素。") from e

    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        elem = it[0]
        if not isinstance(elem.item(), (int, float, np.number)):
            raise TypeError(f"{name} 在索引 {it.multi_index} 內的元素型態 {type(elem.item())} 不正確。")
        it.iternext()


class TCI_env:
    def __init__(self, tensor_data: 'Tensor', dim: int, M: int):
        """
        僅接受 tensor_data 作為來源的 TCI_env 類別。
        """
        if not isinstance(tensor_data, Tensor):
            raise TypeError("tensor_data 必須是 cytnx.Tensor 物件")
        if dim is None or M is None:
            raise ValueError("提供 tensor_data 時，必須指定 dim 與 M")
        self.dim = int(dim)
        self.M = int(M)
        self._tensor = tensor_data

    def get_tensor(self):
        return self._tensor

    def get_quantics_tensor(self):
        return UniTensor(self._tensor)


class TCI:
    def __init__(self, env, tolerance=1e-6, verbose=True, max_iter=50, pivot1=None):
        if not hasattr(env, "dim") or not hasattr(env, "M"):
            raise ValueError("TCI_env 物件必須具有 'dim' 與 'M' 屬性")
        self.dim = int(env.dim)
        self.M_value = int(env.M)
        self.env = env

        self.tolerance = tolerance
        self.verbose = verbose
        self.max_iter = max_iter
        self.localDim = [self.M_value for _ in range(self.dim)]

        self.pivot1 = [0] * self.dim if pivot1 is None else pivot1
        if len(self.pivot1) != self.dim:
            raise ValueError("初始 pivot 的長度必須與 dim 相同")
        logger.info(f"使用初始 pivot: {self.pivot1}")

        quantics_utensor = env.get_quantics_tensor()
        if not hasattr(quantics_utensor, "get_block_"):
            raise ValueError("env.get_quantics_tensor() 返回物件必須有 get_block_ 方法")
        self.tensor_data = quantics_utensor.get_block_()
        if not isinstance(self.tensor_data, Tensor):
            raise TypeError("get_block_() 返回的物件必須為 cytnx.Tensor")

        def safe_f(*idx):
            try:
                value = self.tensor_data[idx].item()
            except Exception as e:
                raise ValueError(f"無法讀取 tensor_data[{idx}]") from e
            if not isinstance(value, (int, float)):
                raise TypeError("f 返回值必須是數值型態")
            return value

        self.f = safe_f

        try:
            initial_value = self.f(*self.pivot1)
        except Exception as e:
            raise ValueError("計算初始 pivot 值失敗") from e
        self.pivotError = [abs(initial_value)]
        if self.pivotError[0] == 0.0:
            raise ValueError("初始 pivot f(pivot1) 為 0，不合法")

        self.localSet = [IndexSet() for _ in range(self.dim)]
        self.Iset = [IndexSet() for _ in range(self.dim)]
        self.Jset = [IndexSet() for _ in range(self.dim)]
        for p in range(self.dim):
            for i in range(self.localDim[p]):
                self.localSet[p].push_back((i,))
            self.Iset[p].push_back(tuple(self.pivot1[:p]))
            self.Jset[p].push_back(tuple(self.pivot1[p+1:]))

        self.pi_matrices = []
        self.pi_bool = []
        self.cross_data_list = []
        for p in range(self.dim - 1):
            Pi = self.buildPiAt(p)
            self.pi_matrices.append(Pi)
            cross_obj = CrossData(Pi.n_rows, Pi.n_cols)
            self.cross_data_list.append(cross_obj)

        self.T3 = [None] * self.dim
        self.P = [None] * self.dim
        for p in range(self.dim - 1):
            self.addPivot(p)
            if p == 0:
                self.T3[p] = self.buildCube(self.cross_data_list[p].C,
                                            len(self.Iset[p]), len(self.localSet[p]), len(self.Jset[p]))
            self.T3[p+1] = self.buildCube(self.cross_data_list[p].R,
                                          len(self.Iset[p+1]), len(self.localSet[p+1]), len(self.Jset[p+1]))
            self.P[p] = self.cross_data_list[p].pivotMat()
        self.P[-1] = self.identityMatrix(1)

        self.iterate(self.max_iter)

    def kron(self, index_set1, index_set2):
        result = IndexSet()
        for a, b in product(index_set1.get_all(), index_set2.get_all()):
            result.push_back(a + b)
        return result

    def buildPiAt(self, p):
        left_set = self.kron(self.Iset[p], self.localSet[p])
        right_set = self.kron(self.localSet[p+1], self.Jset[p+1])
        n_rows = len(left_set.get_all())
        n_cols = len(right_set.get_all())
        Pi_matrix = zeros((n_rows, n_cols), dtype=Type.Double, device=Device.cpu)
        for i, left_index in enumerate(left_set.get_all()):
            for j, right_index in enumerate(right_set.get_all()):
                full_index = left_index + right_index
                Pi_matrix[i, j] = self.f(*full_index)
        Pi = type("PiMatrix", (), {})()
        Pi.n_rows = n_rows
        Pi.n_cols = n_cols
        Pi.data = Pi_matrix
        return Pi

    def addPivot(self, p):
        A_tensor = self.pi_matrices[p].data
        if not isinstance(A_tensor, Tensor):
            raise TypeError("pi_matrices[p].data 必須是 cytnx.Tensor")
        shape_list = A_tensor.shape()
        n_rows, n_cols = shape_list
        max_val = 0.0
        pivot_i, pivot_j = 0, 0
        # 掃描所有元素，找出絕對值最大的 pivot
        for i in range(n_rows):
            for j in range(n_cols):
                try:
                    value = A_tensor[i, j].item()
                except Exception as e:
                    raise ValueError(f"無法讀取 tensor_data[{i}, {j}]") from e
                if abs(value) > max_val:
                    max_val = abs(value)
                    pivot_i, pivot_j = i, j
        # 顯示 pivot 的實際數值與其在原始 tensor 的位置
        pivot_value = A_tensor[pivot_i, pivot_j].item()
        logger.info(f"addPivot: 選擇 Pivot 值 {pivot_value}，位置為 ({pivot_i}, {pivot_j})")
        # 更新交叉資料
        self.cross_data_list[p].addPivot(pivot_i, pivot_j, A_tensor)


    def buildCube(self, data, d1, d2, d3):
        if data is None:
            raise ValueError("buildCube 接收到 None")
        expected_size = d1 * d2 * d3
        total_elements = np.prod(data.shape())
        if total_elements != expected_size:
            raise ValueError(f"Tensor 大小 {total_elements} ≠ 預期大小 {expected_size}")
        return data.reshape(d1, d2, d3)

    def identityMatrix(self, size):
        size_int = int(size)
        if size_int <= 0:
            raise ValueError("identityMatrix 參數需 > 0")
        return eye(size_int, dtype=Type.Double, device=Device.cpu)

    def iterate(self, nIter=1): 
        nIter = int(nIter)
        self.cIter = 0
        self.pivotErrorLastIter = [1.0] * (self.dim - 1)
        for t in range(nIter):
            self.cIter += 1
            logger.info(f"Starting iteration {self.cIter}")
            if self.cIter == 1:
                logger.info("First iteration skipped (initialization phase).")
                continue
            # 根據迭代次數決定處理 bond 的順序
            pivot_order = range(self.dim - 1) if self.cIter % 2 == 0 else reversed(range(self.dim - 1))
            for p in pivot_order:
                logger.info(f"Iteration {self.cIter}, processing bond index {p}.")
                self.addPivot(p)
                err = cytnx.linalg.Norm(self.pi_matrices[p].data).item()
                self.pivotErrorLastIter[p] = err
                logger.info(f"Iteration {self.cIter}, bond {p} error: {err:.6e}")
            max_err = max(self.pivotErrorLastIter)
            self.pivotError.append(max_err)
            logger.info(f"Iteration {self.cIter} completed. Max error: {max_err:.6e}")

    def get_TensorTrain(self, mode=0):
        tt = {"cores": self.T3, "pivots": self.P}
        for idx, core in enumerate(tt["cores"]):
            if core is None:
                logger.error(f"T3[{idx}] 為 None")
            if check_nan_in_tensor(core):
                raise ValueError(f"Tensor core {idx} 含有 NaN")
        return tt
    def reconstruct_tt(self):
        """
        從 TT 的核心還原出完整張量，使用 cytnx.Tensor 的運算。
        
        假設每個核心形狀為 (r_{k-1}, n_k, r_k)，且第一個與最後一個維度均為 1，
        還原後的張量形狀為 (n_1, n_2, ..., n_d)。
        """
        # 取第一個核心，若第一維為 1 則 squeeze 掉
        res = self.T3[0]
        shape = res.shape()
        if shape[0] == 1:
            res = res.reshape(*shape[1:])
        # 依序對剩下的核心進行收縮，每次收縮 t1 最後一軸與 t2 第一軸
        for core in self.T3[1:]:
            res = contract_last_first(res, core)
        # 如果最後一個維度為 1，則 squeeze 掉
        final_shape = res.shape()
        if final_shape[-1] == 1:
            res = res.reshape(*final_shape[:-1])
        return res

    def compute_tt_error(self, tol):
        """
        計算原始張量與 TT 近似張量之間的相對誤差，並與指定容忍值比較。
        
        參數:
          tol (float): 容忍的相對誤差
          
        回傳:
          float: 相對誤差，計算方式為 ||original - reconstructed|| / ||original||
        """
        reconstructed = self.reconstruct_tt()
        error_tensor = self.tensor_data - reconstructed
        error_norm = cytnx.linalg.Norm(error_tensor).item()
        original_norm = cytnx.linalg.Norm(self.tensor_data).item()
        rel_error = error_norm / original_norm if original_norm != 0 else error_norm

        if rel_error <= tol:
            print(f"相對誤差 {rel_error:.6e} 在容忍值 {tol:.6e} 之內。")
        else:
            print(f"相對誤差 {rel_error:.6e} 超出容忍值 {tol:.6e}。")
        return rel_error
    


