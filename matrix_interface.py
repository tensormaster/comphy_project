# matrix_interface.py

from cytnx import Tensor, zeros, Type
from IndexSet import * 


class IMatrix:
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def submat(self, rows: list[int], cols: list[int]) -> list:
        raise NotImplementedError

    def eval(self, ids: list[tuple[int, int]]) -> list:
        raise NotImplementedError

    def forget_row(self, i: int): pass
    def forget_col(self, j: int): pass


class MatDense(IMatrix):
    def __init__(self, data: Tensor):
        assert len(data.shape()) == 2, "Tensor must be 2D"
        super().__init__(data.shape()[0], data.shape()[1])
        self.data = data.clone()

    def submat(self, rows: list[int], cols: list[int]) -> list:
        sub_values = []
        for i in rows:
            for j in cols:
                sub_values.append(self.data[i, j].item())
        return sub_values

    def eval(self, ids: list[tuple[int, int]]) -> list:
        return [self.data[i, j].item() for i, j in ids]

    def set_rows(self, new_nrows: int, P: list[int], fnew) -> None:
        new_data = zeros((new_nrows, self.n_cols), dtype=Type.Double)
        for i, pi in enumerate(P):
            for j in range(self.n_cols):
                new_data[pi, j] = self.data[i, j]
        Pc = [i for i in range(new_nrows) if i not in P]
        for i in Pc:
            for j in range(self.n_cols):
                new_data[i, j] = fnew(i, j)
        self.data = new_data
        self.n_rows = new_nrows

    def set_cols(self, new_ncols: int, Q: list[int], fnew) -> None:
        new_data = zeros((self.n_rows, new_ncols), dtype=Type.Double)
        for j, qj in enumerate(Q):
            for i in range(self.n_rows):
                new_data[i, qj] = self.data[i, j]
        Qc = [j for j in range(new_ncols) if j not in Q]
        for j in Qc:
            for i in range(self.n_rows):
                new_data[i, j] = fnew(i, j)
        self.data = new_data
        self.n_cols = new_ncols


class IMatrixIndex(IMatrix):
    """
    支援任意 Index 的矩陣介面，對應 C++ 的 IMatrix<T,Index>。
    """
    def __init__(self, f, Iset_list: list, Jset_list: list):
        self.Iset = IndexSet(Iset_list)
        self.Jset = IndexSet(Jset_list)
        # 用 len() 取長度，避免呼叫不存在的 size()
        super().__init__(len(self.Iset), len(self.Jset))
        self.A = f

    def submat(self, rows: list[int], cols: list[int]) -> list:
        Ivals = self.Iset.from_int()
        Jvals = self.Jset.from_int()
        sub_values = []
        for i in rows:
            xi = Ivals[i]
            for j in cols:
                yj = Jvals[j]
                sub_values.append(self.A(xi, yj))
        return sub_values

    def eval(self, ids: list[tuple[int, int]]) -> list:
        Ivals = self.Iset.from_int()
        Jvals = self.Jset.from_int()
        return [self.A(Ivals[i], Jvals[j]) for (i, j) in ids]

    def set_rows(self, new_Iset: list) -> list[int]:
        old = self.Iset.from_int()
        new_set = IndexSet(new_Iset)
        # pos 與 C++ 一致：回傳 old 索引在 new_set 中的位置
        pos = new_set.pos(old)
        self.Iset = new_set
        self.n_rows = len(self.Iset)
        return pos

    def set_cols(self, new_Jset: list) -> list[int]:
        old = self.Jset.from_int()
        new_set = IndexSet(new_Jset)
        pos = new_set.pos(old)
        self.Jset = new_set
        self.n_cols = len(self.Jset)
        return pos


class MatDenseIndex(IMatrixIndex, MatDense):
    """
    同時擁有稠密存取與任意 Index 支援，對應 C++ 的 MatDense<T,Index>。
    """
    def __init__(self, f, Iset_list: list, Jset_list: list):
        # 先用 f 建稠密 data
        data = zeros((len(Iset_list), len(Jset_list)), dtype=Type.Double)
        for i, xi in enumerate(Iset_list):
            for j, yj in enumerate(Jset_list):
                data[i, j] = f(xi, yj)
        MatDense.__init__(self, data)
        IMatrixIndex.__init__(self, f, Iset_list, Jset_list)


class MatLazy(IMatrix):
    """
    延遲評估版本，對應 C++ 的 MatLazy<T>。
    """
    def __init__(self, f, n_rows: int, n_cols: int):
        super().__init__(n_rows, n_cols)
        self.f = f
        self._cache = {}

    def submat(self, rows: list[int], cols: list[int]) -> list:
        ids = [(i, j) for i in rows for j in cols]
        return self.eval(ids)

    def eval(self, ids: list[tuple[int, int]]) -> list:
        out = []
        for idx in ids:
            if idx in self._cache:
                out.append(self._cache[idx])
            else:
                val = self.f(idx[0], idx[1])
                self._cache[idx] = val
                out.append(val)
        return out

    def forget_row(self, i: int):
        for key in list(self._cache):
            if key[0] == i:
                del self._cache[key]

    def forget_col(self, j: int):
        for key in list(self._cache):
            if key[1] == j:
                del self._cache[key]

    def set_rows(self, new_nrows: int, P: list[int], fnew):
        self.n_rows = new_nrows
        self._cache.clear()
        self.f = fnew

    def set_cols(self, new_ncols: int, Q: list[int], fnew):
        self.n_cols = new_ncols
        self._cache.clear()
        self.f = fnew


def make_IMatrix(f, n_rows: int, n_cols: int, full: bool = False) -> IMatrix:
    """
    工廠函數：full=True 回傳 MatDense，否則 MatLazy。
    """
    if full:
        return MatDense(lambda i, j: f(i, j), n_rows, n_cols)
    else:
        return MatLazy(lambda i, j: f(i, j), n_rows, n_cols)
def test_matdense():
    # 建立一個 3×4 的矩陣，內容為 A[i,j] = i*10 + j
    data = zeros((3, 4), dtype=Type.Double)
    for i in range(3):
        for j in range(4):
            data[i, j] = i*10 + j
    mat = MatDense(data)
    print("原始 MatDense 矩陣：")
    print(mat.data)

    # 測試 submat
    vals = mat.submat([0, 2], [1, 3])
    print("submat([0,2], [1,3]) =>", vals)
    # 預期 [ A[0,1], A[0,3], A[2,1], A[2,3] ] = [1, 3, 21, 23]

    # 測試 eval
    pairs = [(1,2), (2,0)]
    print("eval", pairs, "=>", mat.eval(pairs))
    # 預期 [ A[1,2], A[2,0] ] = [12, 20]

    # 測試 set_rows：擴增到 5 列，保留原來 0→0,1→2,2→4，其餘用 fnew 填 -1
    def fnew_row(i, j): return -1.0
    mat.set_rows(5, [0, 2, 4], fnew_row)
    print("執行 set_rows(5, [0,2,4], fnew) 後：")
    print(mat.data)
    # 你會看到第 1、3 列被填為 -1，其餘位置保留原值

    # 測試 set_cols：擴增到 6 欄，保留原來 0→1,1→3,2→5，其餘用 fnew 填 -2
    def fnew_col(i, j): return -2.0
    mat.set_cols(6, [1, 3, 5], fnew_col)
    print("執行 set_cols(6, [1,3,5], fnew) 後：")
    print(mat.data)
    # 你會看到欄 0,2,4 被填為 -2，其餘保留上一版的數值

def test_imatrixindex():
    # 定義一個簡單函式 f(x,y) = x*100 + y
    I = [0, 1, 2]
    J = [10, 20]
    def f(x, y): return x*100 + y

    mat = IMatrixIndex(f, I, J)
    print("IMatrixIndex 原始 Iset, Jset：", mat.Iset.from_int(), mat.Jset.from_int())

    # 測試 submat
    sm = mat.submat([0, 2], [0, 1])
    print("IMatrixIndex submat([0,2],[0,1]) =>", sm)
    # 預期 [ f(0,10), f(0,20), f(2,10), f(2,20) ] = [10,20,210,220]

    # 測試 eval
    ev = mat.eval([(1, 0), (2, 1)])
    print("IMatrixIndex eval([(1,0),(2,1)]) =>", ev)
    # 預期 [ f(1,10), f(2,20) ] = [110, 220]

    # 測試 set_rows：加入新索引 3,5，保留原 0→0,1→1,2→2，其餘用自動補位
    new_I = [0, 1, 2, 3, 5]
    pos = mat.set_rows(new_I)
    print("執行 set_rows:", new_I)
    print("新的 Iset:", mat.Iset.from_int(), "位置映射:", pos)
    print("新的 n_rows:", mat.n_rows)

    # 測試 set_cols：加入新索引 30，保留原 10→0,20→1
    new_J = [10, 20, 30]
    pos2 = mat.set_cols(new_J)
    print("執行 set_cols:", new_J)
    print("新的 Jset:", mat.Jset.from_int(), "位置映射:", pos2)
    print("新的 n_cols:", mat.n_cols)
if __name__ == "__main__":
    print("=== 測試 MatDense ===")
    test_matdense()
    print("\n=== 測試 IMatrixIndex ===")
    test_imatrixindex()



