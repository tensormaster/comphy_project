import cytnx
from cytnx import *
import numpy as np
from IndexSet import IndexSet
from Qgrid import QuanticGrid

class TCI:
    def __init__(self, grid, tolerance=1e-6, verbose=True, max_iter=50, pivot1=None):
        self.grid = grid
        self.dim = grid.dim
        self.tolerance = tolerance
        self.verbose = verbose
        self.max_iter = max_iter

        # 假設每個維度的大小皆為 grid.M
        self.localDim = [grid.M for _ in range(self.dim)]
        
        # 1. 設置初始樞軸
        if pivot1 is None:
            # 如果未提供 pivot1，則設為全 0 向量，長度等於維數
            self.pivot1 = [0] * self.dim
            print(f"pivot 1 :",self.pivot1)
        else:
            self.pivot1 = pivot1
        
        # 評估函數 f 在 pivot1 處的值（假設 grid.f 為待近似的函數）
        self.f = grid.f
        initial_value = self.f(*self.pivot1)
        self.pivotError = [abs(initial_value)]
        if self.pivotError[0] == 0.0:
            raise ValueError("Not expecting f(pivot1)=0. Provide a better first pivot in the param")
        
        # 假設 self.dim 為維度數，self.localDim 為每個維度的大小，self.pivot1 為初始樞軸列表
        # 建立一個列表，每個元素都是一個 IndexSet 物件
        self.localSet = [IndexSet() for _ in range(self.dim)]
        self.Iset = [IndexSet() for _ in range(self.dim)]
        self.Jset = [IndexSet() for _ in range(self.dim)]

        for p in range(self.dim):
            # 填充 localSet[p]：每個可能的局部索引都要加入
            for i in range(self.localDim[p]):
                # 注意這裡以 tuple 存放單一索引
                self.localSet[p].push_back((i,))
            
            # 填充 Iset[p]：使用 pivot1 的前 p 個元素，轉成 tuple 後存入
            self.Iset[p].push_back(tuple(self.pivot1[:p]))
            
            # 填充 Jset[p]：使用 pivot1 從 p+1 開始的元素（也轉成 tuple）
            self.Jset[p].push_back(tuple(self.pivot1[p+1:]))

        if self.verbose:
            print("[Debug] localSet:", self.localSet)
            print("[Debug] Iset:", self.Iset)
            print("[Debug] Jset:", self.Jset)

        # 3. 初始化樞軸矩陣、交叉資料、T3 張量與樞軸矩陣 P
        # 這部分需要你依據具體算法實作
        self.Pi_mat = []
        self.Pi_bool = []
        self.cross = []
        self.T3 = [None] * self.dim  # 用來存放每個鍵上的 3D 張量
        self.P = [None] * self.dim   # 存放每個鍵上的樞軸矩陣
        
        for p in range(self.dim - 1):
            # 建構第 p 個鍵的樞軸矩陣（需要實作 buildPiAt 方法）
            Pi = self.buildPiAt(p)
            self.Pi_mat.append(Pi)
            # 如有需要，建構對應的布林樞軸（例如過濾某些樞軸）
            # self.Pi_bool.append(self.buildPiBoolAt(p))  # 如果條件函數存在
            # 根據 Pi 的尺寸初始化 cross 結構（這裡僅示意）
            cross_data = {"n_rows": Pi.n_rows, "n_cols": Pi.n_cols, "C": None, "R": None, "pivotMat": None}
            self.cross.append(cross_data)
        
        for p in range(self.dim - 1):
            # 利用 cross[p] 更新 pivot 資訊，這裡假設有個 addPivot 方法
            self.addPivot(p)
            # 建構 T3 張量：
            # 如果 p==0，從 cross[p] 中取 C 部分來構造 T3[0]；否則從 R 部分構造 T3[p+1]
            if p == 0:
                self.T3[p] = self.buildCube(self.cross[p]["C"], len(self.Iset[p]), len(self.localSet[p]), len(self.Jset[p]))
            self.T3[p+1] = self.buildCube(self.cross[p]["R"], len(self.Iset[p+1]), len(self.localSet[p+1]), len(self.Jset[p+1]))
            # 取得對應的樞軸矩陣
            self.P[p] = self.cross[p].get("pivotMat", None)
        
        # 最後一個 P 設為 1x1 的單位矩陣
        self.P[-1] = self.identityMatrix(1)
        
        # 4. 如有權重設定，可進行 TT 加權（這裡僅示意）
        if hasattr(self, 'weight') and self.weight:
            self.tt_sum = self.computeTTSum(self.get_TensorTrain(0), self.weight)
        
        # 5. 進行迭代更新以收斂近似（實作 iterate 方法）
        self.iterate(self.max_iter)
    
    def buildPiAt(self, p):
        """
        為鍵 p 構建 Pi 矩陣：
        行索引：由 Iset[p] 與 localSet[p] 組合而成，
        - Iset[p] 表示從站點 0 到站點 p-1 的多索引集合，
        - localSet[p] 表示站點 p 的局部索引集合。
        列索引：由 localSet[p+1] 與 Jset[p+1] 組合而成，
        - localSet[p+1] 表示站點 p+1 的局部索引集合，
        - Jset[p+1] 表示從站點 p+2 到最後站點的多索引集合。
        Pi 矩陣的元素定義為： f( i + a + b + j )，
        其中 i 來自 Iset[p]，a 來自 localSet[p]，b 來自 localSet[p+1]，j 來自 Jset[p+1]。
        """
        # 建立行組合索引：對於每個 i ∈ Iset[p] 和 a ∈ localSet[p]，連接成一個 tuple
        left_multi = []
        for left in self.Iset[p].get_all():
            for a in self.localSet[p].get_all():
                left_multi.append(left + a)
        
        # 建立列組合索引：對於每個 b ∈ localSet[p+1] 和 j ∈ Jset[p+1]，連接成一個 tuple
        right_multi = []
        for b in self.localSet[p+1].get_all():
            for right in self.Jset[p+1].get_all():
                right_multi.append(b + right)
        
        n_rows = len(left_multi)
        n_cols = len(right_multi)
        Pi_matrix = np.zeros((n_rows, n_cols))
        
        # 計算每個矩陣元素：組合 left_multi 與 right_multi 後形成完整多索引，再計算 f(*full_index)
        for i, left_index in enumerate(left_multi):
            for j, right_index in enumerate(right_multi):
                full_index = left_index + right_index  # full_index 的長度應等於 self.dim
                Pi_matrix[i, j] = self.f(*full_index)
        
        # 用一個簡單的物件保存 Pi 矩陣及其相關資訊
        Pi = type("PiMatrix", (), {})()  # 建立空物件
        Pi.n_rows = n_rows
        Pi.n_cols = n_cols
        Pi.data = Pi_matrix
        return Pi

    def addPivot(self, p):
        """
        更新交叉數據 cross[p] 中的樞軸資訊：
        根據 C++ 程式碼的邏輯，此處會取 Iset[p+1] 的第一個元素（左側樞軸）
        與 Jset[p] 的第一個元素（右側樞軸），並在 Pi 矩陣中找到它們的位置，
        然後將該樞軸值存入 cross[p]["pivotMat"]。
        
        此範例中為簡化起見，假設找到的位置均為 0。
        真實實作中應重複 buildPiAt 的索引生成流程，比對並確定正確位置。
        """
        # 取得左側樞軸：來自 Iset[p+1]（若存在）
        if self.Iset[p+1].get_all():
            left_pivot = self.Iset[p+1].get_all()[0]
        else:
            left_pivot = ()
        
        # 取得右側樞軸：來自 Jset[p]（若存在）
        if self.Jset[p].get_all():
            right_pivot = self.Jset[p].get_all()[0]
        else:
            right_pivot = ()
        
        # 這裡應依照 buildPiAt 的行、列索引生成過程，找出 left_pivot 和 right_pivot 在 Pi 矩陣中所對應的位置。
        # 為簡化，假設對應位置均為 0。
        pivot_row = 0  # TODO: 根據 left_pivot 計算正確位置
        pivot_col = 0  # TODO: 根據 right_pivot 計算正確位置
        
        # 將樞軸位置存入交叉數據，這裡以 1x1 矩陣形式表示
        self.cross[p]["pivotMat"] = np.array([[self.Pi_mat[p].data[pivot_row, pivot_col]]])

    
    def buildCube(self, data, dim1, dim2, dim3):
        """
        將提供的二維資料重塑為形狀為 (dim1, dim2, dim3) 的三維張量。
        對應於 C++：arma::Cube(f_ptr, d1, d2, d3)。
        
        參數：
            data：輸入的二維資料。若為 None，則會建立一個全零的張量。
            dim1, dim2, dim3：三維張量的維度。
        
        傳回值：
            一個重塑為 (dim1, dim2, dim3) 的 numpy ndarray。
        """
        # 確保維度皆為整數
        try:
            d1, d2, d3 = int(dim1), int(dim2), int(dim3)
        except Exception as e:
            raise ValueError("維度必須可轉換為整數。") from e

        # 若 data 為 None，則建立一個全零張量
        if data is None:
            return np.zeros((d1, d2, d3))
        
        # 若輸入資料不是 numpy 陣列，則進行轉換
        data_arr = np.array(data)
        
        # 檢查資料元素數是否符合預期維度大小
        expected_size = d1 * d2 * d3
        if data_arr.size != expected_size:
            raise ValueError(
                f"資料大小（{data_arr.size}）與預期形狀不符 "
                f"（{d1} x {d2} x {d3}）= {expected_size} 個元素。"
            )
        
        # 將資料重塑為三維張量
        return data_arr.reshape((d1, d2, d3))
    
    def identityMatrix(self, size):
        # 返回一個 size x size 的單位矩陣
        import numpy as np
        return np.eye(size)
    
    def computeTTSum(self, tensorTrain, weight):
        # 實現 TT 加權的計算
        pass
    
    def get_TensorTrain(self, mode):
        # 返回目前的 Tensor Train 近似（示意方法）
        pass
    
    def iterate(self, nIter):
        # 進行 nIter 次的迭代更新，逐步改善 TCI 近似
        pass




def f(x, y, z):
    return np.exp(-x**2) + np.sin(x**2+y**2) + np.cos(y**2+z**2)

def main():
    grid = QuanticGrid(a=0, b=10, nBit=5, dim=3, fused=False, grid_method="uniform", f=f)
    tci = TCI(grid, tolerance=1e-2, verbose=True, max_iter=5)

if __name__ == "__main__":
    main()
