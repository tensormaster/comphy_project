import cytnx  
from cytnx import *
import numpy as np
from cytnx_prrLU import RankRevealingLU

# --------------------------------------------------------
# 網格生成函數
# --------------------------------------------------------
def uniform_grid(a, b, M):
    return cytnx.arange(a, b, (b - a) / M)

def gauss_kronrod_grid(a, b):
    M = 15
    xs = uniform_grid(a, b, M)
    ws = cytnx.ones(M, dtype=Type.Double, device=-1) * ((b - a) / M)
    return xs, ws

# --------------------------------------------------------
# QuanticGrid 類別：生成量化張量
# --------------------------------------------------------
class QuanticGrid:
    def __init__(self, a=0.0, b=1.0, nBit=4, dim=1, fused=False, grid_method="uniform", custom_grid_func=None, f=None):
        self.a = a
        self.b = b
        self.nBit = nBit
        self.dim = dim
        self.fused = fused
        self.M = 1 << nBit  # 每個變數的格點數，必須是2的冪
        self.grid_method = grid_method.lower()
        self.custom_grid_func = custom_grid_func
        self.f = f
        self.grids = []
        for i in range(dim):
            if self.custom_grid_func is not None:
                xs, _ = self.custom_grid_func(a, b)
            elif self.grid_method == "uniform":
                xs = uniform_grid(a, b, self.M)
            elif self.grid_method == "gk":
                xs, _ = gauss_kronrod_grid(a, b)
            else:
                raise ValueError("grid_method must be 'uniform', 'GK', or use custom_grid_func")
            self.grids.append(xs)
        
    def get_cartesian_grid(self):
        from itertools import product
        prod = list(product(*[range(self.M) for _ in range(self.dim)]))
        grid_points = []
        for idx in prod:
            pt = []
            for d in range(self.dim):
                pt.append(float(self.grids[d][idx[d]].item()))
            grid_points.append(pt)
        return grid_points

    def coord_to_bin(self, coord):
        scaled = int((coord - self.a) / (self.b - self.a) * (self.M - 1))
        bin_str = format(scaled, '0{}b'.format(self.nBit))
        return [int(bit) for bit in bin_str]
    
    def get_bin_indices(self):
        bin_indices = []
        for d in range(self.dim):
            arr = []
            for i in range(self.M):
                arr.append(self.coord_to_bin(self.grids[d][i]))
            bin_indices.append(np.array(arr))
        return bin_indices

    def fuse_indices(self):
        bin_indices = self.get_bin_indices()
        fused = []
        for d in range(self.dim):
            ints = []
            for bits in bin_indices[d]:
                ints.append(int("".join(str(b) for b in bits), 2))
            fused.append(np.array(ints))
        return fused

    def interleave_indices(self):
        bin_indices = self.get_bin_indices()
        from itertools import product
        all_indices = list(product(range(self.M), repeat=self.dim))
        res = []
        for idx in all_indices:
            new_idx = []
            for d in range(self.dim):
                bits = []
                for r in range(self.nBit):
                    bits.append(bin_indices[d][idx[d]][r])
                new_idx.append(int("".join(str(b) for b in bits), 2))
            res.append(new_idx)
        return np.array(res)

    def get_quantics_tensor(self):
        if self.f is None:
            raise ValueError("Function f is not defined in QuanticGrid.")
        pts = self.get_cartesian_grid()
        # 注意：對於多維情況，呼叫函數時使用 *pt 解包列表（例如 f(x,y,z)）
        values = np.array([self.f(*pt) for pt in pts])
        shape = [self.M] * self.dim
        # 注意這裡 T 為 cytnx Tensor
        T = from_numpy(values.reshape(shape))
        
        # 如果 T 是一維則 reshape 成 (M,1)
        if len(T.shape()) == 1:
            T = T.reshape(shape[0], 1)
        # 返回包裝為 UniTensor 的物件，方便後續操作
        return UniTensor(T)
    
    def reshape_to_mps(self, T):
        # T 必須是一個 UniTensor，且其 underlying tensor 尺寸必須與離散張量一致
        # 檢查 self.M 是否是 2 的冪
        if (self.M & (self.M - 1)) != 0:
            raise ValueError("self.M 必須是 2 的冪")
        
        if self.fused is False:
            # 非 fused 狀態下，總位數為 dim * log2(self.M)
            total_bits = self.dim * int(np.log2(self.M))
            shape = [2 for _ in range(total_bits)]
            print("Reshape shape (non-fused):", shape, type(shape))
            # 對 UniTensor 進行就地重塑，使用 reshape_ 方法，傳入展開後的各個維度
            T.reshape_(*shape)
            ut = UniTensor(T.get_block_())  # 根據重塑後的底層 Tensor 重新建立 UniTensor
            return ut
        else:
            # fused 狀態下，總站點數為 log2(self.M)
            total_sites = int(np.log2(self.M))
            shape = [2 ** self.dim for _ in range(total_sites)]
            print("Reshape shape (fused):", shape, type(shape))
            T.reshape_(*shape)
            ut = UniTensor(T.get_block_())
            return ut

    


# ============================================================
# 主程式：測試 3D 情況下離散化與重塑
# ============================================================
def f3(x, y, z):
    return x**2 + y**2 + z**2

if __name__ == "__main__":
    # 創建一個 3D 網格，nBit=3 則每個變數 2^3=8 個點，dim=3 則總共 8^3 = 512 個點
    three_grid = QuanticGrid(a=0, b=1, nBit=3, dim=3, grid_method="uniform", f=f3)
    T_quantics = three_grid.get_quantics_tensor()
    print("Quantics 張量的形狀：", T_quantics.shape())
    
    # 將 Quantics 張量重塑成 MPS 形式
    mps_tensor = three_grid.reshape_to_mps(T_quantics)
    print("Reshape 成 MPS 張量：")
    mps_tensor.print_diagram()

    prr = RankRevealingLU(mps_tensor, 3)