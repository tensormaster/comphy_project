from cytnx import *
import numpy as np

# ---------------------------
# 定義 IndexSet 類別（現有實現）
# ---------------------------
class IndexSet:
    ##------修改區----------------####
    def __init__(self, ids=None):
        self._indexes = []
        self._index_set = set()
        if ids is not None:
            for idx in ids:
                self.push_back(idx)

    def push_back(self, index):
    # 如果 index 是可迭代，就做 tuple(index)；否則視為單一元素
        try:
            tup = tuple(index)
        except TypeError:
            tup = (index,)
        if tup not in self._index_set:
            self._indexes.append(index)
            self._index_set.add(tup)

    def pos(self, index):
        """
        如果 index 是 list/tuple，回傳各元素在 _indexes 中的索引列表；
        否則回傳單一元素的位置。
        """
        if isinstance(index, (list, tuple)):
            return [self._indexes.index(idx) for idx in index]
        return self._indexes.index(index)

    def from_int(self):
        """回傳 _indexes 的淺複本。"""
        return self._indexes.copy()

    def get_all(self):
        """返回內部所有索引的複製列表。"""
        return self._indexes.copy()

    def union(self, other):
        """返回 self 與 other 的聯集，結果為一個新的 IndexSet。"""
        union_set = IndexSet(*self._indexes)
        for index in other._indexes:
            union_set.push_back(index)
        return union_set

    def intersection(self, other):
        """返回 self 與 other 的交集，結果為一個新的 IndexSet。"""
        inter = IndexSet()
        for index in self._indexes:
            if index in other._index_set:
                inter.push_back(index)
        return inter

    def sort(self, key=None, reverse=False):
        """對內部索引列表進行排序，並更新集合。"""
        self._indexes.sort(key=key, reverse=reverse)
        self._index_set = set(self._indexes)

    def __repr__(self):
        return f"IndexSet({self._indexes})"
    
    def __len__(self):
        """允許使用 len() 取得索引數量。"""
        return len(self._indexes)
    
if __name__ == "__main__":
    # 建立初始 IndexSet，注意所有索引都必須為 tuple 格式
    idx_set = IndexSet((0, 1), (1, 2), (2, 3), (0, 1))  # 重複的 (0,1) 會被排除
    print("初始 idx_set：", idx_set)

    # 加入新索引 (0, 1, 2)
    idx_set.push_back((0, 1, 2))
    print("加入 (0, 1, 2) 後：", idx_set)

    # 查詢某個索引的索引位置
    print("位置 pos((1, 2)) =", idx_set.pos((1, 2)))  # 存在，回傳 index
    print("位置 pos((1, 3)) =", idx_set.pos((1, 3)))  # 不存在，應拋出 ValueError

    # 列出所有索引
    print("所有索引 =", idx_set.get_all())

    # 建立另一個 IndexSet
    idx_set2 = IndexSet((1, 2), (2, 3), (3, 4))
    print("idx_set2 =", idx_set2)

    # 聯集
    union_set = idx_set.union(idx_set2)
    print("聯集 =", union_set)

    # 交集
    inter_set = idx_set.intersection(idx_set2)
    print("交集 =", inter_set)

    # 排序：依據每個 tuple 的第二個元素
    idx_set.sort(key=lambda x: x[1])
    print("排序後的 idx_set =", idx_set)
