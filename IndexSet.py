from cytnx import *
import numpy as np

# ---------------------------
# 定義 IndexSet 類別（現有實現）
# ---------------------------
class IndexSet:
    def __init__(self, *args):
        """
        初始化 IndexSet，可選參數作為初始索引逐個加入集合。
        """
        self._indexes = []         # 儲存索引的列表（保持順序）
        self._index_set = set()      # 用於快速查重
        for arg in args:
            self.push_back(arg)

    def push_back(self, index):
            index_tuple = tuple(index)  # 將索引轉為元組
            if index_tuple not in self._index_set:
                self._indexes.append(index_tuple)
                self._index_set.add(index_tuple)

    def pos(self, index):
            index_tuple = tuple(index)
            try:
                return self._indexes.index(index_tuple)
            except ValueError:
                return -1

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
    
if __name__ == "__main__":
    idx_set = IndexSet((0, 1), (1, 2), (2, 3), (0, 1))
    print(idx_set)
    idx_set.push_back((0, 1,2))
    print(idx_set)
    print(idx_set.pos((1, 2)))
    print(idx_set.pos((1, 3)))
    print(idx_set.get_all())
    idx_set2 = IndexSet((1, 2), (2, 3), (3, 4))
    print(idx_set.union(idx_set2))
    print(idx_set.intersection(idx_set2))
    idx_set.sort(key=lambda x: x[1])
    print(idx_set)