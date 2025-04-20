# filename: matrix_interface.py (Corrected)

from cytnx import Tensor, zeros, Type
from IndexSet import IndexSet # Assuming IndexSet.py is available
from typing import List, Tuple, Callable, Any, Dict, Optional # <--- Added Optional
import logging

logger = logging.getLogger(__name__)

# Define MultiIndex type alias for clarity (often tuples of ints or other hashables)
MultiIndex = Any # Use Any to allow flexibility as in IndexSet

class IMatrix:
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def submat(self, rows: list[int], cols: list[int]) -> list:
        """ Get values for a submatrix defined by row/col integer indices. """
        raise NotImplementedError

    def eval(self, ids: list[tuple[int, int]]) -> list:
        """ Get values for specific (row, col) integer index pairs. """
        raise NotImplementedError

    def forget_row(self, i: int): pass
    def forget_col(self, j: int): pass


class MatDense(IMatrix):
    def __init__(self, data: Tensor):
        shape = data.shape()
        if len(shape) != 2:
             if len(shape) == 1:
                  logger.warning(f"MatDense input is 1D {shape}, reshaping to ({shape[0]}, 1)")
                  data = data.reshape(shape[0], 1)
             else:
                  raise ValueError("Tensor must be 2D or 1D")
        super().__init__(data.shape()[0], data.shape()[1])
        self.data = data.clone()

    def submat(self, rows: list[int], cols: list[int]) -> list:
        sub_values = []
        try:
            for i in rows:
                for j in cols:
                    sub_values.append(self.data[i, j].item())
        except IndexError as e:
             logger.error(f"IndexError in MatDense.submat: {e}. rows={rows}, cols={cols}, shape={self.data.shape()}")
             raise
        return sub_values

    def eval(self, ids: list[tuple[int, int]]) -> list:
        try:
            return [self.data[i, j].item() for i, j in ids]
        except IndexError as e:
             logger.error(f"IndexError in MatDense.eval: {e}. ids={ids}, shape={self.data.shape()}")
             raise

    def set_rows(self, new_nrows: int, P: list[int], fnew: Callable[[int, int], float]) -> None:
        if len(P) != self.n_rows:
             raise ValueError(f"Permutation P length ({len(P)}) must match current number of rows ({self.n_rows}).")
        new_data = zeros((new_nrows, self.n_cols), dtype=self.data.dtype(), device=self.data.device())
        if P and max(P) >= new_nrows:
             raise ValueError("Permutation P contains indices out of bounds for new_nrows.")
        for old_i, new_i in enumerate(P):
            if old_i < self.data.shape()[0]:
                 new_data[new_i, :] = self.data[old_i, :]

        new_indices = set(range(new_nrows)) - set(P)
        for i in new_indices:
            for j in range(self.n_cols):
                new_data[i, j] = fnew(i, j)
        self.data = new_data
        self.n_rows = new_nrows

    def set_cols(self, new_ncols: int, Q: list[int], fnew: Callable[[int, int], float]) -> None:
        if len(Q) != self.n_cols:
             raise ValueError(f"Permutation Q length ({len(Q)}) must match current number of columns ({self.n_cols}).")
        new_data = zeros((self.n_rows, new_ncols), dtype=self.data.dtype(), device=self.data.device())
        if Q and max(Q) >= new_ncols:
             raise ValueError("Permutation Q contains indices out of bounds for new_ncols.")
        for old_j, new_j in enumerate(Q):
            if old_j < self.data.shape()[1]:
                new_data[:, new_j] = self.data[:, old_j]

        new_indices = set(range(new_ncols)) - set(Q)
        for j in new_indices:
            for i in range(self.n_rows):
                new_data[i, j] = fnew(i, j)
        self.data = new_data
        self.n_cols = new_ncols


class IMatrixIndex(IMatrix):
    """ Matrix interface supporting arbitrary index types via IndexSet. """
    def __init__(self, f: Callable[[MultiIndex, MultiIndex], float], Iset_list: list, Jset_list: list):
        self.Iset = IndexSet(Iset_list)
        self.Jset = IndexSet(Jset_list)
        # Explicitly call the correct base class __init__ directly
        IMatrix.__init__(self, len(self.Iset), len(self.Jset)) # <--- Fixed Call
        self.A = f # The function acting on original index types

    def _get_original_indices(self, i: int, j: int) -> Tuple[MultiIndex, MultiIndex]:
        """ Helper to map integer indices back to original index types """
        if not (0 <= i < len(self.Iset)):
             raise IndexError(f"Row index {i} out of bounds for Iset of size {len(self.Iset)}")
        if not (0 <= j < len(self.Jset)):
             raise IndexError(f"Column index {j} out of bounds for Jset of size {len(self.Jset)}")
        xi = self.Iset.from_int()[i]
        yj = self.Jset.from_int()[j]
        return xi, yj

    def submat(self, rows: list[int], cols: list[int]) -> list:
        sub_values = []
        for i in rows:
            for j in cols:
                try:
                    xi, yj = self._get_original_indices(i, j)
                    sub_values.append(self.A(xi, yj))
                except IndexError as e:
                     logger.error(f"IndexError in IMatrixIndex.submat: {e}")
                     raise
        return sub_values

    def eval(self, ids: list[tuple[int, int]]) -> list:
        vals = []
        for i, j in ids:
             try:
                  xi, yj = self._get_original_indices(i, j)
                  vals.append(self.A(xi, yj))
             except IndexError as e:
                  logger.error(f"IndexError in IMatrixIndex.eval for id ({i},{j}): {e}")
                  raise
        return vals

    def set_rows(self, new_Iset_list: list) -> list[int]:
        """ Updates the row IndexSet and returns the permutation of old indices. """
        old_indices = self.Iset.from_int()
        self.Iset = IndexSet(new_Iset_list)
        self.n_rows = len(self.Iset)
        # Calculate permutation P: find positions of old_indices in the new Iset
        new_map = {idx: new_pos for new_pos, idx in enumerate(self.Iset.from_int())}
        P = [new_map.get(old_idx, -1) for old_idx in old_indices]
        # It's generally expected that old indices are a subset, handle -1 if necessary
        # Filter out -1 if the interface expects only valid mappings
        P_valid = [p for p in P if p != -1]
        if len(P_valid) != len(P):
             logger.warning(f"set_rows: {len(P) - len(P_valid)} old row indices not found in new Iset.")
        # The permutation required by MatDense/MatLazy.set_rows needs to map
        # the *old* integer indices (0 to len(old_indices)-1) to the *new* integer indices.
        old_map = {old_idx: old_pos for old_pos, old_idx in enumerate(old_indices)}
        # Permutation maps old int index -> new int index
        P_for_set = [-1] * len(old_indices)
        for new_idx_val, new_idx_pos in new_map.items():
             if new_idx_val in old_map:
                  old_idx_pos = old_map[new_idx_val]
                  P_for_set[old_idx_pos] = new_idx_pos

        if -1 in P_for_set:
            logger.warning(f"set_rows: Could not map all old row indices to new ones.")
            # Depending on how Mat{Dense/Lazy}.set_rows uses P, this might be an error
            # For now, return the potentially incomplete permutation
        return P_for_set


    def set_cols(self, new_Jset_list: list) -> list[int]:
        """ Updates the column IndexSet and returns the permutation of old indices. """
        old_indices = self.Jset.from_int()
        self.Jset = IndexSet(new_Jset_list)
        self.n_cols = len(self.Jset)
        # Calculate permutation Q (old int index -> new int index)
        new_map = {idx: new_pos for new_pos, idx in enumerate(self.Jset.from_int())}
        old_map = {old_idx: old_pos for old_pos, old_idx in enumerate(old_indices)}
        Q_for_set = [-1] * len(old_indices)
        for new_idx_val, new_idx_pos in new_map.items():
             if new_idx_val in old_map:
                  old_idx_pos = old_map[new_idx_val]
                  Q_for_set[old_idx_pos] = new_idx_pos
        if -1 in Q_for_set:
             logger.warning(f"set_cols: Could not map all old col indices to new ones.")
        return Q_for_set


class MatDenseIndex(IMatrixIndex, MatDense):
    """ Dense matrix with arbitrary index types. """
    def __init__(self, f: Callable[[MultiIndex, MultiIndex], float], Iset_list: list, Jset_list: list):
        n_rows = len(Iset_list)
        n_cols = len(Jset_list)
        data = zeros((n_rows, n_cols), dtype=Type.Double) # Assume float
        for i, xi in enumerate(Iset_list):
            for j, yj in enumerate(Jset_list):
                data[i, j] = f(xi, yj)
        # Initialize MatDense first with data
        MatDense.__init__(self, data)
        # Then initialize IMatrixIndex (which calls IMatrix.__init__)
        IMatrixIndex.__init__(self, f, Iset_list, Jset_list)

    def set_rows(self, new_Iset_list: list) -> list[int]:
        """ Overrides set_rows to update both index set and dense data. """
        # 1. Update IMatrixIndex part (updates self.Iset, self.n_rows, returns P)
        P = IMatrixIndex.set_rows(self, new_Iset_list)

        # 2. Define fnew for dense update based on the *new* self.Iset
        def fnew_for_dense(new_i_int, j_int):
             new_xi = self.Iset.from_int()[new_i_int]
             yj = self.Jset.from_int()[j_int]
             return self.A(new_xi, yj)

        # 3. Update MatDense part using P and fnew
        MatDense.set_rows(self, self.n_rows, P, fnew_for_dense)
        return P

    def set_cols(self, new_Jset_list: list) -> list[int]:
        """ Overrides set_cols to update both index set and dense data. """
        # 1. Update IMatrixIndex part (updates self.Jset, self.n_cols, returns Q)
        Q = IMatrixIndex.set_cols(self, new_Jset_list)

        # 2. Define fnew for dense update based on the *new* self.Jset
        def fnew_for_dense(i_int, new_j_int):
             xi = self.Iset.from_int()[i_int]
             new_yj = self.Jset.from_int()[new_j_int]
             return self.A(xi, new_yj)

        # 3. Update MatDense part using Q and fnew
        MatDense.set_cols(self, self.n_cols, Q, fnew_for_dense)
        return Q


class MatLazy(IMatrix):
    """ Lazy evaluation matrix using integer indices. """
    def __init__(self, f: Callable[[int, int], float], n_rows: int, n_cols: int):
        super().__init__(n_rows, n_cols)
        self.f = f
        self._cache: Dict[Tuple[int, int], float] = {}

    def submat(self, rows: list[int], cols: list[int]) -> list:
        ids_to_eval = [(i, j) for i in rows for j in cols]
        eval_results = self.eval(ids_to_eval)
        return eval_results

    def eval(self, ids: list[tuple[int, int]]) -> list:
        out = []
        ids_to_compute = []
        indices_to_compute = []

        # First pass: check cache
        for idx_enum, idx in enumerate(ids):
            if idx in self._cache:
                out.append(self._cache[idx])
            else:
                i, j = idx
                if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
                     raise IndexError(f"Lazy evaluation index ({i},{j}) out of bounds ({self.n_rows},{self.n_cols})")
                # Mark for computation
                out.append(None) # Placeholder
                ids_to_compute.append(idx)
                indices_to_compute.append(idx_enum) # Store original position

        # Second pass: compute missing values (potentially in batch if f supports it)
        if ids_to_compute:
            logger.debug(f"MatLazy evaluating {len(ids_to_compute)} points.")
            try:
                # Simple one-by-one evaluation
                computed_values = [self.f(i, j) for i, j in ids_to_compute]
            except Exception as e:
                 logger.error(f"Error evaluating lazy function f: {e}")
                 raise

            # Fill placeholders and update cache
            for original_index, computed_idx, computed_val in zip(indices_to_compute, ids_to_compute, computed_values):
                out[original_index] = computed_val
                self._cache[computed_idx] = computed_val

        return out


    def forget_row(self, i: int):
        self._cache = {k: v for k, v in self._cache.items() if k[0] != i}

    def forget_col(self, j: int):
        self._cache = {k: v for k, v in self._cache.items() if k[1] != j}

    def set_rows(self, new_nrows: int, P: list[int], fnew: Callable[[int, int], float]):
        """ Updates rows, reorders cache, sets new function. P maps old int index -> new int index """
        new_cache = {}
        # P should map old_int_idx -> new_int_idx. Length should be self.n_rows
        if len(P) != self.n_rows:
            logger.error(f"set_rows permutation P length ({len(P)}) mismatch with n_rows ({self.n_rows})")
            # Fallback: clear cache completely if P is unusable
            self._cache = {}
        else:
            for (old_i, j), val in self._cache.items():
                 if old_i < len(P) and P[old_i] != -1: # If old index has a valid mapping
                      new_i = P[old_i]
                      if 0 <= new_i < new_nrows: # Check if mapped index is within new bounds
                           new_cache[(new_i, j)] = val
            self._cache = new_cache

        self.n_rows = new_nrows
        self.f = fnew

    def set_cols(self, new_ncols: int, Q: list[int], fnew: Callable[[int, int], float]):
        """ Updates columns, reorders cache, sets new function. Q maps old int index -> new int index """
        new_cache = {}
        if len(Q) != self.n_cols:
             logger.error(f"set_cols permutation Q length ({len(Q)}) mismatch with n_cols ({self.n_cols})")
             self._cache = {}
        else:
            for (i, old_j), val in self._cache.items():
                 if old_j < len(Q) and Q[old_j] != -1:
                      new_j = Q[old_j]
                      if 0 <= new_j < new_ncols:
                            new_cache[(i, new_j)] = val
            self._cache = new_cache

        self.n_cols = new_ncols
        self.f = fnew


# --- Implementation of MatLazyIndex ---
class MatLazyIndex(IMatrixIndex, MatLazy):
    """ Lazy evaluation matrix with arbitrary index types. """
    def __init__(self, f: Callable[[MultiIndex, MultiIndex], float], Iset_list: list, Jset_list: list):
        # 1. Initialize IMatrixIndex
        IMatrixIndex.__init__(self, f, Iset_list, Jset_list)

        # 2. Define the integer-indexed function for MatLazy
        def f_int(i: int, j: int) -> float:
            try:
                xi, yj = self._get_original_indices(i, j)
                return self.A(xi, yj)
            except IndexError as e:
                 logger.error(f"Internal error: f_int called with invalid integer index ({i},{j}): {e}")
                 raise
            except Exception as e:
                 logger.error(f"Error calling underlying function self.A in f_int({i},{j}): {e}")
                 raise

        # 3. Initialize MatLazy
        MatLazy.__init__(self, f_int, self.n_rows, self.n_cols)


    def set_rows(self, new_Iset_list: list) -> list[int]:
        """ Overrides set_rows to update both IndexSet and the lazy function. """
        # 1. Update IMatrixIndex (updates self.Iset, self.n_rows, returns P: old_int_idx -> new_int_idx)
        P = IMatrixIndex.set_rows(self, new_Iset_list)

        # 2. Define the *new* integer-indexed function
        def fnew_int(i: int, j: int) -> float:
            try:
                 xi, yj = self._get_original_indices(i, j) # Uses updated Iset
                 return self.A(xi, yj)
            except IndexError as e:
                 logger.error(f"Internal error: fnew_int called with invalid integer index ({i},{j}): {e}")
                 raise
            except Exception as e:
                 logger.error(f"Error calling underlying function self.A in fnew_int({i},{j}): {e}")
                 raise

        # 3. Update MatLazy (updates self.f to fnew_int, self.n_rows, permutes cache using P)
        MatLazy.set_rows(self, self.n_rows, P, fnew_int)
        return P

    def set_cols(self, new_Jset_list: list) -> list[int]:
        """ Overrides set_cols to update both IndexSet and the lazy function. """
        # 1. Update IMatrixIndex (updates self.Jset, self.n_cols, returns Q: old_int_idx -> new_int_idx)
        Q = IMatrixIndex.set_cols(self, new_Jset_list)

        # 2. Define the *new* integer-indexed function
        def fnew_int(i: int, j: int) -> float:
            try:
                 xi, yj = self._get_original_indices(i, j) # Uses updated Jset
                 return self.A(xi, yj)
            except IndexError as e:
                 logger.error(f"Internal error: fnew_int called with invalid integer index ({i},{j}): {e}")
                 raise
            except Exception as e:
                 logger.error(f"Error calling underlying function self.A in fnew_int({i},{j}): {e}")
                 raise

        # 3. Update MatLazy (updates self.f to fnew_int, self.n_cols, permutes cache using Q)
        MatLazy.set_cols(self, self.n_cols, Q, fnew_int)
        return Q


# --- Factory Function (remains the same) ---
def make_IMatrix(f: Callable,
                 n_rows_or_Iset: Any,
                 n_cols_or_Jset: Any,
                 full: bool = False) -> IMatrix:
    """ Factory function to create different IMatrix types. """
    is_index_version = isinstance(n_rows_or_Iset, list) or isinstance(n_cols_or_Jset, list)

    if is_index_version:
        Iset = n_rows_or_Iset if isinstance(n_rows_or_Iset, list) else list(range(n_rows_or_Iset))
        Jset = n_cols_or_Jset if isinstance(n_cols_or_Jset, list) else list(range(n_cols_or_Jset))
        if full:
            logger.debug("make_IMatrix: Creating MatDenseIndex")
            return MatDenseIndex(f, Iset, Jset)
        else:
            logger.debug("make_IMatrix: Creating MatLazyIndex")
            return MatLazyIndex(f, Iset, Jset)
    else: # Integer index version
        n_rows = n_rows_or_Iset
        n_cols = n_cols_or_Jset
        if full:
            logger.debug("make_IMatrix: Creating MatDense")
            data = zeros((n_rows, n_cols), dtype=Type.Double) # Assume float
            for i in range(n_rows):
                for j in range(n_cols):
                    data[i, j] = f(i, j)
            return MatDense(data)
        else:
            logger.debug("make_IMatrix: Creating MatLazy")
            return MatLazy(f, n_rows, n_cols)

# --- Test Functions ---
def test_matlazyindex():
    print("\n=== 測試 MatLazyIndex ===")
    I = [(0,), (1,), (2,)] # Use tuples for multi-indices
    J = [(10,), (20,)]
    eval_count = 0
    def f(x_tuple, y_tuple):
        nonlocal eval_count
        eval_count += 1
        res = (x_tuple[0] if x_tuple else 0) * 100 + (y_tuple[0] if y_tuple else 0)
        print(f"      Lazy func called ({eval_count}): f({x_tuple}, {y_tuple}) -> {res}")
        return float(res)

    mat = MatLazyIndex(f, I, J)
    print("MatLazyIndex 初始 Iset, Jset:", mat.Iset.from_int(), mat.Jset.from_int())
    print("MatLazyIndex dimensions:", mat.n_rows, mat.n_cols)

    print("\nTesting eval([(1, 0), (2, 1)])...")
    ev = mat.eval([(1, 0), (2, 1)])
    print("MatLazyIndex eval([(1,0),(2,1)]) =>", ev)

    print("\nTesting eval([(2, 1)]) again (should use cache)...")
    ev2 = mat.eval([(2, 1)])
    print("MatLazyIndex eval([(2,1)]) =>", ev2)

    print("\nTesting submat([0, 2], [0, 1])...")
    sm = mat.submat([0, 2], [0, 1])
    print("MatLazyIndex submat([0,2],[0,1]) =>", sm)

    print("\nTesting set_rows([(0,), (2,), (3,)])...")
    new_I = [(0,), (2,), (3,)]
    # old_indices = mat.Iset.from_int() # [(0,), (1,), (2,)]
    P = mat.set_rows(new_I) # Expected P = [0, -1, 1] (old 0 maps to new 0, old 1 lost, old 2 maps to new 1)
    print("New Iset:", mat.Iset.from_int(), "Permutation P (old_idx->new_idx):", P)
    print("New dimensions:", mat.n_rows, mat.n_cols)

    print("\nTesting eval([(0, 1)]) after set_rows...") # new index 0 corresponds to old (0,)
    ev3 = mat.eval([(0, 1)]) # Should eval f((0,), (20,)) = 20
    print("MatLazyIndex eval([(0,1)]) =>", ev3)

    print("\nTesting eval([(2, 0)]) after set_rows...") # new index 2 corresponds to new (3,)
    ev4 = mat.eval([(2, 0)]) # Should eval f((3,), (10,)) = 310
    print("MatLazyIndex eval([(2,0)]) =>", ev4)

    print("\nTesting eval([(1, 0)]) after set_rows...") # new index 1 corresponds to old (2,)
    ev5 = mat.eval([(1, 0)]) # Should eval f((2,), (10,)) = 210 (might be cached if submat was called before set_rows)
    print("MatLazyIndex eval([(1,0)]) =>", ev5)

    print(f"\nTotal lazy function calls: {eval_count}")


if __name__ == "__main__":
    test_matlazyindex()