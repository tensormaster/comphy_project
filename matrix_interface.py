# filename: matrix_interface.py (Corrected based on user feedback and Cytnx API)
import cytnx
import numpy as np # 必須導入 numpy

from cytnx import Tensor, zeros, Type, Device, from_numpy # 導入 from_numpy
from IndexSet import IndexSet # Assuming IndexSet.py is in the Python path
from typing import Tuple, Callable, Any, Dict, List, Optional 

import logging

logger = logging.getLogger(__name__)

MultiIndex = Any # Or more specific, e.g., Tuple[int, ...]

class IMatrix:
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.dtype: int = Type.Double 
        self.device: int = Device.cpu

    def submat(self, rows: list[int], cols: list[int]) -> list:
        """ Get values for a submatrix defined by row/col integer indices. Returns a flat list. """
        raise NotImplementedError("submat must be implemented by a subclass")

    def eval(self, ids: list[tuple[int, int]]) -> list:
        """ Get values for specific (row, col) integer index pairs. Returns a list of values. """
        raise NotImplementedError("eval must be implemented by a subclass")

    def _get_np_dtype(self) -> Any: # Helper to map cytnx.Type to numpy.dtype
        # Add more mappings as needed based on types your functions 'f' can return
        if self.dtype == Type.ComplexDouble: return np.complex128
        if self.dtype == Type.Float: return np.float32
        if self.dtype == Type.ComplexFloat: return np.complex64
        if self.dtype == Type.Int64: return np.int64
        if self.dtype == Type.Uint64: return np.uint64
        if self.dtype == Type.Int32: return np.int32
        if self.dtype == Type.Uint32: return np.uint32
        if self.dtype == Type.Int16: return np.int16
        if self.dtype == Type.Uint16: return np.uint16
        if self.dtype == Type.Bool: return np.bool_
        return np.float64 # Default for Type.Double or unknown

    def get_row_as_tensor(self, row_idx: int) -> Tensor:
        if not (0 <= row_idx < self.n_rows):
            raise IndexError(f"IMatrix: Row index {row_idx} out of bounds for {self.n_rows} rows.")
        
        row_data_list = self.submat([row_idx], list(range(self.n_cols)))
        
        if not row_data_list: # Handle empty row
            # If n_cols is 0, shape should be [0]. Otherwise, [self.n_cols] for a row.
            shape = [0] if self.n_cols == 0 else [self.n_cols]
            # If submat returns empty list for a row that should have elements,
            # it implies all elements were perhaps filtered or function returned None.
            # For now, assume if submat returns empty for non-zero n_cols, it means zeros.
            # However, submat should ideally return list of numbers.
            if self.n_cols > 0 :
                 logger.debug(f"IMatrix.get_row_as_tensor: Row {row_idx} data list is empty, "
                              f"but n_cols={self.n_cols}. Returning zeros.")
                 return zeros(shape, dtype=self.dtype, device=self.device)
            else: # n_cols == 0
                 return Tensor(shape=shape, dtype=self.dtype, device=self.device, init_zero=True)

        np_arr = np.array(row_data_list, dtype=self._get_np_dtype())
        return from_numpy(np_arr).to(self.device)

    def get_col_as_tensor(self, col_idx: int) -> Tensor:
        if not (0 <= col_idx < self.n_cols):
            raise IndexError(f"IMatrix: Column index {col_idx} out of bounds for {self.n_cols} columns.")
        
        col_data_list = self.submat(list(range(self.n_rows)), [col_idx])

        if not col_data_list: # Handle empty column
            shape = [0] if self.n_rows == 0 else [self.n_rows]
            if self.n_rows > 0:
                logger.debug(f"IMatrix.get_col_as_tensor: Col {col_idx} data list is empty, "
                             f"but n_rows={self.n_rows}. Returning zeros.")
                return zeros(shape, dtype=self.dtype, device=self.device)
            else: # n_rows == 0
                return Tensor(shape=shape, dtype=self.dtype, device=self.device, init_zero=True)

        np_arr = np.array(col_data_list, dtype=self._get_np_dtype())
        return from_numpy(np_arr).to(self.device)

    def forget_row(self, i: int): pass 
    def forget_col(self, j: int): pass 


class MatDense(IMatrix):
    def __init__(self, data: Tensor):
        shape = data.shape()
        if len(shape) != 2:
            if len(shape) == 1: 
                data = data.reshape(shape[0], 1)
            else:
                raise ValueError(f"Input Tensor for MatDense must be 2D or 1D. Got shape: {shape}")
        # Call IMatrix constructor AFTER setting dtype and device from data
        self.dtype = data.dtype() 
        self.device = data.device()
        super().__init__(data.shape()[0], data.shape()[1])
        self.data = data.clone()

    def submat(self, rows: list[int], cols: list[int]) -> list:
        sub_values = []
        try:
            # Optimized submat for dense data
            # This constructs a potentially large intermediate tensor if rows/cols are many.
            # For the IMatrix default get_row/col_as_tensor, submat is called for a single row/col.
            if len(rows) == 1 and len(cols) == self.n_cols: # Special case for getting a full row
                return [self.data[rows[0], j].item() for j in cols]
            if len(cols) == 1 and len(rows) == self.n_rows: # Special case for getting a full col
                return [self.data[i, cols[0]].item() for i in rows]
            
            # General case (less efficient for single row/col for default IMatrix methods)
            for i in rows:
                for j in cols: 
                    sub_values.append(self.data[i,j].item())
        except IndexError as e: 
            logger.error(f"IndexError in MatDense.submat: {e}. Requested rows/cols might be out of bounds for shape {self.data.shape()}.")
            raise
        return sub_values

    def eval(self, ids: list[tuple[int, int]]) -> list:
        vals = []
        try: 
            for i,j in ids:
                if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
                    raise IndexError(f"Eval index ({i},{j}) out of bounds for MatDense {self.n_rows}x{self.n_cols}")
                vals.append(self.data[i,j].item())
            return vals
        except IndexError as e: 
            logger.error(f"IndexError in MatDense.eval: {e}. ids={ids}, shape={self.data.shape()}")
            raise
        return [] 

    def get_row_as_tensor(self, row_idx: int) -> Tensor: 
        if not (0 <= row_idx < self.n_rows):
            raise IndexError(f"MatDense: Row index {row_idx} out of bounds ({self.n_rows}).")
        return self.data[row_idx, :].clone() 

    def get_col_as_tensor(self, col_idx: int) -> Tensor: 
        if not (0 <= col_idx < self.n_cols):
            raise IndexError(f"MatDense: Col index {col_idx} out of bounds ({self.n_cols}).")
        return self.data[:, col_idx].clone() 
    
    def set_rows(self, new_nrows: int, P: list[int], fnew: Callable[[int, int], Any]) -> None:
        if len(P) != self.n_rows: 
            raise ValueError(f"Permutation P length ({len(P)}) must match current rows ({self.n_rows}).")
        new_data = zeros((new_nrows, self.n_cols), dtype=self.dtype, device=self.device)
        
        valid_P_indices = [idx for idx in P if idx != -1]
        if valid_P_indices and (max(valid_P_indices) >= new_nrows): 
            raise ValueError("Permutation P contains indices out of bounds for new_nrows.")

        for old_i, new_i in enumerate(P):
            if new_i != -1 : # If -1, this old row is not mapped to a new row directly
                if old_i < self.data.shape()[0]: # Ensure old_i is within bounds of current self.data
                    new_data[new_i, :] = self.data[old_i, :]
        
        # Identify rows in the new matrix that were not filled by permutation
        newly_created_row_indices = set(range(new_nrows)) - set(p for p in P if p != -1)
        for i in newly_created_row_indices:
            for j in range(self.n_cols): 
                new_data[i, j] = fnew(i, j) 
        
        self.data = new_data
        self.n_rows = new_nrows

    def set_cols(self, new_ncols: int, Q: list[int], fnew: Callable[[int, int], Any]) -> None:
        if len(Q) != self.n_cols: 
            raise ValueError(f"Permutation Q length ({len(Q)}) must match current columns ({self.n_cols}).")
        new_data = zeros((self.n_rows, new_ncols), dtype=self.dtype, device=self.device)
        
        valid_Q_indices = [idx for idx in Q if idx != -1]
        if valid_Q_indices and (max(valid_Q_indices) >= new_ncols): 
            raise ValueError("Permutation Q contains indices out of bounds for new_ncols.")

        for old_j, new_j in enumerate(Q):
            if new_j != -1: # If -1, this old col is not mapped
                if old_j < self.data.shape()[1]: # Ensure old_j is within bounds
                    new_data[:, new_j] = self.data[:, old_j]
        
        newly_created_col_indices = set(range(new_ncols)) - set(q for q in Q if q != -1)
        for j in newly_created_col_indices:
            for i in range(self.n_rows): 
                new_data[i, j] = fnew(i, j) 
        
        self.data = new_data
        self.n_cols = new_ncols


class IMatrixIndex(IMatrix):
    def __init__(self, f: Callable[[MultiIndex, MultiIndex], Any], 
                 Iset_list: list, Jset_list: list, 
                 dtype: int = Type.Double, device: int = Device.cpu):
        self.Iset = IndexSet(Iset_list)
        self.Jset = IndexSet(Jset_list)
        # Store dtype and device first, as IMatrix.__init__ might use them or they are part of this layer's state
        self.dtype = dtype 
        self.device = device
        # Explicitly call the IMatrix base class __init__
        IMatrix.__init__(self, len(self.Iset), len(self.Jset)) 
        self.A = f 

        
    def _get_original_indices(self, i: int, j: int) -> Tuple[MultiIndex, MultiIndex]:
        if not (0 <= i < len(self.Iset) and 0 <= j < len(self.Jset)):
            raise IndexError(f"IMatrixIndex: Integer index ({i},{j}) out of bounds for IndexSets "
                             f"(len Iset: {len(self.Iset)}, len Jset: {len(self.Jset)}).")
        return self.Iset.from_int()[i], self.Jset.from_int()[j]

    def submat(self, rows: list[int], cols: list[int]) -> list:
        sub_values = []
        for i in rows:
            for j in cols:
                try: 
                    xi, yj = self._get_original_indices(i,j)
                    sub_values.append(self.A(xi,yj))
                except IndexError as e: 
                    logger.error(f"IMatrixIndex.submat: Error getting original indices or calling A for ({i},{j}): {e}")
                    raise # Or handle error, e.g., append a default value or skip
        return sub_values

    def eval(self, ids: list[tuple[int, int]]) -> list:
        vals = []
        for i,j in ids:
            try: 
                xi, yj = self._get_original_indices(i,j)
                vals.append(self.A(xi,yj))
            except IndexError as e: 
                logger.error(f"IMatrixIndex.eval: Error getting original indices or calling A for ({i},{j}): {e}")
                raise
        return vals

    # get_row_as_tensor and get_col_as_tensor are inherited from IMatrix.
    # They will use self.submat (implemented above) and self.dtype/self.device (set in __init__).
        
    def set_rows(self, new_Iset_list: list) -> list[int]:
        old_indices_map = {idx_val: i for i, idx_val in enumerate(self.Iset.from_int())}
        self.Iset = IndexSet(new_Iset_list)
        self.n_rows = len(self.Iset) # Update n_rows from IMatrix base
        new_indices_map = {idx_val: i for i, idx_val in enumerate(self.Iset.from_int())}
        
        P = [-1] * len(old_indices_map) # Permutation for old integer indices
        for old_val, old_int_idx in old_indices_map.items():
            P[old_int_idx] = new_indices_map.get(old_val, -1) # Map to new int_idx or -1 if not found
        return P

    def set_cols(self, new_Jset_list: list) -> list[int]:
        old_indices_map = {idx_val: i for i, idx_val in enumerate(self.Jset.from_int())}
        self.Jset = IndexSet(new_Jset_list)
        self.n_cols = len(self.Jset) # Update n_cols from IMatrix base
        new_indices_map = {idx_val: i for i, idx_val in enumerate(self.Jset.from_int())}

        Q = [-1] * len(old_indices_map) # Permutation for old integer indices
        for old_val, old_int_idx in old_indices_map.items():
            Q[old_int_idx] = new_indices_map.get(old_val, -1)
        return Q


class MatDenseIndex(IMatrixIndex, MatDense): # MatDense should ideally come first for MRO of __init__ if MatDense.__init__ is more primary for data
    def __init__(self, f: Callable[[MultiIndex, MultiIndex], Any], 
                 Iset_list: list, Jset_list: list,
                 dtype: int = Type.Double, device: int = Device.cpu):
        
        # 1. Determine numpy dtype for materialization
        np_dt = np.float64 
        if dtype == Type.ComplexDouble: np_dt = np.complex128
        elif dtype == Type.Float: np_dt = np.float32
        elif dtype == Type.ComplexFloat: np_dt = np.complex64
        # ... add other mappings as needed

        # 2. Materialize data
        if not Iset_list or not Jset_list :
             materialized_data_tensor = zeros((len(Iset_list),len(Jset_list)), dtype=dtype, device=device)
        else:
            try:
                data_list_of_lists = [[f(xi, yj) for yj in Jset_list] for xi in Iset_list]
                np_array_materialized = np.array(data_list_of_lists, dtype=np_dt)
                materialized_data_tensor = from_numpy(np_array_materialized).to(device)
                if materialized_data_tensor.dtype() != dtype:
                    materialized_data_tensor = materialized_data_tensor.astype(dtype)
            except Exception as e:
                logger.error(f"MatDenseIndex __init__: Error materializing data: {e}", exc_info=True)
                materialized_data_tensor = zeros((len(Iset_list), len(Jset_list)), dtype=dtype, device=device)

        # 3. Initialize MatDense part with the materialized data
        MatDense.__init__(self, materialized_data_tensor) 
        # MatDense.__init__ sets self.data, self.n_rows, self.n_cols, self.dtype, self.device

        # 4. Initialize IMatrixIndex part (provides IndexSet handling and original func A)
        # Pass the dtype/device determined by MatDense (from data) to IMatrixIndex
        IMatrixIndex.__init__(self, f, Iset_list, Jset_list, dtype=self.dtype, device=self.device) 
        
        # Consistency check for n_rows/n_cols (should be set by MatDense based on data tensor)
        if self.n_rows != len(Iset_list) or self.n_cols != len(Jset_list):
             logger.warning(f"MatDenseIndex: Dimension mismatch. MatDense dims: ({self.n_rows},{self.n_cols}), "
                            f"IndexSet lengths: ({len(Iset_list)},{len(Jset_list)}). Using MatDense dims.")
             # IMatrixIndex n_rows/n_cols might be re-set here if they were different
             # This can happen if IMatrixIndex.__init__ was called before MatDense.__init__ had set them.
             # The current order is MatDense then IMatrixIndex.
             # IMatrixIndex.__init__ also calls super().__init__ which is IMatrix.__init__
             # Ensure IMatrix.n_rows/n_cols are final and correct.
             # The dimensions should primarily come from the materialized data.

    # Inherits get_row_as_tensor, get_col_as_tensor from MatDense.
    
    def set_rows(self, new_Iset_list: list) -> list[int]:
        # This method needs to update both the IndexSet (from IMatrixIndex)
        # and the dense data array (from MatDense).
        # 1. Update IndexSet and get permutation P (old_int -> new_int or -1)
        P = IMatrixIndex.set_rows(self, new_Iset_list) 
        # self.Iset and self.n_rows (from IMatrix base) are now updated.
        
        # 2. Define fnew for MatDense.set_rows based on the *original* MultiIndex function self.A
        #    and the *new* self.Iset from IMatrixIndex.
        def fnew_for_dense_data(new_integer_row_idx: int, integer_col_idx: int) -> Any:
            # Convert new integer row index and existing integer col index to original MultiIndex types
            new_multi_idx_row = self.Iset.from_int()[new_integer_row_idx] # Uses updated self.Iset
            multi_idx_col = self.Jset.from_int()[integer_col_idx]
            return self.A(new_multi_idx_row, multi_idx_col) # Call original func self.A

        # 3. Update the dense data using MatDense.set_rows
        # MatDense.set_rows expects new_nrows, P (old_int->new_int), and fnew (int,int->val)
        MatDense.set_rows(self, self.n_rows, P, fnew_for_dense_data)
        return P

    def set_cols(self, new_Jset_list: list) -> list[int]:
        Q = IMatrixIndex.set_cols(self, new_Jset_list)
        def fnew_for_dense_data(integer_row_idx: int, new_integer_col_idx: int) -> Any:
            multi_idx_row = self.Iset.from_int()[integer_row_idx]
            new_multi_idx_col = self.Jset.from_int()[new_integer_col_idx] # Uses updated self.Jset
            return self.A(multi_idx_row, new_multi_idx_col)
        MatDense.set_cols(self, self.n_cols, Q, fnew_for_dense_data)
        return Q


class MatLazy(IMatrix):
    def __init__(self, f: Callable[[int, int], Any], n_rows: int, n_cols: int,
                 dtype: int = Type.Double, device: int = Device.cpu):
        # Store dtype and device first
        self.dtype = dtype
        self.device = device
        # Explicitly call the IMatrix base class __init__
        IMatrix.__init__(self, n_rows, n_cols)
        self.f = f 
        self._cache: Dict[Tuple[int, int], Any] = {} 

    def submat(self, rows: list[int], cols: list[int]) -> list:
        return self.eval([(i,j) for i in rows for j in cols])

    def eval(self, ids: list[tuple[int, int]]) -> list:
        out = [None] * len(ids) 
        ids_to_compute_map = {} 
        for original_pos, (i,j) in enumerate(ids):
            if not (0 <= i < self.n_rows and 0 <= j < self.n_cols):
                raise IndexError(f"MatLazy: Eval index ({i},{j}) out of bounds ({self.n_rows}x{self.n_cols}).")
            if (i,j) in self._cache: out[original_pos] = self._cache[(i,j)]
            else: ids_to_compute_map[original_pos] = (i,j)
        
        if ids_to_compute_map:
            # logger.debug(f"MatLazy evaluating {len(ids_to_compute_map)} points.")
            try:
                for original_pos, (i,j) in ids_to_compute_map.items():
                    val = self.f(i,j)
                    self._cache[(i,j)] = val
                    out[original_pos] = val
            except Exception as e: 
                logger.error(f"MatLazy: Error evaluating lazy function f for points {list(ids_to_compute_map.values())}: {e}", exc_info=True)
                raise
        return out # type: ignore 

    # get_row_as_tensor and get_col_as_tensor are inherited from IMatrix.
    # They use self.submat (which uses self.eval here) and self.dtype/self.device.
    # This is generally correct for MatLazy.

    def forget_row(self, i: int): self._cache = {k:v for k,v in self._cache.items() if k[0]!=i}
    def forget_col(self, j: int): self._cache = {k:v for k,v in self._cache.items() if k[1]!=j}
    
    def set_rows(self, new_nrows: int, P: list[int], fnew: Callable[[int, int], Any]):
        new_cache = {}
        if len(P) == self.n_rows: # Check if P length matches old number of rows
            for (old_i,j), val in self._cache.items():
                if old_i < len(P) and P[old_i] != -1: # Check if old_i has a mapping in P
                    new_i = P[old_i]
                    if 0 <= new_i < new_nrows: new_cache[(new_i,j)] = val
        else: 
            logger.warning(f"MatLazy.set_rows: Permutation P length ({len(P)}) mismatch "
                           f"with old n_rows ({self.n_rows}). Clearing cache instead of remapping.")
        self._cache = new_cache
        self.n_rows = new_nrows # Update n_rows from IMatrix base
        self.f = fnew

    def set_cols(self, new_ncols: int, Q: list[int], fnew: Callable[[int, int], Any]):
        new_cache = {}
        if len(Q) == self.n_cols: # Check if Q length matches old number of columns
            for (i,old_j), val in self._cache.items():
                if old_j < len(Q) and Q[old_j] != -1:
                    new_j = Q[old_j]
                    if 0 <= new_j < new_ncols: new_cache[(i,new_j)] = val
        else:
            logger.warning(f"MatLazy.set_cols: Permutation Q length ({len(Q)}) mismatch "
                           f"with old n_cols ({self.n_cols}). Clearing cache instead of remapping.")
        self._cache = new_cache
        self.n_cols = new_ncols # Update n_cols from IMatrix base
        self.f = fnew


class MatLazyIndex(IMatrixIndex, MatLazy): 
    def __init__(self, f: Callable[[MultiIndex, MultiIndex], Any], 
                 Iset_list: list, Jset_list: list,
                 dtype: int = Type.Double, device: int = Device.cpu): 
        
        # 1. Call IMatrixIndex's __init__
        # This will set up self.A, self.Iset, self.Jset, 
        # and also self.n_rows, self.n_cols, self.dtype, self.device via its call to IMatrix.__init__
        IMatrixIndex.__init__(self, f, Iset_list, Jset_list, dtype=dtype, device=device)

        # 2. Define f_int for MatLazy, using attributes set by IMatrixIndex
        def f_int(i: int, j: int) -> Any: 
            try:
                original_i_val, original_j_val = self._get_original_indices(i, j) 
                return self.A(original_i_val, original_j_val) 
            except IndexError: 
                # Propagate error if indices from MatLazy part are somehow out of bounds 
                # for the current Iset/Jset (should not happen if n_rows/n_cols are consistent)
                logger.error(f"MatLazyIndex.f_int: Internal index inconsistency for ({i},{j})")
                raise 
            except Exception as e: 
                logger.error(f"Error in MatLazyIndex's internal f_int({i},{j}): {e}", exc_info=True)
                raise
        
        # 3. Call MatLazy's __init__
        # Pass n_rows, n_cols, dtype, device that were established by IMatrixIndex's initialization chain.
        # MatLazy will call IMatrix.__init__ again. This is generally okay if IMatrix.__init__ is simple.
        # A more advanced super() pattern could avoid this, but direct calls are clearer here.
        MatLazy.__init__(self, f_int, self.n_rows, self.n_cols, dtype=self.dtype, device=self.device)


    # get_row_as_tensor and get_col_as_tensor are inherited from MatLazy.
    # MatLazy.get_row/col_as_tensor calls self.eval, which uses MatLazy.f (which is f_int here).
    # f_int calls IMatrixIndex._get_original_indices and then self.A (the original MultiIndex func).
    # This chain ensures caching and correct index translation.
    
    def set_rows(self, new_Iset_list: list) -> list[int]:
        # 1. Update IMatrixIndex part (updates self.Iset, self.n_rows) and get permutation P
        P = IMatrixIndex.set_rows(self, new_Iset_list) 
        
        # 2. Define the *new* integer-indexed function for MatLazy
        #    This new f_int will use the updated self.Iset (from IMatrixIndex)
        def fnew_int_for_lazy(i:int, j:int) -> Any: 
            # _get_original_indices now uses the updated self.Iset
            xi,yj = self._get_original_indices(i,j) 
            return self.A(xi,yj) # self.A is the original MultiIndex func passed to MatLazyIndex

        # 3. Update MatLazy part (updates MatLazy.f to fnew_int, MatLazy.n_rows, and remaps cache)
        # self.n_rows was updated by IMatrixIndex.set_rows
        MatLazy.set_rows(self, self.n_rows, P, fnew_int_for_lazy) 
        return P

    def set_cols(self, new_Jset_list: list) -> list[int]:
        Q = IMatrixIndex.set_cols(self, new_Jset_list) 
        def fnew_int_for_lazy(i:int, j:int) -> Any:
            xi,yj = self._get_original_indices(i,j) 
            return self.A(xi,yj)
        MatLazy.set_cols(self, self.n_cols, Q, fnew_int_for_lazy)
        return Q
    

# --- Factory Function ---
def make_IMatrix(f: Callable, n_rows_or_Iset: Any, n_cols_or_Jset: Any, 
                 full: bool = False, 
                 dtype: int = Type.Double, device: int = Device.cpu) -> IMatrix:
    is_index_version = isinstance(n_rows_or_Iset, list) or isinstance(n_cols_or_Jset, list)
    
    # Determine default np_dtype from cytnx_dtype for materialization
    np_dt = np.float64
    if dtype == Type.ComplexDouble: np_dt = np.complex128
    elif dtype == Type.Float: np_dt = np.float32
    elif dtype == Type.ComplexFloat: np_dt = np.complex64
    # Add other type mappings if f can return other types

    if is_index_version:
        # Ensure Iset_l and Jset_l are lists of appropriate MultiIndex types if IndexSet expects that
        Iset_l = list(n_rows_or_Iset) if isinstance(n_rows_or_Iset, list) else list(range(n_rows_or_Iset))
        Jset_l = list(n_cols_or_Jset) if isinstance(n_cols_or_Jset, list) else list(range(n_cols_or_Jset))
        
        # If elements are simple integers, wrap them in tuples for MultiIndex consistency if needed by f
        # Example: if f expects tuples like (idx,), and Iset_l is [0,1,2]
        # Iset_l = [(idx,) for idx in Iset_l if isinstance(idx, int)] # Or based on actual MultiIndex format
        # Jset_l = [(idx,) for idx in Jset_l if isinstance(idx, int)]

        if full: 
            return MatDenseIndex(f, Iset_l, Jset_l, dtype=dtype, device=device)
        else: 
            return MatLazyIndex(f, Iset_l, Jset_l, dtype=dtype, device=device)
    else: # Integer indexed
        n_rows, n_cols = int(n_rows_or_Iset), int(n_cols_or_Jset) # Ensure int
        if full:
            if n_rows == 0 or n_cols == 0:
                materialized_data_tensor = Tensor(shape=[n_rows, n_cols], dtype=dtype, device=device, init_zero=True)
            else:
                # f here is Callable[[int,int], Any]
                data_list_of_lists = [[f(i,j) for j in range(n_cols)] for i in range(n_rows)]
                np_materialized_data = np.array(data_list_of_lists, dtype=np_dt) # Use inferred np_dt
                materialized_data_tensor = from_numpy(np_materialized_data).to(device)
                if materialized_data_tensor.dtype() != dtype: # Ensure final cytnx type matches request
                     materialized_data_tensor = materialized_data_tensor.astype(dtype)
            return MatDense(materialized_data_tensor)
        else: # MatLazy from int-indexed function f
            return MatLazy(f, n_rows, n_cols, dtype=dtype, device=device)


def test_matlazyindex():
    print("\n=== 測試 MatLazyIndex ===")
    I = [(0,), (1,), (2,)] 
    J = [(10,), (20,)]
    eval_count = 0
    def f_multi_idx(x_tuple: MultiIndex, y_tuple: MultiIndex) -> float: 
        nonlocal eval_count
        eval_count += 1
        x_val = x_tuple[0] if isinstance(x_tuple, tuple) and x_tuple else 0
        y_val = y_tuple[0] if isinstance(y_tuple, tuple) and y_tuple else 0
        res = x_val * 100 + y_val
        return float(res)

    # MatLazyIndex 的構造函數現在也接受 dtype 和 device
    mat = MatLazyIndex(f_multi_idx, I, J, dtype=Type.Double, device=Device.cpu)
    print(f"MatLazyIndex initial dimensions: {mat.n_rows}x{mat.n_cols}")

    # 要打印 mat 的 dtype 和 device，我們可以直接訪問實例的屬性
    # 因為我們在 __init__ 中存儲了它們
    # 或者，我們可以通過一個由 mat 生成的 tensor 來獲取這些訊息
    # 這裡直接訪問實例屬性更直接
    print(f"MatLazyIndex configured dtype enum: {mat.dtype}, configured device enum: {mat.device}")
    # 如果需要字符串，且 mat 本身是一個 Tensor (它不是，它是 IMatrix)，則用 .dtype_str()
    # 對於 MatLazyIndex，我們存儲了 Type.Double 和 Device.cpu 的整數枚舉值

    print("\nTesting get_row_as_tensor(1)...") 
    row1 = mat.get_row_as_tensor(1) 
    # 現在使用 Tensor 物件的 .dtype_str() 和 .device_str()
    print(f"Row 1 (dtype: {row1.dtype_str()}, device: {row1.device_str()}): {row1.numpy()}")
    assert np.allclose(row1.numpy(), np.array([110., 120.]))

    print("\nTesting get_col_as_tensor(0)...") 
    col0 = mat.get_col_as_tensor(0) 
    print(f"Col 0 (dtype: {col0.dtype_str()}, device: {col0.device_str()}): {col0.numpy()}")
    assert np.allclose(col0.numpy(), np.array([10., 110., 210.]))
    
    print("\nTesting eval([(0,1)])")
    val_0_1 = mat.eval([(0,1)])[0]
    print(f"eval(0,1): {val_0_1}")
    assert abs(val_0_1 - 20.0) < 1e-9

    print(f"\nTotal lazy function calls (f_multi_idx): {eval_count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) 
    test_matlazyindex()