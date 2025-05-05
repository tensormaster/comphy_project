import functools
from typing import Callable, Tuple, Any, Dict, Optional, List, Union
import numpy as np
import time
import itertools
import logging

from IndexSet import IndexSet
MultiIndex = Tuple[int, ...]

logger = logging.getLogger(__name__)

class TensorFunction:
    """
    Wraps a function that computes tensor elements f(vector<int>) -> T,
    providing optional caching via lru_cache and block evaluation.
    Mimics tensor_function.h.
    """
    def __init__(self, func: Callable[[MultiIndex], Any], use_cache: bool = True):
        """
        Initializes the Tensor Function wrapper.

        Args:
            func: A Python function that takes a tuple of integers (multi-index)
                  and returns the tensor element value.
            use_cache: Whether to enable LRU caching for function calls.
        """
        if not callable(func):
            raise TypeError("`func` must be a callable function.")

        self._raw_func = func
        self.use_cache = use_cache
        self._cached_func = None

        if self.use_cache:
            # Using functools.lru_cache for caching
            self._cached_func = functools.lru_cache(maxsize=None)(self._raw_func)
            logger.debug("TensorFunction: LRU Caching enabled.")
        else:
            self._cached_func = self._raw_func
            logger.debug("TensorFunction: Caching disabled.")

        self._eval_count = 0 # Counter for non-cached calls triggered via this class

    def __call__(self, index: MultiIndex) -> Any:
        """
        Evaluates the function for a single multi-index, using the cache if enabled.

        Args:
            index: A tuple of integers representing the multi-index.

        Returns:
            The tensor element value.
        """
        # Ensure index is a tuple for hashability/lru_cache
        if not isinstance(index, tuple):
            # Attempt conversion, handle single element case if necessary
            try: index = tuple(index)
            except TypeError: index = (index,)

        # Directly call the cached (or raw) function
        # We can't easily track cache hits/misses of lru_cache without wrapping it further,
        # but lru_cache handles the core caching logic.
        return self._cached_func(index)

    def eval_block(self, I_indices: List[MultiIndex], J_indices: List[MultiIndex]) -> np.ndarray:
        """
        Evaluates a block of the tensor defined by combinations of row and column multi-indices.
        Mimics C++ eval(vector<MultiIndex> I, vector<MultiIndex> J).

        Args:
            I_indices: A list of multi-indices for the 'row' part.
            J_indices: A list of multi-indices for the 'column' part.

        Returns:
            A NumPy array of shape (len(I_indices), len(J_indices)) containing the results.
        """
        n_rows = len(I_indices)
        n_cols = len(J_indices)
        if n_rows == 0 or n_cols == 0:
            return np.array([]) # Return empty array

        # Pre-allocate result array (find dtype from first element)
        first_full_index = I_indices[0] + J_indices[0]
        first_val = self(first_full_index) # Use __call__ to utilize cache
        results = np.empty((n_rows, n_cols), dtype=type(first_val))
        results[0, 0] = first_val

        # Use itertools.product for cleaner iteration, skip first element
        indices_to_eval = list(itertools.product(range(n_rows), range(n_cols)))[1:]

        eval_start_time = time.time()
        evaluated_count = 1
        # We rely on the underlying self.__call__ using lru_cache here.
        for i, j in indices_to_eval:
            full_index = I_indices[i] + J_indices[j]
            results[i, j] = self(full_index)
            evaluated_count += 1

        eval_duration = time.time() - eval_start_time
        logger.debug(f"eval_block: Evaluated {evaluated_count}/{n_rows*n_cols} elements in {eval_duration:.4f}s.")

        # Cannot easily track non-cached calls here as lru_cache handles it internally
        # self._eval_count += number_of_non_cached_calls # Difficult to get from lru_cache easily

        return results

    def evaluate_list(self, indices: Union[List[MultiIndex], Any], # Keep IndexSet possible?
                      optimize_duplicates: bool = False) -> Union[np.ndarray, List]:
        """
        Evaluates the function for a specific list of multi-indices.

        Args:
            indices: A list/tuple of multi-indices or an IndexSet object.
            optimize_duplicates: If True and input is a list, optimize for duplicate indices.

        Returns:
            A NumPy array containing the results in the input order, or list if conversion fails.
        """
        # (Implementation remains largely the same as before, using self.__call__)
        # ... (Code from previous tensorfucn.py's evaluate_list) ...
        # Ensure it calls self(idx) internally to use the cache
        original_indices: List[MultiIndex]
        # Handle IndexSet input if IndexSet class is available and imported
        is_indexset_input = 'IndexSet' in globals() and isinstance(indices, IndexSet)

        if is_indexset_input:
            original_indices = indices.get_all() # Assumes get_all returns list of tuples
            optimize_duplicates = False # No duplicates in IndexSet input
            logger.debug("evaluate_list: Input is IndexSet, evaluating unique indices.")
        elif isinstance(indices, (list, tuple)):
            original_indices = []
            for idx in indices:
                 if not isinstance(idx, tuple):
                     try: idx = tuple(idx)
                     except TypeError: idx = (idx,) # Handle single elements
                 original_indices.append(idx)
        else:
            raise TypeError("Input 'indices' must be a list/tuple of tuples or an IndexSet object.")

        results = []
        eval_start_time = time.time()

        if optimize_duplicates and not is_indexset_input:
            logger.debug("evaluate_list: Optimizing for duplicates...")
            unique_indices_dict: Dict[MultiIndex, Any] = {}
            unique_indices_list: List[MultiIndex] = []
            for idx in original_indices:
                if idx not in unique_indices_dict:
                    unique_indices_dict[idx] = None
                    unique_indices_list.append(idx)

            logger.debug(f"evaluate_list: Evaluating function for {len(unique_indices_list)} unique indices...")
            for unique_idx in unique_indices_list:
                unique_indices_dict[unique_idx] = self(unique_idx) # Use __call__ for cache

            logger.debug("evaluate_list: Assembling results...")
            results = [unique_indices_dict[idx] for idx in original_indices]

        else:
            logger.debug(f"evaluate_list: Evaluating function for {len(original_indices)} indices (direct iteration)...")
            results = [self(idx) for idx in original_indices]

        eval_duration = time.time() - eval_start_time
        logger.debug(f"evaluate_list: Evaluation took {eval_duration:.4f}s.")

        try:
            result_array = np.array(results)
        except ValueError as e:
            logger.warning(f"evaluate_list: Could not convert results to a uniform NumPy array ({e}). Returning list.")
            return results
        return result_array


    def clear_cache(self):
        """Clears the LRU cache."""
        if self.use_cache and hasattr(self._cached_func, 'cache_clear'):
            self._cached_func.cache_clear()
            logger.info("TensorFunction cache cleared.")
        self._eval_count = 0 # Reset internal counter too

    def cache_info(self):
        """Gets LRU cache information (hits, misses, size)."""
        if self.use_cache and hasattr(self._cached_func, 'cache_info'):
            return self._cached_func.cache_info()
        else:
            return "Cache not enabled or info not available via lru_cache."

    def n_eval(self) -> int:
        """
        Returns an estimate of function evaluations.
        Note: With lru_cache, precisely tracking *new* evaluations triggered
        *through this class* vs total calls handled by the cache is complex.
        This currently returns cache size, which is a lower bound on evaluations.
        """
        if self.use_cache and hasattr(self._cached_func, 'cache_info'):
             # cache_info gives hits/misses, currentsize. Misses = actual calls.
             info = self._cached_func.cache_info()
             # This isn't quite right - misses counts calls since cache creation/clear
             # return info.misses # Might be closer?
             return info.currsize # Number of items currently in cache
        else:
             # Need a manual counter if cache is off
             # The current _eval_count only increments in eval_block/evaluate_list
             # A better way would be to wrap _raw_func again to increment counter
             return self._eval_count # Return manually tracked count (incomplete)

# --- Example Usage (updated) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Show debug messages for testing

    call_counter = 0
    def simple_func(index: MultiIndex) -> float:
        global call_counter
        call_counter += 1
        # time.sleep(0.001) # Simulate work
        return float(sum(x * (i + 1) for i, x in enumerate(index)))

    # Create with caching
    func_runner = TensorFunction(simple_func, use_cache=True)
    print(f"Initial call count: {call_counter}")
    print(f"Cache info: {func_runner.cache_info()}")

    # Test __call__
    print("\n--- Testing __call__ ---")
    print(f"Calling with (1, 2): {func_runner((1, 2))}")
    print(f"Call count after (1, 2): {call_counter}")
    print(f"Calling with (1, 2) again: {func_runner((1, 2))}") # Should use cache
    print(f"Call count after repeat: {call_counter}")
    print(f"Calling with (3, 4): {func_runner((3, 4))}")
    print(f"Call count after (3, 4): {call_counter}")
    print(f"Cache info: {func_runner.cache_info()}")

    # Test eval_block
    print("\n--- Testing eval_block ---")
    I_list = [(0,), (1,)]
    J_list = [(10,), (20,), (30,)]
    # Expected calls: (0,10), (0,20), (0,30), (1,10), (1,20), (1,30) -> 6 calls
    # Except (1,2) was called above, if J_list contained (2,), it might hit cache
    # (1,20) requires index (1,20), func computes 1*1 + 20*2 = 41
    block_result = func_runner.eval_block(I_list, J_list)
    print(f"Block result:\n{block_result}")
    print(f"Call count after block: {call_counter}") # Should increase by 6 (or less if overlap with prev calls)
    print(f"Cache info: {func_runner.cache_info()}")

    # Test eval_block again (should use cache)
    print("\n--- Testing eval_block again ---")
    block_result_2 = func_runner.eval_block(I_list, J_list)
    print(f"Block result 2:\n{block_result_2}")
    print(f"Call count after repeat block: {call_counter}") # Should not increase
    print(f"Cache info: {func_runner.cache_info()}")

    # Test evaluate_list
    print("\n--- Testing evaluate_list ---")
    index_list_eval = [(1, 2), (3, 4), (1, 2), (5, 6), (0, 0), (3, 4), (1, 2)]
    results_list = func_runner.evaluate_list(index_list_eval)
    print(f"evaluate_list results: {results_list}")
    print(f"Call count after list: {call_counter}") # (5,6) and (0,0) should be new calls
    print(f"Cache info: {func_runner.cache_info()}")

    func_runner.clear_cache()
    print("\n--- After Cache Clear ---")
    call_counter = 0 # Reset manual counter for clarity
    print(f"Cache info: {func_runner.cache_info()}")
    print(f"Call count: {call_counter}")
    print(f"Calling (1, 2): {func_runner((1, 2))}")
    print(f"Call count after clear and call: {call_counter}")