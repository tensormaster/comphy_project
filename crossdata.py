import cytnx
from cytnx import *
import numpy as np
from AdaptiveLU import AdaptiveLU  # Assuming AdaptiveLU is in AdaptiveLU.py

def ensure_2d(t: Tensor, axis=0) -> Tensor:
    """
    Ensure the input Tensor t is at least 2D. If 1D, reshape it based on the axis value.
    """
    shape = t.shape()
    if len(shape) == 1:
        n = shape[0]
        if axis == 0:
            t2 = t.reshape(n, 1)
        else:
            t2 = t.reshape(1, n)
        print(f"[DEBUG] ensure_2d: reshaped tensor from {shape} to {t2.shape()}")
        return t2
    return t


def manual_hstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    """
    Manually stack two tensors horizontally using basic cytnx operations.
    
    Args:
        tensor1: First cytnx.Tensor with shape (m, n)
        tensor2: Second cytnx.Tensor with shape (m, p)
    
    Returns:
        cytnx.Tensor with shape (m, n + p)
    """
    # Get shapes
    m, n = tensor1.shape()
    p = tensor2.shape()[1]
    
    # Verify that the number of rows matches
    if tensor1.shape()[0] != tensor2.shape()[0]:
        raise ValueError("Tensors must have the same number of rows for horizontal stacking.")
    
    # Create a new tensor with the combined shape
    result = cytnx.zeros((m, n + p))
    
    # Copy data from tensor1 to the first n columns
    result[:, :n] = tensor1
    
    # Copy data from tensor2 to the next p columns
    result[:, n:n + p] = tensor2
    
    return result

def manual_vstack(tensor1: cytnx.Tensor, tensor2: cytnx.Tensor) -> cytnx.Tensor:
    """
    Manually stack two tensors vertically using basic cytnx operations.
    
    Args:
        tensor1: First cytnx.Tensor with shape (m, n)
        tensor2: Second cytnx.Tensor with shape (p, n)
    
    Returns:
        cytnx.Tensor with shape (m + p, n)
    """
    # Get shapes
    m, n = tensor1.shape()
    p = tensor2.shape()[0]
    
    # Verify that the number of columns matches
    if tensor1.shape()[1] != tensor2.shape()[1]:
        raise ValueError("Tensors must have the same number of columns for vertical stacking.")
    
    # Create a new tensor with the combined shape
    result = cytnx.zeros((m + p, n))
    
    # Copy data from tensor1 to the first m rows
    result[:m, :] = tensor1
    
    # Copy data from tensor2 to the next p rows
    result[m:m + p, :] = tensor2
    
    return result


class CrossData:
    def __init__(self, n_rows: int, n_cols: int):
        """
        Initialize an empty CrossData instance.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.C = None  # Column submatrix
        self.R = None # Row submatrix
        self.lu = AdaptiveLU(n_rows, n_cols)  # AdaptiveLU instance for decomposition
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}  # Cache for computed values
        print(f"[DEBUG] Initialized CrossData with {n_rows} rows and {n_cols} columns.")

    def pivotMat(self) -> Tensor:
        """
        Return the pivot matrix, i.e., rows of C corresponding to lu.Iset.
        """
        if self.C is None or not self.lu.Iset:
            print("[DEBUG] pivotMat: No pivot set found, returning None.")
            return None
        Iset_uvec = np.array(self.lu.Iset, dtype=np.uint32)
        pivot_matrix = self.C[Iset_uvec, :]
        print(f"[DEBUG] pivotMat: pivot matrix shape: {pivot_matrix.shape()}")
        return pivot_matrix

    def leftMat(self) -> Tensor:
        """
        Return the left matrix L * D (cached).
        """
        k = self.rank()
        if self.cache['LD'] is None:
            print(f"[DEBUG] leftMat: self.lu.D shape: {self.lu.D.shape()}")
            D_diag = cytnx.linalg.Diag(self.lu.D)[:k]
            print(f"[DEBUG] leftMat: D_diag:",D_diag)
            D_diag_matrix = cytnx.linalg.Diag(D_diag)
            print(f"[DEBUG] leftMat D_diag_matrix:",D_diag_matrix)
            self.cache['LD'] = self.lu.L[:,:k] @ D_diag_matrix  # Use D_diag_matrix here
            print("[DEBUG] leftMat: Computed new left matrix (L * D).")
        else:
            print("[DEBUG] leftMat: Using cached left matrix.")
        print(f"[DEBUG] leftMat: shape: {self.cache['LD'].shape()}")
        return self.cache['LD']

    def rightMat(self) -> Tensor:
        """
        Return the right matrix U.
        """
        k = self.rank()
        print(f"[DEBUG] rightMat: shape: {self.lu.U.shape()}")
        return self.lu.U[:k,:]

    def availRows(self) -> list:
        """
        Return available row indices not yet in lu.Iset (cached).
        """
        if self.cache['I_avail'] is None:
            self.cache['I_avail'] = [i for i in range(self.n_rows) if i not in self.lu.Iset]
            print(f"[DEBUG] availRows: Computed available rows: {self.cache['I_avail']}")
        else:
            print("[DEBUG] availRows: Using cached available rows.")
        return self.cache['I_avail']

    def availCols(self) -> list:
        """
        Return available column indices not yet in lu.Jset (cached).
        """
        if self.cache['J_avail'] is None:
            self.cache['J_avail'] = [j for j in range(self.n_cols) if j not in self.lu.Jset]
            print(f"[DEBUG] availCols: Computed available columns: {self.cache['J_avail']}")
        else:
            print("[DEBUG] availCols: Using cached available columns.")
        return self.cache['J_avail']

    def rank(self) -> int:
        """
        Return the current rank of the decomposition.
        """
        current_rank = len(self.lu.Iset)
        print(f"[DEBUG] rank: Current rank is {current_rank}")
        return current_rank

    def firstPivotValue(self) -> float:
        """
        Return the value of the first pivot.
        """
        if self.C is None or not self.lu.Iset:
            print("[DEBUG] firstPivotValue: No pivot found, returning 1.0")
            return 1.0
        pivot_value = self.C[self.lu.Iset[0], 0].item()
        print(f"[DEBUG] firstPivotValue: The first pivot value is {pivot_value}")
        return pivot_value

    def eval(self, i: int, j: int) -> float:
        """
        Compute the approximate value at position (i, j) using the cross formula Aapprox = C * P^-1 * R.
        """
        if self.C is None or self.R is None:
            print(f"[DEBUG] eval: Either C or R is None, returning 0.0 for position ({i}, {j})")
            return 0.0
        val = (self.leftMat()[i, :] @ self.rightMat()[:, j]).item()
        print(f"[DEBUG] eval: Evaluated value at ({i}, {j}) is {val}")
        return val

    def addPivot(self, i: int, j: int, A: Tensor):
        """
        Update the cross data by adding a new pivot at position (i, j) of matrix A.
        """
        print(f"[DEBUG] addPivot: Adding pivot at position ({i}, {j}).")
        A = ensure_2d(A, axis=0)
        self.addPivotRow(i, A)
        self.addPivotCol(j, A)
        print(f"[DEBUG] addPivot: Updated LU and matrices with new pivot at ({i}, {j}).")

    def addPivotRow(self, i: int, A: Tensor):
        """
        Add row i of matrix A to R and update lu.
        """
        print(f"[DEBUG] addPivotRow: Adding row {i} to R.")
        A = ensure_2d(A, axis=0)
        row = A[i, :]
        print(f"[DEBUG] addPivotRow: Row shape: {row.shape()}")
        if self.R is None:
            self.R = row.reshape(1, self.n_cols)
            print(f"[DEBUG] addPivotRow: Initialized R with shape {self.R.shape()}.")
        else:
            # Ensure row is 2D with shape (1, n_cols) for vertical stacking
            row_2d = row.reshape(1, self.n_cols)
            self.R = manual_vstack(self.R, row_2d)
            print(f"[DEBUG] addPivotRow: Updated R, new shape: {self.R.shape()}.")
        self.lu.add_pivot_row(i, row)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}  # Clear cache

    def addPivotCol(self, j: int, A: Tensor):
        """
        Add column j of matrix A to C and update lu.
        """
        print(f"[DEBUG] addPivotCol: Adding column {j} to C.")
        A = ensure_2d(A, axis=0)
        col = A[:, j]
        print(f"[DEBUG] addPivotCol: Column shape: {col.shape()}")
        if self.C is None:
            self.C = col.reshape(self.n_rows, 1)
            print(f"[DEBUG] addPivotCol: Initialized C with shape {self.C.shape()}.")
        else:
            # Ensure col is 2D with shape (n_rows, 1) for horizontal stacking
            col_2d = col.reshape(self.n_rows, 1)
            self.C = manual_hstack(self.C, col_2d)
            print(f"[DEBUG] addPivotCol: Updated C, new shape: {self.C.shape()}.")
        self.lu.add_pivot_col(j, col)
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}  # Clear cache

    def mat(self, A: Tensor) -> Tensor:
        """
        Return the full approximated matrix Aapprox = C * P^-1 * R.
        
        Args:
            A (Tensor): The original matrix, needed to compute P.
        
        Returns:
            Tensor: Approximated matrix with shape (n_rows, n_cols).
        """
        if self.C is None or self.R is None:
            print("[DEBUG] mat: Either C or R is None, returning zero matrix.")
            return cytnx.zeros((self.n_rows, self.n_cols))
        
        # Get pivot indices
        Iset = np.array(self.lu.Iset, dtype=np.float128).tolist()
        Jset = np.array(self.lu.Jset, dtype=np.float128).tolist()
        
        # Compute P = A[Iset, Jset]
        P = A[Iset, :][:, Jset]  # Shape: (k, k), e.g., (1, 1)
        print(f"[DEBUG] mat: P shape: {P.shape()}")
        
        # Compute P_inv
        P_inv = cytnx.linalg.Inv(P)  # Shape: (k, k)
        print(f"[DEBUG] mat: P_inv shape: {P_inv.shape()}")
        
        # Compute left = C @ P_inv
        left = self.C @ P_inv  # Shape: (n_rows, k), e.g., (5, 1)
        print(f"[DEBUG] mat: left shape: {left.shape()}")
        
        # Compute approx_matrix = left @ R
        approx_matrix = left @ self.R  # Shape: (n_rows, n_cols), e.g., (5, 5)
        print(f"[DEBUG] mat: Approximated matrix shape: {approx_matrix.shape()}")
        
        return approx_matrix

    def eval_multiple(self, rows: list, cols: list) -> list:
        """
        Compute approximate values for multiple positions.
        """
        print(f"[DEBUG] eval_multiple: Evaluating multiple entries for rows {rows} and cols {cols}.")
        if self.C is None or self.R is None or len(rows) != len(cols):
            print("[DEBUG] eval_multiple: Mismatch in rows and columns or C/R is None. Returning zeros.")
            return [0.0] * len(rows)
        left = self.leftMat()[rows, :]
        right = self.rightMat()[:, cols]
        values = []
        for i in range(len(rows)):
            val = (left[i, :] @ right[:, i]).item()
            print(f"[DEBUG] eval_multiple: Value at position ({rows[i]}, {cols[i]}) is {val}")
            values.append(val)
        return values
    
    def row(self, i: int) -> cytnx.Tensor:
        """
        Return the approximate i-th row.
        """
        print(f"[DEBUG] row: Retrieving row {i}.")
        if self.C is None or self.R is None:
            print("[DEBUG] row: Either C or R is None, returning zero row.")
            return cytnx.zeros(self.n_cols)
        row_val = self.leftMat()[i, :] @ self.rightMat()
        print(f"[DEBUG] row: Retrieved row {i} with shape: {row_val.shape()}")
        return row_val

    def col(self, j: int) -> cytnx.Tensor:
        """
        Return the approximate j-th column.
        """
        print(f"[DEBUG] col: Retrieving column {j}.")
        if self.C is None or self.R is None:
            print("[DEBUG] col: Either C or R is None, returning zero column.")
            return cytnx.zeros(self.n_rows)
        col_val = self.leftMat() @ self.rightMat()[:, j]
        print(f"[DEBUG] col: Retrieved column {j} with shape: {col_val.shape()}")
        return col_val

    def submat(self, rows: list, cols: list) -> cytnx.Tensor:
        """
        Return the approximate submatrix defined by the given rows and columns.
        """
        print(f"[DEBUG] submat: Extracting submatrix with rows {rows} and columns {cols}.")
        if self.C is None or self.R is None:
            print("[DEBUG] submat: Either C or R is None, returning zero matrix.")
            return cytnx.zeros((len(rows), len(cols)))
        left_sub = self.leftMat()[rows, :]
        right_sub = self.rightMat()[:, cols]
        submatrix = left_sub @ right_sub
        print(f"[DEBUG] submat: Extracted submatrix shape: {submatrix.shape()}")
        return submatrix

    def setRows(self, C_new: cytnx.Tensor, P: list):
        """
        Increase the rows of the matrix according to C_new, reordering old rows per P.
        """
        print("[DEBUG] setRows: Setting new rows and reordering according to permutation P.")
        self.n_rows = C_new.shape[0]
        self.lu.Iset = [P[i] for i in self.lu.Iset]
        L_reordered = self.lu.L[P, :]
        new_L = cytnx.zeros((self.n_rows, self.lu.L.shape[1]))
        new_L[:len(P), :] = L_reordered
        Pc = [i for i in range(self.n_rows) if i not in P]
        for k in range(self.lu.rank()):
            new_L[Pc, k] = C_new[Pc, k]
            for l in range(k):
                new_L[Pc, k] -= new_L[Pc, l] * (self.lu.U[l, self.lu.Jset[k]].item() * self.lu.D[l, l].item())
        self.lu.L = new_L
        self.C = C_new
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        print(f"[DEBUG] setRows: New L matrix shape: {self.lu.L.shape()}, C shape: {self.C.shape()}")

    def setCols(self, R_new: cytnx.Tensor, Q: list):
        """
        Increase the columns of the matrix according to R_new, reordering old columns per Q.
        """
        print("[DEBUG] setCols: Setting new columns and reordering according to permutation Q.")
        self.n_cols = R_new.shape[1]
        self.lu.Jset = [Q[j] for j in self.lu.Jset]
        U_reordered = self.lu.U[:, Q]
        new_U = cytnx.zeros((self.lu.U.shape[0], self.n_cols))
        new_U[:, :len(Q)] = U_reordered
        Qc = [j for j in range(self.n_cols) if j not in Q]
        for k in range(self.lu.rank()):
            new_U[k, Qc] = R_new[k, Qc]
            for l in range(k):
                new_U[k, Qc] -= (self.lu.L[self.lu.Iset[k], l].item() * self.lu.D[l, l].item()) * new_U[l, Qc]
        self.lu.U = new_U
        self.R = R_new
        self.cache = {'LD': None, 'I_avail': None, 'J_avail': None}
        print(f"[DEBUG] setCols: New U matrix shape: {self.lu.U.shape()}, R shape: {self.R.shape()}")

    def estimate_error(self, A: cytnx.Tensor, sample_size: int = 100) -> float:
        """
        Estimate the approximation error using a random sample of positions.
        """
        print("[DEBUG] estimate_error: Estimating approximation error using random sampling.")
        import random
        import numpy as np
        if self.C is None or self.R is None:
            print("[DEBUG] estimate_error: Either C or R is None, error is infinite.")
            return float('inf')
        errors = []
        for _ in range(sample_size):
            i = random.randint(0, self.n_rows - 1)
            j = random.randint(0, self.n_cols - 1)
            approx_val = self.eval(i, j)
            true_val = A[i, j].item()
            error_val = (approx_val - true_val) ** 2
            errors.append(error_val)
            print(f"[DEBUG] estimate_error: At ({i}, {j}), approx: {approx_val}, true: {true_val}, squared error: {error_val}")
        rmse = np.sqrt(np.mean(errors))
        print(f"[DEBUG] estimate_error: Estimated RMSE is {rmse}")
        return rmse

# Test script
if __name__ == "__main__":
    print("[TEST] Starting debug of CrossData with random matrix.")
    # Create a random 5x5 matrix
    M, N = 5, 5
    np.random.seed(0)  # For reproducibility
    arr = np.random.rand(M, N)
    A = from_numpy(arr)
    print("[TEST] Original matrix A:")
    print(A)

    # Initialize CrossData with matrix dimensions
    cross_data = CrossData(M, N)

    # Add pivots iteratively along the diagonal and monitor the approximation
    print("\n[TEST] Adding pivots and tracking approximation error:")
    for k in range(min(M, N)):
        print(f"\n[TEST] --- Iteration {k} ---")
        cross_data.addPivot(k, k, A)  # Add pivot at position (k, k)
        rank = cross_data.rank()      # Get the current rank
        approx_matrix = cross_data.mat(A)  # Get the current approximated matrix
        error = cytnx.linalg.Norm(A - approx_matrix, 'f').item()  # Frobenius norm of the error
        print(f"[TEST] Approximation error (Frobenius norm): {error}")
        print(f"[TEST] Pivot ({k}, {k}) added: rank = {rank}, Frobenius error = {error:.6e}")
        print("[TEST] Current approximated matrix:")
        print(approx_matrix)

    # Final evaluation and comparison
    final_approx_matrix = cross_data.mat()
    final_error_matrix = A - final_approx_matrix
    max_error = cytnx.linalg.Norm(final_error_matrix, 'inf').item()  # Maximum absolute error

    print("\n[TEST] Final maximum absolute error:", f"{max_error:.6e}")
    print("\n[TEST] Final approximated matrix:\n", final_approx_matrix)
    print("\n[TEST] Original matrix:\n", A)
