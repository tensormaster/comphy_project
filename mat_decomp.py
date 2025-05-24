# filename: mat_decomp.py
from cytnx_prrLU import RankRevealingLU
# -*- coding: utf-8 -*-
"""mat_decomp.py
================
`MatprrLUFixedTol` – Python port of the *pivot rank‑revealing LU with
fixed tolerance* functor defined in **mat_decomp.h**.  This is a *thin*
wrapper around :class:`cytnx_prrLU.RankRevealingLU`, exposing an API that
mirrors the original C++ template‑functor behaviour (i.e. "call me and
I return a `(left, right)` tuple suitable for low‑rank reconstruction").

Only this single functor is kept, as requested.  The other QR/SVD/CUR
variants were removed for brevity.
"""


from typing import Tuple, Any
import numpy as np
from cytnx import from_numpy

# --- external deps ---
try:
    import cytnx
    from cytnx import UniTensor, Tensor, Type, Device  # type: ignore
except ImportError as exc:  # pragma: no cover – fails in pure‑numpy env
    raise RuntimeError(
        "cytnx and cytnx_prrLU must be installed to use MatprrLUFixedTol"
    ) from exc


# ---------------------------------------------------------------------------
# helper utilities -----------------------------------------------------------
# ---------------------------------------------------------------------------

ArrayLike = Any  # numpy ndarray, cytnx Tensor, or anything convertible via np.asarray

# ---------------------------------------------------------------------------
# main functor ---------------------------------------------------------------
# ---------------------------------------------------------------------------


class MatprrLUFixedTol:
    """Rank‑revealing LU with fixed tolerance (pivot version).

    Parameters
    ----------
    tol
        Relative tolerance for truncating the rank.  ``tol<=0`` disables
        tolerance‑based early stopping (uses full `min(m, n)` pivots or
        `rankMax` if positive).
    rankMax
        Maximum rank to keep (``0`` → no explicit cap).
    pivot_method
        Either ``"full"`` (full pivot search, default) or ``"rook"``.
    nRookIter
        Number of rook iterations when *pivot_method* is ``"rook"``.
    """

    # ------------------------------------------------------------------
    # ctor
    # ------------------------------------------------------------------

    def __init__(
        self,
        tol: float = 1e-12,
        rankMax: int = 0,
        *,
        pivot_method: str = "full",
        nRookIter: int = 5,
    ) -> None:
        self.tol = tol
        self.rankMax = rankMax
        self.pivot_method = pivot_method.lower()
        self.nRookIter = nRookIter

        # outputs / state after a call ----------------------------------
        self.Iperm: list[int] | None = None  # row permutation
        self.Jperm: list[int] | None = None  # col permutation
        self.rank: int | None = None
        self.error: float | None = None
        # Keep raw matrices for debugging
        self.L_: np.ndarray | None = None
        self.D_: np.ndarray | None = None
        self.U_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        A: ArrayLike,
        *,
        leftOrthogonal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the decomposition and return ``(left, right)``.

        The returned matrices satisfy, up to numerical precision::

            left @ right  ≈  A

        with ``left`` either *orthogonal* (if *leftOrthogonal* is
        ``True`` – the usual choice) or its counterpart depending on the
        variant chosen in the original C++ code.
        """
        # 1. normalise input -------------------------------------------
        cy_A = self._as_cytnx_tensor(A)

        # 2. run cytnx_prrLU ------------------------------------------
        rr = RankRevealingLU(
            UniTensor(cy_A, rowrank=2, is_diag=False),
            verbose=False,
            pivot_method=self.pivot_method,
            tol=self.tol,
        )
        # NOTE: `.PrrLU()` triggers the decomposition in‑place and
        # returns (L, D, U) **as UniTensor** instances.
        ut_L, ut_D, ut_U = rr.PrrLU()

        # store conversions to ndarray ---------------------------------
        self.L_ = ut_L.get_block()
        self.D_ = ut_D.get_block()
        self.U_ = ut_U.get_block()
        

        # post‑info -----------------------------------------------------
        self.rank = rr.get_rank_info()[-1] if rr.get_rank_info() else 0
        self.error = rr.get_error()
        # permutations are kept internally by RankRevealingLU as
        # `pivot_set`; we expose them as simple lists (optional)
        self.Iperm = [p[0] for p in rr.pivot_set.get_all()]  # type: ignore[attr-defined]
        self.Jperm = [p[1] for p in rr.pivot_set.get_all()]  # type: ignore[attr-defined]

        # 3. build (left, right) --------------------------------------
        if leftOrthogonal:
            left = self.L_
            right = self.D_ @ self.U_
        else:
            left = self.L_ @ self.D_
            right = self.U_
        return left, right

    # ------------------------------------------------------------------
    # helpers -----------------------------------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _as_cytnx_tensor(x: ArrayLike) -> Tensor:
        """Convert *x* into a *cytnx Tensor* (2‑D)."""
        if isinstance(x, Tensor):
            # Ensure 2‑D (reshape if 1‑D)
            if len(x.shape()) == 1:
                return x.reshape(x.shape()[0], 1)
            return x
        if isinstance(x, np.ndarray):
            t = cytnx.from_numpy(x.astype(float, copy=False))
            return t
        raise TypeError(
            "Input must be a NumPy ndarray or cytnx Tensor (got "
            f"{type(x)})."
        )


if __name__ == "__main__":
    # Test of file
    dec = MatprrLUFixedTol(tol=1e-8, rankMax=50)
    # 測試用矩陣
    M, N = 6, 5
    np.random.seed(0)
    arr = np.random.rand(M, N)
    T = from_numpy(arr)
    L, R = dec(T)            # A: numpy ndarray or cytnx Tensor
    A_approx = L @ R
    print(dec.rank, dec.error)
    print("type of L:", type(L))
    print("type of R:", type(R))
    print("type of A_approx:", type(A_approx))
    print("shape of L:", L.shape)
    print("shape of R:", R.shape)
