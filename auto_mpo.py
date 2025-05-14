#filename: auto_mpo.py
import cytnx
import numpy as np
import logging
from typing import Dict, Optional, List
from tensor_train import TensorTrain

logger = logging.getLogger(__name__)

class ProdOp:
    """Product operator: maps site positions to local operators."""
    one: cytnx.Tensor  # class‐level identity

    @classmethod
    def set_identity(cls, d: int) -> None:
        """
        Initialize the static identity operator for all ProdOp instances.
        After this, ProdOp.one.shape() == (d, d).
        """
        cls.one = cytnx.eye(d)  # dtype=Double, device=cpu

    def __init__(self, ops: Optional[Dict[int, cytnx.Tensor]] = None) -> None:
        """
        :param ops: mapping from site index to a (d, d) cytnx.Tensor local operator
        """
        self.ops: Dict[int, cytnx.Tensor] = {}
        if ops is not None:
            for site, op in ops.items():
                if not (isinstance(site, int) and site >= 0):
                    raise ValueError(f"Site index must be a non-negative int, got {site}")
                if not isinstance(op, cytnx.Tensor):
                    raise TypeError(f"Operator at site {site} must be a cytnx.Tensor, got {type(op)}")
                shape = op.shape()
                if len(shape) != 2 or shape[0] != shape[1]:
                    raise ValueError(f"Operator at site {site} must be square 2D tensor, got shape {shape}")
                self.ops[site] = op.clone()

    def length(self) -> int:
        """
        Return the effective length of site indices:
        - 0 if no operators set
        - otherwise max(site indices) + 1
        """
        return 0 if not self.ops else max(self.ops.keys()) + 1

    def to_tensorTrain(self, length_: int) -> TensorTrain:
        """
        Convert this ProdOp into an MPO-style TensorTrain of length `length_`,
        where each core is a (1, d, d) cytnx.Tensor.
        """
        if length_ < 0:
            raise ValueError(f"length_ must be non-negative, got {length_}")
        if not hasattr(ProdOp, 'one') or ProdOp.one is None:
            raise RuntimeError("ProdOp.one is not set. Call ProdOp.set_identity(d) first.")

        d = ProdOp.one.shape()[0]
        cores: List[cytnx.Tensor] = []
        for i in range(length_):
            op = self.ops.get(i, ProdOp.one)
            shape = op.shape()
            if len(shape) != 2 or shape[0] != d or shape[1] != d:
                raise ValueError(f"Operator at site {i} must be ({d}, {d}), got {shape}")
            core = op.reshape(1, d, d)
            logger.debug(f"to_tensorTrain: site {i} → core shape {core.shape()}")
            cores.append(core)
        return TensorTrain(cores)

    def overlap(self, mps: TensorTrain) -> float:
        """
        Compute inner-product between two MPO-style TensorTrains.
        Requires both input TT’s to have cores shaped (1, d, d).
        Returns a scalar overlap via TensorTrain.overlap().
        """
        L = len(mps.M)
        if self.length() and max(self.ops.keys()) >= L:
            raise ValueError(f"Operator has site {max(self.ops.keys())}, but TT length is {L}")
        op_tt = self.to_tensorTrain(L)
        result = mps.overlap(op_tt)
        logger.debug(f"ProdOp.overlap (MPO vs MPO): length={L}, result={result}")
        return result

    def overlap_mps(self, psi: TensorTrain) -> float:
        """
        Compute expectation value ⟨psi|O|psi⟩ for an MPO acting on a physical MPS.
        MPO cores: (1, d, d); MPS cores: (1, d, 1).
        Empty ProdOp → 0.0; out-of-bounds → ValueError.
        """
        if not self.ops:
            return 0.0
        L = len(psi.M)
        if max(self.ops.keys(), default=-1) >= L:
            raise ValueError("Operator out of bounds")
        d = ProdOp.one.shape()[0]
        C = cytnx.eye(1)
        for i in range(L):
            A = (self.ops[i] if i in self.ops else ProdOp.one).reshape(1, d, d)
            ψ = psi.M[i]
            C_new = cytnx.zeros([1, 1])
            for p in range(d):
                for q in range(d):
                    A_pq = A[0, p, q].item()
                    left = ψ[:, p, :].permute(1, 0).Conj()
                    right = ψ[:, q, :]
                    C_new += A_pq * (left @ C @ right)
            C = C_new
        logger.debug(f"ProdOp.overlap_mps: L={L}, result={float(C.item())}")
        return float(C.item())


def build_physical_mps(d: int, L: int) -> TensorTrain:
    """
    Build a “physical” MPS of length L where each core is shape (1, d, 1)
    filled with ones.
    """
    cores = [cytnx.ones([d]).reshape(1, d, 1) for _ in range(L)]
    return TensorTrain(cores)

if __name__ == "__main__":
    d, L = 3, 4
    ProdOp.set_identity(d)
    mps = build_physical_mps(d, L)

    # 1) Empty ProdOp → 0.0
    op_empty = ProdOp()
    print("Empty ProdOp overlap (MPS):", op_empty.overlap_mps(mps))  # expect 0.0

    # 2) Identity MPO → d**L
    ops_id = {i: ProdOp.one for i in range(L)}
    op_id = ProdOp(ops_id)
    id_res = op_id.overlap_mps(mps)
    print("Identity MPO overlap (MPS):", id_res, "expected", d**L)

    # 3) Single‐site scaled MPO → 2.0 * (d * d**(L-1))
    ops_scaled = {2: cytnx.eye(d) * 2.0}
    op_scaled = ProdOp(ops_scaled)
    scaled_res = op_scaled.overlap_mps(mps)
    expected_scaled = 2.0 * (d * (d**(L-1)))
    print("Single‐site scaled MPO overlap (MPS):", scaled_res, "expected", expected_scaled)

    # 4) Out‐of‐bounds operator → ValueError
    try:
        op_oob = ProdOp({L: ProdOp.one})
        op_oob.overlap_mps(mps)
    except ValueError as e:
        print("Caught expected ValueError for out‐of‐bounds:", e)
    else:
        print("ERROR: Expected ValueError for out‐of‐bounds operator")

    print("All overlap_mps tests done.")
