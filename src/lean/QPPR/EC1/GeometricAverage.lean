import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC1

/-- The sum ∑_{t=0}^{n-1} z^t, written recursively to avoid indexing conventions. -/
def geometricSum (z : ℂ) : ℕ → ℂ
  | 0 => 0
  | n + 1 => geometricSum z n + z ^ n

/-- The exact geometric-series identity used for the off-diagonal spectral terms. -/
theorem geometricSum_identity (z : ℂ) (n : ℕ) :
    (1 - z) * geometricSum z n = 1 - z ^ n := by
  induction n with
  | zero =>
      simp [geometricSum]
  | succ n ih =>
      calc
        (1 - z) * geometricSum z (n + 1) =
            (1 - z) * geometricSum z n + (1 - z) * z ^ n := by
              rw [geometricSum]
              ring
        _ = 1 - z ^ n + (1 - z) * z ^ n := by rw [ih]
        _ = 1 - z ^ (n + 1) := by rw [pow_succ]; ring

/-- When both eigenvalues coincide, their ratio is 1 and the sum is n. -/
theorem geometricSum_one (n : ℕ) : geometricSum 1 n = (n : ℂ) := by
  induction n with
  | zero =>
      simp [geometricSum]
  | succ n ih =>
      calc
        geometricSum 1 (n + 1) = geometricSum 1 n + (1 : ℂ) ^ n := rfl
        _ = (n : ℂ) + 1 := by rw [ih]; simp
        _ = ((n + 1 : ℕ) : ℂ) := by simp

end QPPR.EC1
