import QPPR.EC1.GeometricAverage

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC1

/-- Off-diagonal finite geometric series is exactly (1-z^n)/(1-z). -/
theorem geometricSum_closed_form
    (z : ℂ) (n : ℕ) (hz : z ≠ 1) :
    geometricSum z n = (1 - z ^ n) / (1 - z) := by
  have hden : (1 : ℂ) - z ≠ 0 := sub_ne_zero.mpr (Ne.symm hz)
  apply (eq_div_iff hden).2
  calc
    geometricSum z n * (1 - z) = (1 - z) * geometricSum z n := by ring
    _ = 1 - z ^ n := geometricSum_identity z n

end QPPR.EC1
