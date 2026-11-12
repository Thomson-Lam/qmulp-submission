import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC1

/-- If both approximate values differ from their true values by at most ε,
    a true gap strictly greater than 2ε preserves their ordering. -/
theorem pairwise_rank_stability
    (a b aHat bHat ε : ℝ)
    (ha : |aHat - a| ≤ ε)
    (hb : |bHat - b| ≤ ε)
    (hgap : 2 * ε < a - b) :
    bHat < aHat := by
  rcases abs_le.mp ha with ⟨haLower, haUpper⟩
  rcases abs_le.mp hb with ⟨hbLower, hbUpper⟩
  linarith

/-- Version directly applicable to two indexed QPPR score vectors. -/
theorem rank_stability_of_uniform_error
    {ι : Type*} (exact approximate : ι → ℝ) (ε : ℝ)
    (herror : ∀ i, |approximate i - exact i| ≤ ε)
    (i j : ι) (hgap : 2 * ε < exact i - exact j) :
    approximate j < approximate i := by
  exact pairwise_rank_stability
    (exact i) (exact j) (approximate i) (approximate j) ε
    (herror i) (herror j) hgap

end QPPR.EC1
