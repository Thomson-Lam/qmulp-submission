import QPPR.EC3.Symmetry

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC3

/-- An exchange-invariant classical transition G with exchange-invariant
    initial distribution gives identical finite scores at swapped labels. -/
theorem symmetric_classical_finite_scores
    {V Labels : Type*} [AddCommGroup V] [Module ℝ V]
    (G Q : V →ₗ[ℝ] V)
    (hGQ : ∀ v, Q (G v) = G (Q v))
    (initial : V) (hinitial : Q initial = initial)
    (swap : Labels → Labels)
    (measure : V → Labels → ℝ)
    (hmeasure : ∀ s i, measure (Q s) i = measure s (swap i))
    (τ : ℕ) (i : Labels) :
    QPPR.Core.finiteAverage G measure initial τ i =
      QPPR.Core.finiteAverage G measure initial τ (swap i) := by
  exact symmetric_finite_averages_equal
    G Q swap measure hGQ hmeasure initial hinitial τ i

end QPPR.EC3
