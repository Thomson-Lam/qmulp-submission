import QPPR.EC2.DiscriminantIntertwining
import QPPR.EC3.Symmetry

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC3

open QPPR.EC2

/-- Symmetry of the discriminant extends to symmetry of coefficient U=W². -/
theorem symmetry_commutes_with_recorded_step
    {V : Type*} [AddCommGroup V] [Module ℂ V]
    (D Q : V →ₗ[ℂ] V)
    (hDQ : ∀ v, D (Q v) = Q (D v))
    (p : V × V) :
    embedPair Q (recordedCoefficientStep D p) =
      recordedCoefficientStep D (embedPair Q p) := by
  exact recordedCoefficientStep_intertwining D D Q hDQ p

/-- If Q fixes both initial coefficient vectors and commutes with D,
    then Q fixes the coefficients at every recorded QPPR time. -/
theorem symmetry_of_QPPR_coefficients
    {V : Type*} [AddCommGroup V] [Module ℂ V]
    (D Q : V →ₗ[ℂ] V)
    (hDQ : ∀ v, D (Q v) = Q (D v))
    (p : V × V) (hp : embedPair Q p = p)
    (t : ℕ) :
    embedPair Q (QPPR.Core.evolve (recordedCoefficientStep D) t p) =
      QPPR.Core.evolve (recordedCoefficientStep D) t p := by
  exact symmetry_preserved_under_evolution
    (recordedCoefficientStep D) (embedPair Q)
    (symmetry_commutes_with_recorded_step D Q hDQ)
    p hp t

/-- A physically equivariant observable assigns the same score to
    individuals related by the operator-preserving permutation, at
    every finite averaging horizon. -/
theorem symmetric_QPPR_finite_scores
    {V Labels : Type*} [AddCommGroup V] [Module ℂ V]
    (D Q : V →ₗ[ℂ] V)
    (hDQ : ∀ v, D (Q v) = Q (D v))
    (p : V × V) (hp : embedPair Q p = p)
    (labelSwap : Labels → Labels)
    (measure : (V × V) → Labels → ℝ)
    (hmeasure : ∀ s i, measure (embedPair Q s) i = measure s (labelSwap i))
    (τ : ℕ) (i : Labels) :
    QPPR.Core.finiteAverage (recordedCoefficientStep D) measure p τ i =
      QPPR.Core.finiteAverage (recordedCoefficientStep D) measure p τ
        (labelSwap i) := by
  exact symmetric_finite_averages_equal
    (recordedCoefficientStep D) (embedPair Q)
    labelSwap measure
    (symmetry_commutes_with_recorded_step D Q hDQ)
    hmeasure p hp τ i

end QPPR.EC3
