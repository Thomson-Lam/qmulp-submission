import QPPR.EC3.DiscriminantSymmetry
import QPPR.EC3.ClassicalSymmetry
import QPPR.EC3.LimitSymmetry

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript

/-- Quantum category equality at every finite recorded horizon. -/
theorem proposition3_finite_quantum_equality
    {V Labels : Type*} [AddCommGroup V] [Module ℂ V]
    (D Q : V →ₗ[ℂ] V)
    (hDQ : ∀ v, D (Q v) = Q (D v))
    (initial : V × V) (hinit : QPPR.EC2.embedPair Q initial = initial)
    (swapLabels : Labels → Labels)
    (measurement : (V × V) → Labels → ℝ)
    (hmeasurement : ∀ s i,
      measurement (QPPR.EC2.embedPair Q s) i =
        measurement s (swapLabels i))
    (tau : ℕ) (i : Labels) :
    QPPR.Core.finiteAverage (QPPR.EC2.recordedCoefficientStep D)
        measurement initial tau i =
      QPPR.Core.finiteAverage (QPPR.EC2.recordedCoefficientStep D)
        measurement initial tau (swapLabels i) := by
  exact QPPR.EC3.symmetric_QPPR_finite_scores D Q hDQ initial hinit
    swapLabels measurement hmeasurement tau i

/-- Symmetric finite score trajectories have the same limit, when it exists. -/
theorem proposition3_equality_of_existing_limits
    (scoreA scoreB : ℕ → ℝ) (limitA limitB : ℝ)
    (hfinite : ∀ tau, scoreA tau = scoreB tau)
    (hA : Filter.Tendsto scoreA Filter.atTop (nhds limitA))
    (hB : Filter.Tendsto scoreB Filter.atTop (nhds limitB)) :
    limitA = limitB := by
  exact QPPR.EC3.symmetry_equal_limiting_scores
    scoreA scoreB limitA limitB hfinite hA hB

end QPPR.Manuscript
