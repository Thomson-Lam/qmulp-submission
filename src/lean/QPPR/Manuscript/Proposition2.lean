import QPPR.EC2.QuotientScores
import QPPR.EC2.ClassicalLumpability

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript

/-- Full and reduced coefficient-average measurements coincide provided
    the quotient observable equals the embedded full observable. This
    assumption is displayed rather than silently encoding it as an axiom. -/
theorem proposition2_conditional_measured_average
    {Small Full Labels : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (D : Full →ₗ[ℂ] Full) (Dbar : Small →ₗ[ℂ] Small)
    (B : Small →ₗ[ℂ] Full)
    (hD : ∀ x, D (B x) = B (Dbar x))
    (fullMeasure : (Full × Full) → Labels → ℝ)
    (quotientMeasure : (Small × Small) → Labels → ℝ)
    (hmeasurement : ∀ p a,
      fullMeasure (QPPR.EC2.embedPair B p) a = quotientMeasure p a)
    (p : Small × Small) (tau : ℕ) (a : Labels) :
    QPPR.Core.finiteAverage (QPPR.EC2.recordedCoefficientStep D)
        fullMeasure (QPPR.EC2.embedPair B p) tau a =
      QPPR.Core.finiteAverage (QPPR.EC2.recordedCoefficientStep Dbar)
        quotientMeasure p tau a := by
  exact QPPR.EC2.quotient_finite_measured_scores
    D Dbar B hD fullMeasure quotientMeasure hmeasurement p tau a

/-- A separately verified classical consequence of CG = Gbar C. -/
theorem proposition2_classical_iterates
    {Full Reduced : Type*}
    [AddCommGroup Full] [Module ℝ Full]
    [AddCommGroup Reduced] [Module ℝ Reduced]
    (G : Full →ₗ[ℝ] Full) (Gbar : Reduced →ₗ[ℝ] Reduced)
    (C : Full →ₗ[ℝ] Reduced)
    (hlump : ∀ p, C (G p) = Gbar (C p))
    (p : Full) (t : ℕ) :
    C (QPPR.Core.evolve G t p) =
      QPPR.Core.evolve Gbar t (C p) := by
  exact QPPR.EC2.classical_lumpability_all_times G Gbar C hlump p t

end QPPR.Manuscript
