import QPPR.EC2.DiscriminantIntertwining

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC2

/-- Finite source/destination measured probabilities of the quotient agree
    with the full coefficient walk provided the embedded block measurement
    equals the quotient measurement on every coefficient vector. -/
theorem quotient_finite_measured_scores
    {Small Full Labels : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (D : Full →ₗ[ℂ] Full) (Dbar : Small →ₗ[ℂ] Small)
    (B : Small →ₗ[ℂ] Full)
    (hD : ∀ x, D (B x) = B (Dbar x))
    (fullBlockMeasure : (Full × Full) → Labels → ℝ)
    (quotientMeasure : (Small × Small) → Labels → ℝ)
    (hmeasure : ∀ p a,
      fullBlockMeasure (embedPair B p) a = quotientMeasure p a)
    (p : Small × Small) (τ : ℕ) (a : Labels) :
    QPPR.Core.finiteAverage (recordedCoefficientStep D)
        fullBlockMeasure (embedPair B p) τ a =
      QPPR.Core.finiteAverage (recordedCoefficientStep Dbar)
        quotientMeasure p τ a := by
  exact finite_averages_agree
    (recordedCoefficientStep Dbar) (recordedCoefficientStep D)
    (embedPair B) quotientMeasure fullBlockMeasure
    (recordedCoefficientStep_intertwining D Dbar B hD)
    hmeasure τ p a

end QPPR.EC2
