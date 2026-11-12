import QPPR.EC2.DiscriminantIntertwining

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript

/-- If B has a left inverse Bdag and DB=B Dbar, then Bdag D B=Dbar.
This is the operator equality H=Dbar applied to an arbitrary x. -/
theorem quotient_discriminant_compression
    {Small Full : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (D : Full →ₗ[ℂ] Full) (Dbar : Small →ₗ[ℂ] Small)
    (B : Small →ₗ[ℂ] Full) (Bdag : Full →ₗ[ℂ] Small)
    (hB : ∀ x : Small, Bdag (B x) = x)
    (hD : ∀ x : Small, D (B x) = B (Dbar x))
    (x : Small) :
    Bdag (D (B x)) = Dbar x := by
  rw [hD x, hB (Dbar x)]

end QPPR.Manuscript
