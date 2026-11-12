import QPPR.EC1.SpectralEvolution
import QPPR.EC1.RankStability
import QPPR.Manuscript.DiscriminantKernel

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript

/-- Spectral component propagation: rigorous EC.1.1 sublemma. -/
theorem proposition1_eigencomponent_expansion
    {V I : Type*} [AddCommGroup V] [Module ℂ V] [Fintype I]
    (U : V →ₗ[ℂ] V) (eig : I → ℂ) (phi : I → V)
    (heig : ∀ i, U (phi i) = eig i • phi i) (t : ℕ) :
    QPPR.Core.evolve U t (∑ i, phi i) = ∑ i, (eig i ^ t) • phi i := by
  exact QPPR.EC1.spectral_evolve U eig phi heig t

/-- The published rank-gap observation, independent of the missing L1 bound. -/
theorem proposition1_rank_gap_sublemma
    {N : Type*} (limit approx : N → ℝ) (eps : ℝ)
    (huniform : ∀ i, |approx i - limit i| ≤ eps)
    (i j : N) (hgap : 2 * eps < limit i - limit j) :
    approx j < approx i := by
  exact QPPR.EC1.rank_stability_of_uniform_error limit approx eps huniform i j hgap

end QPPR.Manuscript
