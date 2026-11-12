import QPPR.Core.Evolution

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC1

open QPPR.Core

/-- If a component is an eigenvector of a linear operator, its trajectory
    is multiplication by the nth power of its eigenvalue. -/
theorem eigencomponent_evolve
    {V : Type*} [AddCommGroup V] [Module ℂ V]
    (U : V →ₗ[ℂ] V) (φ : V) (eig : ℂ)
    (heigen : U φ = eig • φ) (n : ℕ) :
    evolve U n φ = (eig ^ n) • φ := by
  induction n with
  | zero => simp [evolve]
  | succ n ih =>
      calc
        evolve U (n + 1) φ = U (evolve U n φ) := rfl
        _ = U ((eig ^ n) • φ) := by rw [ih]
        _ = (eig ^ n) • (U φ) := by rw [map_smul]
        _ = (eig ^ n) • (eig • φ) := by rw [heigen]
        _ = (eig ^ (n + 1)) • φ := by rw [pow_succ, smul_smul]

/-- One-step evolution distributes over a finite spectral decomposition. -/
theorem spectral_step
    {V I : Type*} [AddCommGroup V] [Module ℂ V] [Fintype I]
    (U : V →ₗ[ℂ] V) (eig : I → ℂ) (φ : I → V)
    (heigen : ∀ i, U (φ i) = eig i • φ i) :
    U (∑ i, φ i) = ∑ i, eig i • φ i := by
  calc
    U (∑ i, φ i) = ∑ i, U (φ i) := map_sum U _ _
    _ = ∑ i, eig i • φ i := by
      apply Finset.sum_congr rfl
      intro i hi
      exact heigen i

/-- Complete time-n evolution of a finite eigencomponent decomposition. -/
theorem spectral_evolve
    {V I : Type*} [AddCommGroup V] [Module ℂ V] [Fintype I]
    (U : V →ₗ[ℂ] V) (eig : I → ℂ) (φ : I → V)
    (heigen : ∀ i, U (φ i) = eig i • φ i)
    (n : ℕ) :
    evolve U n (∑ i, φ i) = ∑ i, (eig i ^ n) • φ i := by
  induction n with
  | zero => simp [evolve]
  | succ n ih =>
      calc
        evolve U (n + 1) (∑ i, φ i) =
            U (evolve U n (∑ i, φ i)) := rfl
        _ = U (∑ i, (eig i ^ n) • φ i) := by rw [ih]
        _ = ∑ i, U ((eig i ^ n) • φ i) := map_sum U _ _
        _ = ∑ i, (eig i ^ (n + 1)) • φ i := by
          apply Finset.sum_congr rfl
          intro i hi
          rw [map_smul, heigen i, smul_smul, pow_succ]

end QPPR.EC1
