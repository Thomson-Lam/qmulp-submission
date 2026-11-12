import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript

noncomputable def reciprocalDiscriminant
    {N : Type*} (G : N → N → ℝ) (i j : N) : ℝ :=
  Real.sqrt (G i j * G j i)

/-- The discriminant kernel is symmetric for every real transition kernel. -/
theorem reciprocalDiscriminant_symmetric
    {N : Type*} (G : N → N → ℝ) (i j : N) :
    reciprocalDiscriminant G i j = reciprocalDiscriminant G j i := by
  unfold reciprocalDiscriminant
  rw [mul_comm]

/-- Any node relabeling preserving the full G matrix also preserves D. -/
theorem reciprocalDiscriminant_permutation
    {N : Type*} (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ i j, G (q i) (q j) = G i j) (i j : N) :
    reciprocalDiscriminant G (q i) (q j) = reciprocalDiscriminant G i j := by
  unfold reciprocalDiscriminant
  rw [hG i j, hG j i]

end QPPR.Manuscript
