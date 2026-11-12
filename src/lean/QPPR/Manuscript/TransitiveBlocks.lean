import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/
namespace QPPR.Manuscript

/-- Two members in the same orbit of score-preserving exchanges have equal
probabilities.  This is the first step toward justifying the block-size
unfolding used in Proposition 2. -/
theorem equal_scores_of_transitive_exchanges
    {N : Type*} (p : N → ℝ) (block : Finset N)
    (htrans : ∀ i ∈ block, ∀ j ∈ block,
      ∃ q : Equiv.Perm N, q i = j ∧ ∀ k, p (q k) = p k)
    (i j : N) (hi : i ∈ block) (hj : j ∈ block) :
    p i = p j := by
  obtain ⟨q, hq, hp⟩ := htrans i hi j hj
  calc
    p i = p (q i) := (hp i).symm
    _ = p j := by rw [hq]

/-- On a nonempty transitive block, every member has exactly the total
block score divided by the number of members. This is the precise last
sentence of manuscript Proposition 2, with transitivity made explicit. -/
theorem member_score_eq_block_average
    {N : Type*} (p : N → ℝ) (block : Finset N)
    (htrans : ∀ i ∈ block, ∀ j ∈ block,
      ∃ q : Equiv.Perm N, q i = j ∧ ∀ k, p (q k) = p k)
    (i : N) (hi : i ∈ block) :
    p i = (∑ j ∈ block, p j) / (block.card : ℝ) := by
  classical
  have hcard : (block.card : ℝ) ≠ 0 := by
    exact_mod_cast (Finset.card_ne_zero.mpr ⟨i, hi⟩)
  have hsum : (∑ j ∈ block, p j) = (block.card : ℝ) * p i := by
    calc
      (∑ j ∈ block, p j) = (∑ j ∈ block, p i) := by
        apply Finset.sum_congr rfl
        intro j hj
        exact (equal_scores_of_transitive_exchanges p block htrans j i hj hi)
      _ = (block.card : ℝ) * p i := by simp
  apply (eq_div_iff hcard).2
  calc
    p i * (block.card : ℝ) = (block.card : ℝ) * p i := mul_comm _ _
    _ = (∑ j ∈ block, p j) := hsum.symm

end QPPR.Manuscript
