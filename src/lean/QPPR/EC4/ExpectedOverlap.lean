import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC4

/-- Expected overlap ratio of two independently tie-broken rankings with
    identical selection probabilities. -/
noncomputable def selfOverlap {ι : Type*} [Fintype ι] (r : ι → ℝ) (k : ℝ) : ℝ :=
  (∑ i : ι, r i * r i) / k

/-- Self-overlap never exceeds 1 when the inclusion probabilities are
    in [0,1] and sum to the positive review budget k. -/
theorem selfOverlap_le_one
    {ι : Type*} [Fintype ι] (r : ι → ℝ) (k : ℝ)
    (hzero : ∀ i, 0 ≤ r i) (hone : ∀ i, r i ≤ 1)
    (hk : 0 < k) (hsum : (∑ i : ι, r i) = k) :
    selfOverlap r k ≤ 1 := by
  have hsq (i : ι) : r i * r i ≤ r i := by
    have h := mul_nonneg (hzero i) (sub_nonneg.mpr (hone i))
    nlinarith
  have hsum_le : (∑ i : ι, r i * r i) ≤ (∑ i : ι, r i) := by
    apply Finset.sum_le_sum
    intro i hi
    exact hsq i
  have hbound : (∑ i : ι, r i * r i) ≤ k := by
    calc
      (∑ i : ι, r i * r i) ≤ (∑ i : ι, r i) := hsum_le
      _ = k := hsum
  unfold selfOverlap
  apply (div_le_iff₀ hk).2
  simpa using hbound

/-- Two people tied for one slot: independent choices overlap in expectation
    with ratio 1/2, not 1. -/
example :
    (((1 / 2 : ℝ) * (1 / 2) + (1 / 2) * (1 / 2)) / 1) = 1 / 2 := by
  norm_num

end QPPR.EC4
