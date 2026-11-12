import QPPR.Core.Evolution

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC3

open QPPR.Core

/-- A commuting symmetry fixing x fixes the full trajectory. -/
theorem symmetry_preserved_under_evolution
    {State : Type*} (step symmetry : State → State)
    (hcommute : ∀ s, symmetry (step s) = step (symmetry s))
    (x : State) (hinitial : symmetry x = x) (t : ℕ) :
    symmetry (evolve step t x) = evolve step t x := by
  induction t with
  | zero => simpa using hinitial
  | succ t ih =>
      calc
        symmetry (evolve step (t + 1) x) =
            symmetry (step (evolve step t x)) := rfl
        _ = step (symmetry (evolve step t x)) := hcommute _
        _ = step (evolve step t x) := by rw [ih]
        _ = evolve step (t + 1) x := rfl

/-- Under equivariant measurement, symmetric labels have identical
    probabilities at every recorded time. -/
theorem symmetric_measurements_equal
    {State ι : Type*} (step symmetry : State → State)
    (swapLabels : ι → ι) (measure : State → ι → ℝ)
    (hcommute : ∀ s, symmetry (step s) = step (symmetry s))
    (hmeasure : ∀ s i, measure (symmetry s) i = measure s (swapLabels i))
    (x : State) (hinitial : symmetry x = x) (t : ℕ) (i : ι) :
    measure (evolve step t x) i = measure (evolve step t x) (swapLabels i) := by
  calc
    measure (evolve step t x) i =
        measure (symmetry (evolve step t x)) i := by
          rw [symmetry_preserved_under_evolution step symmetry hcommute x hinitial t]
    _ = measure (evolve step t x) (swapLabels i) := hmeasure _ _

/-- Equality for all t transfers to the finite time average of the EC. -/
theorem symmetric_finite_averages_equal
    {State ι : Type*} (step symmetry : State → State)
    (swapLabels : ι → ι) (measure : State → ι → ℝ)
    (hcommute : ∀ s, symmetry (step s) = step (symmetry s))
    (hmeasure : ∀ s i, measure (symmetry s) i = measure s (swapLabels i))
    (x : State) (hinitial : symmetry x = x) (τ : ℕ) (i : ι) :
    finiteAverage step measure x τ i =
      finiteAverage step measure x τ (swapLabels i) := by
  have hsum :
      (∑ t ∈ Finset.range τ, measure (evolve step t x) i) =
        (∑ t ∈ Finset.range τ, measure (evolve step t x) (swapLabels i)) := by
    apply Finset.sum_congr rfl
    intro t ht
    exact symmetric_measurements_equal step symmetry swapLabels measure
      hcommute hmeasure x hinitial t i
  unfold finiteAverage
  rw [hsum]

end QPPR.EC3
