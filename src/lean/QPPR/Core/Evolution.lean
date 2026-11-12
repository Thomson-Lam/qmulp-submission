import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Core

/-- Apply a transition `step` repeatedly. -/
def evolve {α : Type*} (step : α → α) : ℕ → α → α
  | 0, x => x
  | n + 1, x => step (evolve step n x)

@[simp] theorem evolve_zero {α : Type*} (step : α → α) (x : α) :
    evolve step 0 x = x := rfl

@[simp] theorem evolve_succ {α : Type*} (step : α → α) (n : ℕ) (x : α) :
    evolve step (n + 1) x = step (evolve step n x) := rfl

/-- Finite-time average of a real-valued observable.
    For τ = 0 this definition is zero (by the division convention on ℝ).
    Mathematical probability averages in the paper use τ > 0. -/
noncomputable def finiteAverage {α ι : Type*} (step : α → α) (measure : α → ι → ℝ)
    (x : α) (τ : ℕ) (i : ι) : ℝ :=
  (∑ t ∈ Finset.range τ, measure (evolve step t x) i) / (τ : ℝ)

end QPPR.Core
