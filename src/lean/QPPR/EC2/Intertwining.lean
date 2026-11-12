import QPPR.Core.Evolution

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC2

open QPPR.Core

/-- If one step intertwines under B, all iterates intertwine. -/
theorem intertwining_iterates
    {Small Full : Type*} (smallStep : Small → Small)
    (fullStep : Full → Full) (B : Small → Full)
    (hintertwine : ∀ x, B (smallStep x) = fullStep (B x))
    (t : ℕ) (x : Small) :
    B (evolve smallStep t x) = evolve fullStep t (B x) := by
  induction t with
  | zero => rfl
  | succ t ih =>
      calc
        B (evolve smallStep (t + 1) x) =
            B (smallStep (evolve smallStep t x)) := rfl
        _ = fullStep (B (evolve smallStep t x)) := hintertwine _
        _ = fullStep (evolve fullStep t (B x)) := by rw [ih]
        _ = evolve fullStep (t + 1) (B x) := rfl

/-- If physical measurements agree after embedding, their averages agree
    at every horizon (including the paper's horizons τ > 0). -/
theorem finite_averages_agree
    {Small Full ι : Type*} (smallStep : Small → Small)
    (fullStep : Full → Full) (B : Small → Full)
    (smallMeasure : Small → ι → ℝ) (fullMeasure : Full → ι → ℝ)
    (hintertwine : ∀ x, B (smallStep x) = fullStep (B x))
    (hmeasure : ∀ x i, fullMeasure (B x) i = smallMeasure x i)
    (τ : ℕ) (x : Small) (i : ι) :
    finiteAverage fullStep fullMeasure (B x) τ i =
      finiteAverage smallStep smallMeasure x τ i := by
  have hsum :
      (∑ t ∈ Finset.range τ, fullMeasure (evolve fullStep t (B x)) i) =
        (∑ t ∈ Finset.range τ, smallMeasure (evolve smallStep t x) i) := by
    apply Finset.sum_congr rfl
    intro t ht
    calc
      fullMeasure (evolve fullStep t (B x)) i =
          fullMeasure (B (evolve smallStep t x)) i := by
            rw [intertwining_iterates smallStep fullStep B hintertwine t x]
      _ = smallMeasure (evolve smallStep t x) i := hmeasure _ _
  unfold finiteAverage
  rw [hsum]

/-- An intertwining identity for W also holds for recorded steps U = W². -/
theorem intertwining_recorded_steps
    {Small Full : Type*} (smallW : Small → Small) (fullW : Full → Full)
    (B : Small → Full)
    (hW : ∀ x, B (smallW x) = fullW (B x))
    (x : Small) :
    B (smallW (smallW x)) = fullW (fullW (B x)) := by
  calc
    B (smallW (smallW x)) = fullW (B (smallW x)) := hW _
    _ = fullW (fullW (B x)) := by rw [hW]

end QPPR.EC2
