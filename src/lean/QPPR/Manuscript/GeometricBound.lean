import QPPR.EC1.GeometricAverage

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/
namespace QPPR.Manuscript

/-- For |z|=1, the magnitude of the geometric numerator is ≤ 2. -/
theorem geometric_numerator_le_two (z : ℂ) (n : ℕ) (hz : ‖z‖ = 1) :
    ‖(1 : ℂ) - z ^ n‖ ≤ 2 := by
  calc
    ‖(1 : ℂ) - z ^ n‖ ≤ ‖(1 : ℂ)‖ + ‖z ^ n‖ := norm_sub_le _ _
    _ = 2 := by
      rw [norm_one, norm_pow, hz]
      norm_num

/-- The core cross-term inequality before dividing by n and the phase gap.
    This proves `‖1-z‖ ‖∑_{t<n} z^t‖ ≤ 2` for every n. -/
theorem geometric_interference_bound (z : ℂ) (n : ℕ) (hz : ‖z‖ = 1) :
    ‖(1 : ℂ) - z‖ * ‖QPPR.EC1.geometricSum z n‖ ≤ 2 := by
  calc
    ‖(1 : ℂ) - z‖ * ‖QPPR.EC1.geometricSum z n‖ =
        ‖((1 : ℂ) - z) * QPPR.EC1.geometricSum z n‖ := (norm_mul _ _).symm
    _ = ‖(1 : ℂ) - z ^ n‖ := by rw [QPPR.EC1.geometricSum_identity]
    _ ≤ 2 := geometric_numerator_le_two z n hz

end QPPR.Manuscript
