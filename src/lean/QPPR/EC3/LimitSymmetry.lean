import QPPR.Core.RealSequenceLimits
import QPPR.EC3.DiscriminantSymmetry
import QPPR.EC3.ClassicalSymmetry

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC3

open QPPR.Core

/-- Equal score trajectories at all finite horizons have equal limiting
    scores whenever the respective limits exist. -/
theorem symmetry_equal_limiting_scores
    (scoreI scoreJ : ℕ → ℝ) (ωI ωJ : ℝ)
    (hfinite : ∀ τ, scoreI τ = scoreJ τ)
    (hI : Filter.Tendsto scoreI Filter.atTop (nhds ωI))
    (hJ : Filter.Tendsto scoreJ Filter.atTop (nhds ωJ)) :
    ωI = ωJ := by
  exact equal_sequences_have_equal_limits scoreI scoreJ ωI ωJ hfinite hI hJ

end QPPR.EC3
