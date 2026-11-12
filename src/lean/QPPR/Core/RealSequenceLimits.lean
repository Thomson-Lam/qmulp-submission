import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Core

/-- Two real-valued sequences which agree term-by-term cannot converge to
    different limits. -/
theorem equal_sequences_have_equal_limits
    (a b : ℕ → ℝ) (L M : ℝ)
    (heq : ∀ t, a t = b t)
    (ha : Filter.Tendsto a Filter.atTop (nhds L))
    (hb : Filter.Tendsto b Filter.atTop (nhds M)) :
    L = M := by
  have hfunctions : a = b := funext heq
  subst b
  exact tendsto_nhds_unique ha hb

end QPPR.Core
