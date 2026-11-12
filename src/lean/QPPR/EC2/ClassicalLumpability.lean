import QPPR.Core.Evolution

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC2

/-- One-step classical lumpability extends to arbitrarily many Markov steps. -/
theorem classical_lumpability_all_times
    {Full Reduced : Type*}
    [AddCommGroup Full] [Module ℝ Full]
    [AddCommGroup Reduced] [Module ℝ Reduced]
    (G : Full →ₗ[ℝ] Full) (Gbar : Reduced →ₗ[ℝ] Reduced)
    (C : Full →ₗ[ℝ] Reduced)
    (hCG : ∀ p, C (G p) = Gbar (C p))
    (p : Full) (t : ℕ) :
    C (QPPR.Core.evolve G t p) =
      QPPR.Core.evolve Gbar t (C p) := by
  induction t with
  | zero => rfl
  | succ t ih =>
      calc
        C (QPPR.Core.evolve G (t + 1) p) =
            C (G (QPPR.Core.evolve G t p)) := rfl
        _ = Gbar (C (QPPR.Core.evolve G t p)) := hCG _
        _ = Gbar (QPPR.Core.evolve Gbar t (C p)) := by rw [ih]
        _ = QPPR.Core.evolve Gbar (t + 1) (C p) := rfl

end QPPR.EC2
