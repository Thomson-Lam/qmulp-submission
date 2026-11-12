import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC1

/-- Reflection–swap update on one eigenspace of the discriminant. -/
def coefficientStep (d : ℂ) (pair : ℂ × ℂ) : ℂ × ℂ :=
  (-pair.2, pair.1 + 2 * d * pair.2)

/-- The recorded step U = W² (Equation following EC.4), on one eigenmode. -/
theorem coefficientStep_twice (d y z : ℂ) :
    coefficientStep d (coefficientStep d (y, z)) =
      (-y - 2 * d * z, 2 * d * y + (4 * d ^ 2 - 1) * z) := by
  apply Prod.ext
  · change -(y + 2 * d * z) = -y - 2 * d * z
    ring
  · change -z + 2 * d * (y + 2 * d * z) =
        2 * d * y + (4 * d ^ 2 - 1) * z
    ring

/-- The usual characteristic equation λ² - 2dλ + 1 = 0 produces
    an eigenvector (1, -λ) for the single W step. -/
theorem coefficientStep_eigenvector (d lam : ℂ)
    (hlam : lam ^ 2 - 2 * d * lam + 1 = 0) :
    coefficientStep d (1, -lam) = (lam, -(lam ^ 2)) := by
  apply Prod.ext
  · change -(-lam) = lam
    ring
  · change (1 : ℂ) + 2 * d * (-lam) = -(lam ^ 2)
    calc
      (1 : ℂ) + 2 * d * (-lam) =
          -(lam ^ 2) + (lam ^ 2 - 2 * d * lam + 1) := by ring
      _ = -(lam ^ 2) := by rw [hlam]; ring

end QPPR.EC1
