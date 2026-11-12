import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC1

/-- (EC.5) Source-register marginal, for complex coefficient vectors. -/
def sourceMarginal
    {N : Type*} [Fintype N]
    (G : N → N → ℝ)
    (D : (N → ℂ) →ₗ[ℂ] (N → ℂ))
    (y z : N → ℂ) (i : N) : ℝ :=
  Complex.normSq (y i) +
    (∑ j : N, G i j * Complex.normSq (z j)) +
      2 * ((star (y i)) * (D z i)).re

/-- (EC.6) Destination-register marginal. -/
def destinationMarginal
    {N : Type*} [Fintype N]
    (G : N → N → ℝ)
    (D : (N → ℂ) →ₗ[ℂ] (N → ℂ))
    (y z : N → ℂ) (i : N) : ℝ :=
  (∑ j : N, G i j * Complex.normSq (y j)) +
    Complex.normSq (z i) +
      2 * ((star (z i)) * (D y i)).re

/-- At initialization z=0, the source marginal is |y_i|². -/
theorem sourceMarginal_initial
    {N : Type*} [Fintype N]
    (G : N → N → ℝ)
    (D : (N → ℂ) →ₗ[ℂ] (N → ℂ))
    (y : N → ℂ) (i : N) :
    sourceMarginal G D y 0 i = Complex.normSq (y i) := by
  simp [sourceMarginal]

/-- At initialization z=0, the destination marginal is G |y|². -/
theorem destinationMarginal_initial
    {N : Type*} [Fintype N]
    (G : N → N → ℝ)
    (D : (N → ℂ) →ₗ[ℂ] (N → ℂ))
    (y : N → ℂ) (i : N) :
    destinationMarginal G D y 0 i =
      ∑ j : N, G i j * Complex.normSq (y j) := by
  simp [destinationMarginal]

end QPPR.EC1
