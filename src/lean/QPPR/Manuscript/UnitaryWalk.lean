import QPPR.Manuscript.PhysicalWalk
import Mathlib.LinearAlgebra.Matrix.Permutation
import Mathlib.LinearAlgebra.UnitaryGroup
import Mathlib.Tactic.Abel

namespace QPPR.Manuscript.UnitaryWalk
open QPPR.Manuscript.ConcreteSymmetry QPPR.Manuscript.PhysicalWalk
open scoped Matrix
variable {N : Type*} [Fintype N] [DecidableEq N]

noncomputable def A (G : N → N → ℝ) : Matrix (N × N) N ℂ :=
  fun p j => if p.1 = j then amplitude G p.1 p.2 else 0

theorem A_star_A (G : N → N → ℝ) (hG : Stochastic G) :
    (A G)ᴴ * A G = 1 := by
  have hs (j k : N) : (starRingEnd ℂ) (amplitude G j k) = amplitude G j k :=
    star_amplitude G j k
  ext i j
  by_cases h : i = j
  · subst j
    simp [Matrix.mul_apply, A, Fintype.sum_prod_type, hs,
      amplitude_sum_sq G hG]
  · simp [Matrix.mul_apply, A, Fintype.sum_prod_type, hs, h,
      Matrix.one_apply, Ne.symm h]

noncomputable def P (G : N → N → ℝ) : Matrix (N × N) (N × N) ℂ := A G * (A G)ᴴ
noncomputable def R (G : N → N → ℝ) : Matrix (N × N) (N × N) ℂ := P G + P G - 1
noncomputable def S : Matrix (N × N) (N × N) ℂ :=
  (show Equiv.Perm (N × N) from Equiv.prodComm N N).permMatrix ℂ
noncomputable def W (G : N → N → ℝ) : Matrix (N × N) (N × N) ℂ := S * R G
noncomputable def U (G : N → N → ℝ) : Matrix (N × N) (N × N) ℂ := W G * W G

theorem P_star (G : N → N → ℝ) : (P G)ᴴ = P G := by simp [P]
theorem P_idem (G : N → N → ℝ) (hG : Stochastic G) : P G * P G = P G := by
  simp only [P]
  calc
    _ = A G * ((A G)ᴴ * A G) * (A G)ᴴ := by simp [Matrix.mul_assoc]
    _ = _ := by rw [A_star_A G hG, Matrix.mul_one]

theorem R_unitary (G : N → N → ℝ) (hG : Stochastic G) : R G ∈ unitary _ := by
  have hstar : star (R G) = R G := by simp [R, Matrix.star_eq_conjTranspose, P_star]
  have hsq : R G * R G = 1 := by
    unfold R
    simp only [sub_mul, mul_sub, add_mul, mul_add, Matrix.one_mul, Matrix.mul_one,
      P_idem G hG]
    abel
  exact ⟨by rw [hstar, hsq], by rw [hstar, hsq]⟩

theorem S_unitary : (S : Matrix (N × N) (N × N) ℂ) ∈ unitary _ := by
  constructor <;> simp [S, Matrix.star_eq_conjTranspose,
    Matrix.conjTranspose_permMatrix, ← Matrix.permMatrix_mul]

theorem U_unitary (G : N → N → ℝ) (hG : Stochastic G) : U G ∈ unitary _ := by
  have hW : W G ∈ unitary _ := (unitary _).mul_mem S_unitary (R_unitary G hG)
  exact (unitary _).mul_mem hW hW

theorem A_adjoint_mulVec (G : N → N → ℝ) (psi : State N) (j : N) :
    ((A G)ᴴ *ᵥ Function.uncurry psi) j =
      ∑ k, star (amplitude G j k) * psi j k := by
  simp [Matrix.mulVec, dotProduct, A, Fintype.sum_prod_type, apply_ite]

theorem P_mulVec (G : N → N → ℝ) (psi : State N) :
    P G *ᵥ Function.uncurry psi = Function.uncurry (project G psi) := by
  rw [P, ← Matrix.mulVec_mulVec]
  funext p
  change (∑ j, A G p j * ((A G)ᴴ *ᵥ Function.uncurry psi) j) = _
  simp only [A_adjoint_mulVec]
  simp [A, project, Function.uncurry]

theorem W_mulVec (G : N → N → ℝ) (psi : State N) :
    W G *ᵥ Function.uncurry psi = Function.uncurry (oneWalk G psi) := by
  rw [W, ← Matrix.mulVec_mulVec, R, Matrix.sub_mulVec, Matrix.add_mulVec,
    P_mulVec, Matrix.one_mulVec, S, Matrix.permMatrix_mulVec]
  funext p
  change project G psi p.2 p.1 + project G psi p.2 p.1 - psi p.2 p.1 =
    2 * project G psi p.2 p.1 - psi p.2 p.1
  ring

theorem U_mulVec (G : N → N → ℝ) (psi : State N) :
    U G *ᵥ Function.uncurry psi = Function.uncurry (recordedWalk G psi) := by
  rw [U, ← Matrix.mulVec_mulVec, W_mulVec, W_mulVec]
  rfl

end QPPR.Manuscript.UnitaryWalk
