import QPPR.Manuscript.GeometricBound
import QPPR.EC1.GeometricAverageExtra
import QPPR.Manuscript.ProofTargets
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.Abel

namespace QPPR.Manuscript.MeasuredSpectrum
open QPPR.EC1
variable {I : Type*} [Fintype I]

noncomputable def average (f : ℕ → ℂ) (n : ℕ) : ℂ :=
  (∑ t ∈ Finset.range n, f t) / (n : ℂ)

theorem average_sum (f : I → ℕ → ℂ) (n : ℕ) :
    average (fun t => ∑ i, f i t) n = ∑ i, average (f i) n := by
  unfold average
  rw [Finset.sum_comm, Finset.sum_div]

theorem average_mul_const (f : ℕ → ℂ) (c : ℂ) (n : ℕ) :
    average (fun t => f t * c) n = average f n * c := by
  simp [average, ← Finset.sum_mul, div_mul_eq_mul_div]

theorem geometricSum_eq (z : ℂ) (n : ℕ) :
    geometricSum z n = ∑ t ∈ Finset.range n, z ^ t := by
  induction n with
  | zero => simp [geometricSum]
  | succ n ih => simp [geometricSum, ih, Finset.sum_range_succ]

noncomputable def phaseAverage (z : ℂ) (n : ℕ) : ℂ := average (fun t => z ^ t) n

theorem phaseAverage_one (n : ℕ) (hn : 0 < n) : phaseAverage 1 n = 1 := by
  simp [phaseAverage, average, ne_of_gt hn]

theorem phaseAverage_bound (z : ℂ) (hz : ‖z‖ = 1) (hne : z ≠ 1)
    (n : ℕ) (hn : 0 < n) :
    ‖phaseAverage z n‖ ≤ (2 / ‖1 - z‖) / (n : ℝ) := by
  have hgap : 0 < ‖(1 : ℂ) - z‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hne.symm)
  have hnum := QPPR.Manuscript.geometric_interference_bound z n hz
  have hs : ‖geometricSum z n‖ ≤ 2 / ‖(1 : ℂ) - z‖ :=
    (le_div_iff₀ hgap).mpr (by simpa [mul_comm] using hnum)
  unfold phaseAverage average
  rw [← geometricSum_eq, norm_div]
  have hnorm : ‖(n : ℂ)‖ = (n : ℝ) := by simp
  rw [hnorm]
  exact div_le_div_of_nonneg_right hs (Nat.cast_nonneg n)

theorem unit_mul_star_self (z : ℂ) (hz : ‖z‖ = 1) : z * star z = 1 := by
  rw [← starRingEnd_apply, Complex.mul_conj, Complex.normSq_eq_norm_sq, hz]
  norm_num

theorem phase_distinct (lam : I → ℂ) (hinj : Function.Injective lam)
    (hunit : ∀ i, ‖lam i‖ = 1) (i j : I) (hij : i ≠ j) :
    lam i * star (lam j) ≠ 1 := by
  intro h
  apply hij
  apply hinj
  have hj := unit_mul_star_self (lam j) (hunit j)
  calc
    lam i = lam i * (star (lam j) * lam j) := by rw [mul_comm (star _), hj, mul_one]
    _ = (lam i * star (lam j)) * lam j := by ring
    _ = lam j := by rw [h, one_mul]

theorem density_expansion (lam phi : I → ℂ) (t : ℕ) :
    (Complex.normSq (∑ i, lam i ^ t * phi i) : ℂ) =
      ∑ i, ∑ j, (lam i * star (lam j)) ^ t * (phi i * star (phi j)) := by
  rw [← Complex.mul_conj]
  simp only [map_sum, map_mul, map_pow, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_pow]
  simp only [starRingEnd_apply]
  ring

theorem average_density_expansion (lam phi : I → ℂ) (n : ℕ) :
    average (fun t => (Complex.normSq (∑ i, lam i ^ t * phi i) : ℂ)) n =
      ∑ i, ∑ j, phaseAverage (lam i * star (lam j)) n * (phi i * star (phi j)) := by
  simp_rw [density_expansion]
  rw [average_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [average_sum]
  apply Finset.sum_congr rfl
  intro j _
  exact average_mul_const _ _ _

noncomputable def remainder (lam : I → ℂ) (n : ℕ) (i j : I) : ℂ :=
  by classical exact if i = j then 0 else phaseAverage (lam i * star (lam j)) n

theorem average_error_expansion (lam phi : I → ℂ)
    (hunit : ∀ i, ‖lam i‖ = 1) (n : ℕ) (hn : 0 < n) :
    average (fun t => (Complex.normSq (∑ i, lam i ^ t * phi i) : ℂ)) n -
      (∑ i, (Complex.normSq (phi i) : ℂ)) =
        ∑ i, ∑ j, remainder lam n i j * (phi i * star (phi j)) := by
  classical
  rw [average_density_expansion]
  have h (i : I) :
      (∑ j, phaseAverage (lam i * star (lam j)) n * (phi i * star (phi j))) =
        (Complex.normSq (phi i) : ℂ) + ∑ j, remainder lam n i j * (phi i * star (phi j)) := by
    calc
      _ = ∑ j, ((if i = j then (Complex.normSq (phi i) : ℂ) else 0) +
        remainder lam n i j * (phi i * star (phi j))) := by
        apply Finset.sum_congr rfl
        intro j _
        by_cases hij : i = j
        · subst j
          rw [unit_mul_star_self _ (hunit i), phaseAverage_one n hn]
          simp [remainder, ← starRingEnd_apply, Complex.mul_conj]
        · simp [remainder, hij]
      _ = _ := by simp [Finset.sum_add_distrib]
  simp_rw [h]
  rw [Finset.sum_add_distrib]
  abel

noncomputable def crossConstant (lam : I → ℂ) (phi : I → ℂ) : ℝ :=
  by classical exact ∑ i, ∑ j, if i = j then 0 else
    (2 / ‖1 - lam i * star (lam j)‖) * (‖phi i‖ * ‖phi j‖)

theorem average_error_bound (lam phi : I → ℂ)
    (hinj : Function.Injective lam) (hunit : ∀ i, ‖lam i‖ = 1)
    (n : ℕ) (hn : 0 < n) :
    ‖average (fun t => (Complex.normSq (∑ i, lam i ^ t * phi i) : ℂ)) n -
      (∑ i, (Complex.normSq (phi i) : ℂ))‖ ≤ crossConstant lam phi / (n : ℝ) := by
  classical
  rw [average_error_expansion lam phi hunit n hn]
  calc
    _ ≤ ∑ i, ∑ j, ‖remainder lam n i j * (phi i * star (phi j))‖ := by
      exact (norm_sum_le _ _).trans (Finset.sum_le_sum (fun i _ => norm_sum_le _ _))
    _ ≤ ∑ i, ∑ j, (if i = j then 0 else
        (2 / ‖1 - lam i * star (lam j)‖) * (‖phi i‖ * ‖phi j‖)) / (n : ℝ) := by
      apply Finset.sum_le_sum
      intro i _
      apply Finset.sum_le_sum
      intro j _
      by_cases hij : i = j
      · simp [remainder, hij]
      · have hu : ‖lam i * star (lam j)‖ = 1 := by simp [norm_mul, hunit]
        have hb := phaseAverage_bound _ hu (phase_distinct lam hinj hunit i j hij) n hn
        simp only [remainder, if_neg hij, norm_mul, norm_star]
        calc
          _ ≤ ((2 / ‖1 - lam i * star (lam j)‖) / (n : ℝ)) * (‖phi i‖ * ‖phi j‖) :=
            mul_le_mul_of_nonneg_right hb (mul_nonneg (norm_nonneg _) (norm_nonneg _))
          _ = _ := by ring
    _ = _ := by simp [crossConstant, Finset.sum_div]

end QPPR.Manuscript.MeasuredSpectrum
