import QPPR.Manuscript.PhysicalWalk
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.LinearAlgebra.Projection
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Algebra.BigOperators.Field

namespace QPPR.Manuscript.ClassicalLimit
open QPPR.Core QPPR.Manuscript.ConcreteSymmetry QPPR.Manuscript.PhysicalWalk
variable {N : Type*} [Fintype N]

noncomputable def T (G : N → N → ℝ) : (N → ℝ) →ₗ[ℝ] (N → ℝ) where
  toFun := classicalStep G
  map_add' x y := by funext i; simp [classicalStep, mul_add, Finset.sum_add_distrib]
  map_smul' c x := by funext i; simp [classicalStep, Finset.mul_sum, mul_left_comm, mul_assoc]

noncomputable def l1 (x : N → ℝ) : ℝ := ∑ j, |x j|

theorem abs_le_l1 (x : N → ℝ) (j : N) : |x j| ≤ l1 x :=
  Finset.single_le_sum (f := fun i => |x i|) (fun i _ => abs_nonneg _) (Finset.mem_univ j)

theorem l1_step_le (G : N → N → ℝ) (hG : Stochastic G) (x : N → ℝ) :
    l1 (T G x) ≤ l1 x := by
  change (∑ i, |∑ j, G i j * x j|) ≤ ∑ j, |x j|
  calc
    _ ≤ ∑ i, ∑ j, G i j * |x j| := by
      apply Finset.sum_le_sum
      intro i _
      calc
        _ ≤ ∑ j, |G i j * x j| := by
          simpa only [Real.norm_eq_abs] using norm_sum_le Finset.univ (fun j => G i j * x j)
        _ = _ := by simp [abs_mul, abs_of_nonneg (hG.1 i _)]
    _ = _ := by rw [Finset.sum_comm]; simp [← Finset.sum_mul, hG.2]

theorem l1_evolve_le (G : N → N → ℝ) (hG : Stochastic G) (x : N → ℝ) (n : ℕ) :
    l1 (evolve (T G) n x) ≤ l1 x := by
  induction n with
  | zero => exact le_rfl
  | succ n ih => exact (l1_step_le G hG _).trans ih

theorem fixed_range_disjoint (G : N → N → ℝ) (hG : Stochastic G) :
    Disjoint (LinearMap.ker (T G - 1)) (LinearMap.range (T G - 1)) := by
  apply Submodule.disjoint_def.mpr
  intro x hx hr
  obtain ⟨y,hy⟩ := hr
  have hfix : T G x = x := sub_eq_zero.mp hx
  have hy' : T G y = y + x := by
    have h : T G y - y = x := hy
    exact sub_eq_iff_eq_add'.mp h
  have hiter (n : ℕ) : evolve (T G) n y = y + (n : ℝ) • x := by
    induction n with
    | zero => simp [evolve]
    | succ n ih =>
      change T G (evolve (T G) n y) = _
      rw [ih, map_add, map_smul, hfix, hy']
      simp [Nat.cast_add, add_smul]
      abel
  have hbound (n : ℕ) (j : N) : (n : ℝ) * |x j| ≤ 2 * l1 y := by
    have he : (n : ℝ) * x j = evolve (T G) n y j - y j := by
      rw [hiter]
      simp
    calc
      _ = |(n : ℝ) * x j| := by simp [abs_mul]
      _ = |evolve (T G) n y j - y j| := by rw [he]
      _ ≤ |evolve (T G) n y j| + |y j| := abs_sub _ _
      _ ≤ l1 y + l1 y := add_le_add ((abs_le_l1 _ j).trans (l1_evolve_le G hG y n)) (abs_le_l1 y j)
      _ = _ := by ring
  funext j
  by_contra hxj
  have ha : 0 < |x j| := abs_pos.mpr hxj
  obtain ⟨n,hn⟩ := exists_nat_gt (2 * l1 y / |x j|)
  have hh : 2 * l1 y < (n : ℝ) * |x j| := (div_lt_iff₀ ha).mp hn
  exact (not_lt_of_ge (hbound n j)) hh

noncomputable def finiteScore (G : N → N → ℝ) (v : N → ℝ) (n : ℕ) (j : N) : ℝ :=
  (∑ t ∈ Finset.range n, evolve (classicalStep G) t v j) / (n : ℝ)

/-- Every finite column-stochastic classical trajectory has a Cesaro limit.
    No irreducibility, aperiodicity or unique stationary distribution is assumed. -/
theorem classical_cesaro_exists (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ) :
    ∃ p : N → ℝ, ∀ j, Filter.Tendsto (fun n => finiteScore G v n j) Filter.atTop (nhds ((p : N → ℝ) j)) := by
  let L := T G - 1
  have hcompl : IsCompl L.ker L.range := by
    apply (Submodule.isCompl_iff_disjoint _ _ ?_).mpr (fixed_range_disjoint G hG)
    rw [add_comm, LinearMap.finrank_range_add_finrank_ker]
  obtain ⟨p,r,hpr,_⟩ := Submodule.existsUnique_add_of_isCompl hcompl v
  obtain ⟨y,hy⟩ := r.property
  have hfix : T G (p : N → ℝ) = p := sub_eq_zero.mp p.property
  have hinit : v = (p : N → ℝ) + (T G y - y) := by
    rw [← hpr, ← hy]
    rfl
  have hs (n : ℕ) :
      (∑ t ∈ Finset.range n, evolve (T G) t v) =
        (n : ℝ) • (p : N → ℝ) + (evolve (T G) n y - y) := by
    induction n with
    | zero => simp [evolve]
    | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      have ht (n : ℕ) : evolve (T G) n v = p + (evolve (T G) (n+1) y - evolve (T G) n y) := by
        induction n with
        | zero => simpa [evolve] using hinit
        | succ n ih =>
          change T G (evolve (T G) n v) = _
          rw [ih, map_add, map_sub, hfix]
          rfl
      rw [ht n]
      simp [Nat.cast_add, add_smul]
      abel
  refine ⟨p,fun j => ?_⟩
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  have hb : ∀ n : ℕ, 0 < n → |finiteScore G v n j - (p : N → ℝ) j| ≤ (2 * l1 y) / (n : ℝ) := by
    intro n hn
    have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (ne_of_gt hn)
    have he := congrFun (hs n) j
    simp only [Finset.sum_apply, Pi.add_apply, Pi.smul_apply, Pi.sub_apply, smul_eq_mul] at he
    change (∑ t ∈ Finset.range n, evolve (classicalStep G) t v j) =
      (n : ℝ) * (p : N → ℝ) j + (evolve (T G) n y j - y j) at he
    unfold finiteScore
    rw [he, add_div, mul_div_cancel_left₀ _ hn']
    simp only [add_sub_cancel_left, abs_div, abs_of_nonneg (show (0 : ℝ) ≤ (n : ℝ) from Nat.cast_nonneg n)]
    apply div_le_div_of_nonneg_right _ (show (0 : ℝ) ≤ (n : ℝ) from Nat.cast_nonneg n)
    exact (abs_sub _ _).trans (by
      have h1 := (abs_le_l1 (evolve (T G) n y) j).trans (l1_evolve_le G hG y n)
      have h2 := abs_le_l1 y j
      linarith)
  apply squeeze_zero' (Filter.Eventually.of_forall (fun n => norm_nonneg _))
    (Filter.eventually_atTop.mpr ⟨1,fun n hn => ?_⟩)
    (tendsto_const_div_atTop_nhds_zero_nat (2 * l1 y))
  simpa only [Real.norm_eq_abs] using hb n (by omega)

end QPPR.Manuscript.ClassicalLimit
