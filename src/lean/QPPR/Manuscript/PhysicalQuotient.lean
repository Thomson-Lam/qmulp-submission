import QPPR.Manuscript.PhysicalWalk

namespace QPPR.Manuscript.PhysicalQuotient
open QPPR.Core QPPR.Manuscript.PhysicalWalk QPPR.Manuscript.ConcreteSymmetry
variable {N R : Type*} [Fintype N] [Fintype R] [DecidableEq R]

def fiber (c : N → R) (a : R) : Finset N := Finset.univ.filter (fun j => c j = a)
noncomputable def aggregate (c : N → R) (p : N → ℝ) (a : R) : ℝ :=
  ∑ j ∈ fiber c a, p j

noncomputable def embedding (c : N → R) (w : R → ℝ) : (R → ℂ) →ₗ[ℂ] (N → ℂ) where
  toFun y j := (w (c j) : ℂ) * y (c j)
  map_add' y z := by funext j; simp [mul_add]
  map_smul' d y := by funext j; simp [mul_left_comm]

def Normalized (c : N → R) (w : R → ℝ) : Prop :=
  ∀ a, (fiber c a).card * (w a * w a) = 1

theorem aggregate_uniform (c : N → R) (w : R → ℝ) (hw : Normalized c w)
    (f : R → ℝ) (a : R) :
    aggregate c (fun j => w (c j) * w (c j) * f (c j)) a = f a := by
  unfold aggregate
  calc
    _ = ∑ j ∈ fiber c a, w a * w a * f a := by
      apply Finset.sum_congr rfl
      intro j hj
      simp only [(Finset.mem_filter.mp hj).2]
    _ = ((fiber c a).card : ℝ) * (w a * w a) * f a := by
      simp [mul_assoc]
    _ = f a := by rw [hw a, one_mul]

theorem sum_by_blocks {A : Type*} [AddCommMonoid A] (c : N → R) (f : N → A) :
    (∑ a, ∑ j ∈ fiber c a, f j) = ∑ j, f j := by
  simpa [fiber] using (Finset.sum_fiberwise_eq_sum_filter Finset.univ Finset.univ c f)

theorem sum_uniform (c : N → R) (w : R → ℝ) (hw : Normalized c w)
    (f : R → ℝ) :
    (∑ j, w (c j) * w (c j) * f (c j)) = ∑ a, f a := by
  rw [← sum_by_blocks c]
  exact Finset.sum_congr rfl (fun a _ => aggregate_uniform c w hw f a)

theorem embedding_normSq (c : N → R) (w : R → ℝ) (y : R → ℂ) (j : N) :
    Complex.normSq (embedding c w y j) =
      w (c j) * w (c j) * Complex.normSq (y (c j)) := by
  simp [embedding, Complex.normSq_mul, Complex.normSq_ofReal]

theorem aggregate_embedding_normSq (c : N → R) (w : R → ℝ)
    (hw : Normalized c w) (y : R → ℂ) (a : R) :
    aggregate c (fun j => Complex.normSq (embedding c w y j)) a =
      Complex.normSq (y a) := by
  simp_rw [embedding_normSq]
  exact aggregate_uniform c w hw (fun a => Complex.normSq (y a)) a

/-- The entrywise form of CG = Gbar C, with the paper's column convention. -/
def Lumpable (c : N → R) (G : N → N → ℝ) (Gb : R → R → ℝ) : Prop :=
  ∀ a k, (∑ j ∈ fiber c a, G j k) = Gb a (c k)

theorem aggregate_transition (c : N → R) (w : R → ℝ)
    (hw : Normalized c w) (G : N → N → ℝ) (Gb : R → R → ℝ)
    (hL : Lumpable c G Gb) (z : R → ℂ) (a : R) :
    aggregate c (fun j => ∑ k, G j k * Complex.normSq (embedding c w z k)) a =
      ∑ b, Gb a b * Complex.normSq (z b) := by
  unfold aggregate
  rw [Finset.sum_comm]
  simp only [← Finset.sum_mul, hL, embedding_normSq]
  calc
    _ = ∑ k, w (c k) * w (c k) * (Gb a (c k) * Complex.normSq (z (c k))) := by
      apply Finset.sum_congr rfl
      intro k _
      rw [hL a k]
      ring
    _ = _ := sum_uniform c w hw (fun b => Gb a b * Complex.normSq (z b))

theorem aggregate_cross (c : N → R) (w : R → ℝ)
    (hw : Normalized c w) (G : N → N → ℝ) (Gb : R → R → ℝ)
    (hD : ∀ z, D G (embedding c w z) = embedding c w (D Gb z))
    (y z : R → ℂ) (a : R) :
    aggregate c (fun j => 2 * (star (embedding c w y j) *
      D G (embedding c w z) j).re) a = 2 * (star (y a) * D Gb z a).re := by
  rw [hD z]
  have h (j : N) :
      2 * (star (embedding c w y j) * embedding c w (D Gb z) j).re =
      w (c j) * w (c j) * (2 * (star (y (c j)) * D Gb z (c j)).re) := by
    simp [embedding, Complex.mul_re, Complex.mul_im]
    ring
  simp_rw [h]
  exact aggregate_uniform c w hw (fun b => 2 * (star (y b) * D Gb z b).re) a

/-- Both physical measurements are derived here from CG=Gbar C and DB=B Dbar. -/
theorem physical_measurements (c : N → R) (w : R → ℝ)
    (hw : Normalized c w) (G : N → N → ℝ) (Gb : R → R → ℝ)
    (hG : Stochastic G) (hGb : Stochastic Gb) (hL : Lumpable c G Gb)
    (hD : ∀ z, D G (embedding c w z) = embedding c w (D Gb z))
    (y z : R → ℂ) (a : R) :
    aggregate c (sourceProbability (pairState G (embedding c w y,embedding c w z))) a =
        sourceProbability (pairState Gb (y,z)) a ∧
    aggregate c (destinationProbability (pairState G (embedding c w y,embedding c w z))) a =
        destinationProbability (pairState Gb (y,z)) a := by
  have hs (y z : R → ℂ) :
      aggregate c (sourceProbability (pairState G (embedding c w y,embedding c w z))) a =
        sourceProbability (pairState Gb (y,z)) a := by
    unfold aggregate
    simp_rw [sourceProbability_pairState G hG, sourceProbability_pairState Gb hGb]
    unfold sourceFormula
    change (∑ j ∈ fiber c a, _) = _
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
    rw [show (∑ j ∈ fiber c a, Complex.normSq (embedding c w y j)) =
      Complex.normSq (y a) from aggregate_embedding_normSq c w hw y a]
    rw [show (∑ j ∈ fiber c a, ∑ k, G j k * Complex.normSq (embedding c w z k)) =
      ∑ b, Gb a b * Complex.normSq (z b) from aggregate_transition c w hw G Gb hL z a]
    rw [show (∑ j ∈ fiber c a, 2 * (star (embedding c w y j) * D G (embedding c w z) j).re) =
      2 * (star (y a) * D Gb z a).re from aggregate_cross c w hw G Gb hD y z a]
  constructor
  · exact hs y z
  · have h := hs z y
    unfold aggregate at h ⊢
    simp_rw [destinationProbability_pairState G hG, destinationProbability_pairState Gb hGb]
    simpa only [sourceProbability_pairState G hG, sourceProbability_pairState Gb hGb] using h

/-- Equality at every recorded time for the actual two-register walk. -/
theorem all_physical_times (c : N → R) (w : R → ℝ)
    (hw : Normalized c w) (G : N → N → ℝ) (Gb : R → R → ℝ)
    (hG : Stochastic G) (hGb : Stochastic Gb) (hL : Lumpable c G Gb)
    (hD : ∀ z, D G (embedding c w z) = embedding c w (D Gb z))
    (y : R → ℂ) (t : ℕ) (a : R) :
    aggregate c (sourceProbability
      (evolve (recordedWalk G) t (pairState G (embedding c w y,0)))) a =
        sourceProbability (evolve (recordedWalk Gb) t (pairState Gb (y,0))) a ∧
    aggregate c (destinationProbability
      (evolve (recordedWalk G) t (pairState G (embedding c w y,0)))) a =
        destinationProbability (evolve (recordedWalk Gb) t (pairState Gb (y,0))) a := by
  have hi := QPPR.EC2.all_recorded_coefficient_times
    (D G) (D Gb) (embedding c w) hD t (y,0)
  have hzero : QPPR.EC2.embedPair (embedding c w) (y,0) = (embedding c w y,0) := by
    simp [QPPR.EC2.embedPair]
  rw [hzero] at hi
  simp_rw [trajectory_pairState G hG, trajectory_pairState Gb hGb, ← hi]
  exact physical_measurements c w hw G Gb hG hGb hL hD _ _ a

noncomputable def weight (c : N → R) (a : R) : ℝ :=
  (Real.sqrt ((fiber c a).card : ℝ))⁻¹

theorem fiber_card_pos (c : N → R) (hc : Function.Surjective c) (a : R) :
    0 < ((fiber c a).card : ℝ) := by
  obtain ⟨j,hj⟩ := hc a
  have h : 0 < (fiber c a).card := Finset.card_pos.mpr ⟨j,by simp [fiber,hj]⟩
  exact_mod_cast h

theorem weight_normalized (c : N → R) (hc : Function.Surjective c) :
    Normalized c (weight c) := by
  intro a
  have hm := fiber_card_pos c hc a
  unfold weight
  rw [← sq, inv_pow, Real.sq_sqrt hm.le, mul_inv_cancel₀ (ne_of_gt hm)]

theorem initial_alignment (c : N → R) (hc : Function.Surjective c)
    (v : N → ℝ) (hconst : ∀ i j, c i = c j → v i = v j) :
    embedding c (weight c) (fun a => (Real.sqrt (aggregate c v a) : ℂ)) =
      fun j => (Real.sqrt (v j) : ℂ) := by
  funext j
  have hv : aggregate c v (c j) = ((fiber c (c j)).card : ℝ) * v j := by
    unfold aggregate
    calc
      _ = ∑ k ∈ fiber c (c j), v j := by
        apply Finset.sum_congr rfl
        intro k hk
        exact hconst k j (Finset.mem_filter.mp hk).2
      _ = _ := by simp
  have hm := fiber_card_pos c hc (c j)
  change (weight c (c j) : ℂ) *
    (Real.sqrt (aggregate c v (c j)) : ℂ) = (Real.sqrt (v j) : ℂ)
  unfold weight
  rw [hv, Real.sqrt_mul hm.le]
  norm_cast
  rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt (Real.sqrt_pos.mpr hm)), one_mul]

/-- Proposition 2 at every recorded time, from its original matrix conditions.
    B is the actual normalized block embedding and vbar is the actual block sum. -/
theorem proposition2_concrete_time (c : N → R) (hc : Function.Surjective c)
    (G : N → N → ℝ) (Gb : R → R → ℝ) (v : N → ℝ)
    (hG : Stochastic G) (hGb : Stochastic Gb) (hL : Lumpable c G Gb)
    (hD : ∀ z, D G (embedding c (weight c) z) = embedding c (weight c) (D Gb z))
    (hconst : ∀ i j, c i = c j → v i = v j) :
    proposition2_measured_target (fiber c)
      (evolve (recordedWalk G) · (initial G v))
      (evolve (recordedWalk Gb) · (initial Gb (aggregate c v))) := by
  intro t a
  have h := all_physical_times c (weight c) (weight_normalized c hc)
    G Gb hG hGb hL hD (fun a => (Real.sqrt (aggregate c v a) : ℂ)) t a
  rw [initial_alignment c hc v hconst] at h
  simpa only [initial_pairState, aggregate, blockAggregate] using h

end QPPR.Manuscript.PhysicalQuotient
