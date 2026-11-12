import QPPR.Manuscript.ConcreteSymmetry
import QPPR.EC2.DiscriminantIntertwining

/-! Physical amplitudes, the discriminant recurrence and Born marginals.
    All identities below refer to the explicit two-register walk. -/
namespace QPPR.Manuscript.PhysicalWalk

open QPPR.Core QPPR.Manuscript.ConcreteSymmetry
variable {N : Type*} [Fintype N]

def Stochastic (G : N → N → ℝ) : Prop :=
  (∀ i j, 0 ≤ G i j) ∧ (∀ j, ∑ i, G i j = 1)

noncomputable def D (G : N → N → ℝ) : (N → ℂ) →ₗ[ℂ] (N → ℂ) where
  toFun z j := ∑ k, amplitude G j k * amplitude G k j * z k
  map_add' y z := by funext j; simp [mul_add, Finset.sum_add_distrib]
  map_smul' c z := by funext j; simp [Finset.mul_sum, mul_left_comm, mul_assoc]

noncomputable def pairState (G : N → N → ℝ) (p : (N → ℂ) × (N → ℂ)) : State N :=
  fun j k => p.1 j * amplitude G j k + p.2 k * amplitude G k j

theorem star_amplitude (G : N → N → ℝ) (j k : N) :
    star (amplitude G j k) = amplitude G j k := by simp [amplitude]

theorem amplitude_sq (G : N → N → ℝ) (hG : Stochastic G) (j k : N) :
    amplitude G j k * amplitude G j k = (G k j : ℂ) := by
  simpa [amplitude, ← Complex.ofReal_mul] using
    congrArg (fun x : ℝ => (x : ℂ)) (Real.mul_self_sqrt (hG.1 k j))

theorem amplitude_normSq (G : N → N → ℝ) (hG : Stochastic G) (j k : N) :
    Complex.normSq (amplitude G j k) = G k j := by
  simpa [amplitude, Complex.normSq_ofReal] using Real.mul_self_sqrt (hG.1 k j)

theorem amplitude_sum_sq (G : N → N → ℝ) (hG : Stochastic G) (j : N) :
    (∑ k, amplitude G j k * amplitude G j k) = 1 := by
  simp_rw [amplitude_sq G hG]
  rw [← Complex.ofReal_sum, hG.2 j, Complex.ofReal_one]

theorem project_pairState (G : N → N → ℝ) (hG : Stochastic G)
    (y z : N → ℂ) (j k : N) :
    project G (pairState G (y,z)) j k =
      amplitude G j k * (y j + D G z j) := by
  unfold project pairState
  simp_rw [star_amplitude, mul_add]
  rw [Finset.sum_add_distrib]
  have h : (∑ m, amplitude G j m * (y j * amplitude G j m)) = y j := by
    calc
      _ = y j * ∑ m, amplitude G j m * amplitude G j m := by
        rw [Finset.mul_sum]; congr 1; funext m; ring
      _ = y j := by rw [amplitude_sum_sq G hG j, mul_one]
  rw [h]
  rw [mul_add]
  simp [D, Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm]

theorem oneWalk_pairState (G : N → N → ℝ) (hG : Stochastic G)
    (p : (N → ℂ) × (N → ℂ)) :
    oneWalk G (pairState G p) = pairState G (QPPR.EC2.coefficientWalk (D G) p) := by
  rcases p with ⟨y,z⟩
  funext j k
  change 2 * project G (pairState G (y,z)) k j - pairState G (y,z) k j = _
  rw [project_pairState G hG]
  change 2 * (amplitude G k j * (y k + D G z k)) -
      (y k * amplitude G k j + z j * amplitude G j k) =
    (-z j) * amplitude G j k + (y k + 2 * D G z k) * amplitude G k j
  ring

theorem recordedWalk_pairState (G : N → N → ℝ) (hG : Stochastic G)
    (p : (N → ℂ) × (N → ℂ)) :
    recordedWalk G (pairState G p) =
      pairState G (QPPR.EC2.recordedCoefficientStep (D G) p) := by
  simp only [recordedWalk, QPPR.EC2.recordedCoefficientStep, oneWalk_pairState G hG]

theorem trajectory_pairState (G : N → N → ℝ) (hG : Stochastic G)
    (p : (N → ℂ) × (N → ℂ)) (t : ℕ) :
    evolve (recordedWalk G) t (pairState G p) =
      pairState G (evolve (QPPR.EC2.recordedCoefficientStep (D G)) t p) := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change recordedWalk G (evolve (recordedWalk G) t (pairState G p)) = _
    rw [ih, recordedWalk_pairState G hG]
    rfl

theorem initial_pairState (G : N → N → ℝ) (v : N → ℝ) :
    initial G v = pairState G ((fun j => (Real.sqrt (v j) : ℂ)),0) := by
  funext j k
  simp [initial, pairState]

noncomputable def sourceFormula (G : N → N → ℝ) (y z : N → ℂ) (j : N) : ℝ :=
  Complex.normSq (y j) + (∑ k, G j k * Complex.normSq (z k)) +
    2 * (star (y j) * D G z j).re

theorem sourceProbability_pairState (G : N → N → ℝ) (hG : Stochastic G)
    (y z : N → ℂ) (j : N) :
    sourceProbability (pairState G (y,z)) j = sourceFormula G y z j := by
  have hc (k : N) :
      (y j * amplitude G j k * star (z k * amplitude G k j)).re =
      (star (y j) * (amplitude G j k * amplitude G k j * z k)).re := by
    simp [amplitude, Complex.mul_re, Complex.mul_im]
    <;> ring
  have hsum : (∑ k, Complex.normSq (y j) * G k j) = Complex.normSq (y j) := by
    rw [← Finset.mul_sum, hG.2 j, mul_one]
  unfold sourceProbability pairState sourceFormula
  simp only [Complex.normSq_add, Complex.normSq_mul, amplitude_normSq G hG, hc]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum, hG.2, mul_one]
  congr 1
  · congr 1
    apply Finset.sum_congr rfl
    intro k _
    ring
  · congr 1
    calc
      _ = ∑ k, (star (y j) * (amplitude G j k * amplitude G k j * z k)).re := by
        apply Finset.sum_congr rfl
        intro k _
        exact hc k
      _ = _ := by simp [D, Finset.mul_sum, Complex.re_sum]

theorem destinationProbability_pairState (G : N → N → ℝ) (hG : Stochastic G)
    (y z : N → ℂ) (j : N) :
    destinationProbability (pairState G (y,z)) j = sourceFormula G z y j := by
  change (∑ k, Complex.normSq (y k * amplitude G k j + z j * amplitude G j k)) = _
  simp_rw [add_comm (y _ * _) (z _ * _)]
  exact sourceProbability_pairState G hG z y j

end QPPR.Manuscript.PhysicalWalk
