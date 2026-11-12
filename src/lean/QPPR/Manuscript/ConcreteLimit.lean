import QPPR.Manuscript.UnitaryWalk
import QPPR.Manuscript.FiniteSpectrum
import QPPR.Manuscript.MeasuredSpectrum
import QPPR.EC1.SpectralEvolution
import Mathlib.Analysis.CStarAlgebra.Spectrum

namespace QPPR.Manuscript.ConcreteLimit
open QPPR.Core QPPR.Manuscript.ConcreteSymmetry QPPR.Manuscript.PhysicalWalk
open QPPR.Manuscript.UnitaryWalk QPPR.Manuscript.FiniteSpectrum
open QPPR.Manuscript.MeasuredSpectrum
open scoped Matrix Matrix.Norms.L2Operator
variable {N : Type*} [Fintype N] [DecidableEq N] [Nonempty N]

noncomputable def phi (G : N → N → ℝ) (v : N → ℝ)
    (z : spectrum ℂ (U G)) : State N :=
  Function.curry (projector (U G) z *ᵥ Function.uncurry (initial G v))

theorem physical_matrix_evolution (G : N → N → ℝ) (psi : State N) (t : ℕ) :
    evolve (Matrix.mulVecLin (U G)) t (Function.uncurry psi) =
      Function.uncurry (evolve (recordedWalk G) t psi) := by
  induction t with
  | zero => rfl
  | succ t ih =>
    change U G *ᵥ evolve (Matrix.mulVecLin (U G)) t (Function.uncurry psi) = _
    rw [ih, U_mulVec]
    rfl

theorem spectral_coordinate (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (t : ℕ) (j k : N) :
    evolve (recordedWalk G) t (initial G v) j k =
      ∑ z : spectrum ℂ (U G), (z : ℂ) ^ t * phi G v z j k := by
  have hn := isStarNormal_of_mem_unitary (U_unitary G hG)
  let components (z : spectrum ℂ (U G)) := projector (U G) z *ᵥ Function.uncurry (initial G v)
  have hsum : (∑ z, components z) = Function.uncurry (initial G v) := by
    dsimp only [components]
    rw [← Matrix.sum_mulVec, sum_projectors (U G) hn, Matrix.one_mulVec]
  have he (z : spectrum ℂ (U G)) :
      Matrix.mulVecLin (U G) (components z) = (z : ℂ) • components z := by
    change U G *ᵥ (projector (U G) z *ᵥ Function.uncurry (initial G v)) = _
    rw [Matrix.mulVec_mulVec, projector_eigen (U G) hn z, Matrix.smul_mulVec]
  have ht := QPPR.EC1.spectral_evolve (Matrix.mulVecLin (U G))
    (fun z : spectrum ℂ (U G) => (z : ℂ)) components he t
  rw [hsum, physical_matrix_evolution] at ht
  have := congrFun ht (j,k)
  simpa [phi, components] using this

noncomputable def limitScore (G : N → N → ℝ) (v : N → ℝ) (j : N) : ℝ :=
  spectralSourceScore (phi G v) j

noncomputable def errorConstant (G : N → N → ℝ) (v : N → ℝ) : ℝ :=
  ∑ j, ∑ k, crossConstant (fun z : spectrum ℂ (U G) => (z : ℂ)) (fun z => phi G v z j k)

theorem coordinate_error_bound (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (n : ℕ) (hn : 0 < n) (j k : N) :
    |((∑ t ∈ Finset.range n, Complex.normSq (evolve (recordedWalk G) t (initial G v) j k)) /
      (n : ℝ)) - ∑ z : spectrum ℂ (U G), Complex.normSq (phi G v z j k)| ≤
        crossConstant (fun z : spectrum ℂ (U G) => (z : ℂ)) (fun z => phi G v z j k) / (n : ℝ) := by
  have hb := average_error_bound (fun z : spectrum ℂ (U G) => (z : ℂ))
    (fun z => phi G v z j k) Subtype.val_injective
    (fun z => spectrum.norm_eq_one_of_unitary (U_unitary G hG) z.property) n hn
  simp_rw [← spectral_coordinate G hG v] at hb
  have hcast :
      average (fun t => (Complex.normSq (evolve (recordedWalk G) t (initial G v) j k) : ℂ)) n -
        (∑ z : spectrum ℂ (U G), (Complex.normSq (phi G v z j k) : ℂ)) =
      (((∑ t ∈ Finset.range n, Complex.normSq (evolve (recordedWalk G) t (initial G v) j k)) /
        (n : ℝ) - ∑ z : spectrum ℂ (U G), Complex.normSq (phi G v z j k) : ℝ) : ℂ) := by
    simp [average]
  rw [hcast] at hb
  rw [Complex.norm_real, Real.norm_eq_abs] at hb
  exact hb

theorem source_error_identity {I : Type*} [Fintype I]
    (psi : ℕ → State N) (ph : I → State N) (n : ℕ) (j : N) :
    finiteSourceScore psi n j - spectralSourceScore ph j =
      ∑ k, ((∑ t ∈ Finset.range n, Complex.normSq (psi t j k)) / (n : ℝ) -
        ∑ z, Complex.normSq (ph z j k)) := by
  unfold finiteSourceScore spectralSourceScore sourceProbability
  rw [Finset.sum_comm (s := Finset.range n), Finset.sum_comm (s := (Finset.univ : Finset I))]
  simp [Finset.sum_sub_distrib, Finset.sum_div]

theorem proposition1_rate (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (n : ℕ) (hn : 0 < n) :
    (∑ j, |finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n j - limitScore G v j|) ≤
      errorConstant G v / (n : ℝ) := by
  unfold limitScore
  simp_rw [source_error_identity]
  calc
    _ ≤ ∑ j, ∑ k, |(∑ t ∈ Finset.range n,
          Complex.normSq (evolve (recordedWalk G) t (initial G v) j k)) / (n : ℝ) -
          ∑ z : spectrum ℂ (U G), Complex.normSq (phi G v z j k)| := by
      apply Finset.sum_le_sum
      intro j _
      simpa only [Real.norm_eq_abs, swap] using (norm_sum_le Finset.univ
        (fun k => (∑ t ∈ Finset.range n, Complex.normSq (evolve (recordedWalk G) t (initial G v) j k)) /
          (n : ℝ) - ∑ z : spectrum ℂ (U G), Complex.normSq (phi G v z j k)))
    _ ≤ ∑ j, ∑ k, crossConstant (fun z : spectrum ℂ (U G) => (z : ℂ))
        (fun z => phi G v z j k) / (n : ℝ) := by
      apply Finset.sum_le_sum
      intro j _
      apply Finset.sum_le_sum
      intro k _
      exact coordinate_error_bound G hG v n hn j k
    _ = _ := by simp [errorConstant, Finset.sum_div]

theorem errorConstant_nonnegative (G : N → N → ℝ) (v : N → ℝ) :
    0 ≤ errorConstant G v := by
  classical
  unfold errorConstant crossConstant
  positivity

theorem proposition1_convergence (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (j : N) :
    Filter.Tendsto (fun n => finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n j)
      Filter.atTop (nhds (limitScore G v j)) := by
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  have hc := tendsto_const_div_atTop_nhds_zero_nat (errorConstant G v)
  apply squeeze_zero' (Filter.Eventually.of_forall (fun n => norm_nonneg _))
    (Filter.eventually_atTop.mpr ⟨1,fun n hn => ?_⟩) hc
  have hb := proposition1_rate G hG v n (by omega)
  have he : |finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n j - limitScore G v j| ≤
      ∑ i, |finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n i - limitScore G v i| :=
    Finset.single_le_sum (f := fun i => |finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n i - limitScore G v i|)
      (fun i _ => abs_nonneg _) (Finset.mem_univ j)
  simpa only [Real.norm_eq_abs] using he.trans hb

/-- The spectral score formula, existence and an explicit L1 c/n bound,
    for the concrete QPPR operator. No limit or spectral decomposition is assumed. -/
theorem proposition1_concrete (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ) :
    proposition1_score_target (evolve (recordedWalk G) · (initial G v)) (phi G v) := by
  constructor
  · exact proposition1_convergence G hG v
  · exact ⟨errorConstant G v,errorConstant_nonnegative G v,proposition1_rate G hG v⟩

noncomputable def finiteDestinationScore (G : N → N → ℝ) (v : N → ℝ) (n : ℕ) (j : N) : ℝ :=
  finiteSourceScore (fun t => swap (evolve (recordedWalk G) t (initial G v))) n j

noncomputable def limitDestinationScore (G : N → N → ℝ) (v : N → ℝ) (j : N) : ℝ :=
  spectralSourceScore (fun z => swap (phi G v z)) j

theorem destination_rate (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (n : ℕ) (hn : 0 < n) :
    (∑ j, |finiteDestinationScore G v n j - limitDestinationScore G v j|) ≤
      errorConstant G v / (n : ℝ) := by
  unfold finiteDestinationScore limitDestinationScore
  simp_rw [source_error_identity]
  calc
    _ ≤ ∑ j, ∑ k, |(∑ t ∈ Finset.range n,
          Complex.normSq (evolve (recordedWalk G) t (initial G v) k j)) / (n : ℝ) -
          ∑ z : spectrum ℂ (U G), Complex.normSq (phi G v z k j)| := by
      apply Finset.sum_le_sum
      intro j _
      simpa only [Real.norm_eq_abs, swap] using (norm_sum_le Finset.univ
        (fun k => (∑ t ∈ Finset.range n, Complex.normSq (evolve (recordedWalk G) t (initial G v) k j)) /
          (n : ℝ) - ∑ z : spectrum ℂ (U G), Complex.normSq (phi G v z k j)))
    _ ≤ ∑ j, ∑ k, crossConstant (fun z : spectrum ℂ (U G) => (z : ℂ))
        (fun z => phi G v z k j) / (n : ℝ) := by
      apply Finset.sum_le_sum
      intro j _
      apply Finset.sum_le_sum
      intro k _
      exact coordinate_error_bound G hG v n hn k j
    _ = _ := by rw [Finset.sum_comm]; simp [errorConstant, Finset.sum_div]

theorem destination_convergence (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (j : N) :
    Filter.Tendsto (fun n => finiteDestinationScore G v n j)
      Filter.atTop (nhds (limitDestinationScore G v j)) := by
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  apply squeeze_zero' (Filter.Eventually.of_forall (fun n => norm_nonneg _))
    (Filter.eventually_atTop.mpr ⟨1,fun n hn => ?_⟩)
    (tendsto_const_div_atTop_nhds_zero_nat (errorConstant G v))
  have hb := destination_rate G hG v n (by omega)
  have he : |finiteDestinationScore G v n j - limitDestinationScore G v j| ≤
      ∑ i, |finiteDestinationScore G v n i - limitDestinationScore G v i| :=
    Finset.single_le_sum (f := fun i => |finiteDestinationScore G v n i - limitDestinationScore G v i|)
      (fun i _ => abs_nonneg _) (Finset.mem_univ j)
  simpa only [Real.norm_eq_abs] using he.trans hb

end QPPR.Manuscript.ConcreteLimit
