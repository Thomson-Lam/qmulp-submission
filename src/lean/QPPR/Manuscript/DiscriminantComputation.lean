import QPPR.Manuscript.PhysicalWalk
import QPPR.Manuscript.UnitaryWalk
import Mathlib.Analysis.Matrix.Spectrum

namespace QPPR.Manuscript.DiscriminantComputation
open QPPR.Core QPPR.Manuscript.ConcreteSymmetry QPPR.Manuscript.PhysicalWalk
open scoped Matrix
variable {N : Type*} [Fintype N] [DecidableEq N]

noncomputable def K (G : N → N → ℝ) : Matrix N N ℂ :=
  fun i j => amplitude G i j * amplitude G j i

theorem K_hermitian (G : N → N → ℝ) : Matrix.IsHermitian (K G) := by
  ext i j
  change star (amplitude G j i * amplitude G i j) = amplitude G i j * amplitude G j i
  rw [star_mul, star_amplitude, star_amplitude]

noncomputable def Q (G : N → N → ℝ) : Matrix.unitaryGroup N ℂ := (K_hermitian G).eigenvectorUnitary
noncomputable def diagonal (G : N → N → ℝ) : Matrix N N ℂ :=
  Matrix.diagonal (fun i => ((K_hermitian G).eigenvalues i : ℂ))

theorem D_matrix (G : N → N → ℝ) (z : N → ℂ) : D G z = K G *ᵥ z := rfl

theorem discriminant_entries (G : N → N → ℝ) (hG : Stochastic G) (i j : N) :
    K G i j = (Real.sqrt (G i j * G j i) : ℂ) := by
  unfold K amplitude
  rw [← Complex.ofReal_mul, ← Real.sqrt_mul (hG.1 j i), mul_comm (G j i)]

theorem physical_discriminant (G : N → N → ℝ) :
    (UnitaryWalk.A G)ᴴ * UnitaryWalk.S * UnitaryWalk.A G = K G := by
  apply Matrix.ext_of_mulVec_single
  intro j
  rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Matrix.mulVec_single_one,
    UnitaryWalk.S, Matrix.permMatrix_mulVec, Matrix.mulVec_single_one]
  funext i
  change ((UnitaryWalk.A G)ᴴ *ᵥ Function.uncurry (fun a b => UnitaryWalk.A G (b,a) j)) i = K G i j
  rw [UnitaryWalk.A_adjoint_mulVec]
  change (∑ k, star (amplitude G i k) * (if k = j then amplitude G k i else 0)) =
    amplitude G i j * amplitude G j i
  simp [star_amplitude, mul_ite]

theorem discriminant_diagonalization (G : N → N → ℝ) :
    K G = (Q G : Matrix N N ℂ) * diagonal G * star (Q G : Matrix N N ℂ) := by
  simpa [Q, diagonal, Unitary.conjStarAlgAut_apply, Function.comp_def] using
    (K_hermitian G).spectral_theorem

theorem D_eigenbasis_intertwining (G : N → N → ℝ) (z : N → ℂ) :
    D G ((Q G : Matrix N N ℂ) *ᵥ z) =
      (Q G : Matrix N N ℂ) *ᵥ (diagonal G *ᵥ z) := by
  rw [D_matrix, Matrix.mulVec_mulVec, Matrix.mulVec_mulVec, discriminant_diagonalization]
  simp [Matrix.mul_assoc, Unitary.coe_star_mul_self]

noncomputable def initialCoordinates (G : N → N → ℝ) (v : N → ℝ) : N → ℂ :=
  star (Q G : Matrix N N ℂ) *ᵥ (fun j => (Real.sqrt (v j) : ℂ))

theorem restore_initial (G : N → N → ℝ) (v : N → ℝ) :
    (Q G : Matrix N N ℂ) *ᵥ initialCoordinates G v = fun j => (Real.sqrt (v j) : ℂ) := by
  have hu : (Q G : Matrix N N ℂ) * star (Q G : Matrix N N ℂ) = 1 :=
    Unitary.mul_star_self_of_mem (Q G).property
  rw [initialCoordinates, Matrix.mulVec_mulVec, hu, Matrix.one_mulVec]

/-- Exact evolution reconstructed using only the eigendecomposition of the N x N
    discriminant and its diagonal coefficient recurrence. Degenerate and endpoint
    eigenvalues need no division and are included. -/
theorem evolution_from_discriminant_eigendecomposition
    (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ) (t : ℕ) :
    evolve (recordedWalk G) t (initial G v) =
      pairState G (QPPR.EC2.embedPair (Matrix.mulVecLin (Q G : Matrix N N ℂ))
        (evolve (QPPR.EC2.recordedCoefficientStep (Matrix.mulVecLin (diagonal G))) t
          (initialCoordinates G v,0))) := by
  rw [initial_pairState, trajectory_pairState G hG]
  have h := QPPR.EC2.all_recorded_coefficient_times (D G)
    (Matrix.mulVecLin (diagonal G)) (Matrix.mulVecLin (Q G : Matrix N N ℂ))
    (D_eigenbasis_intertwining G) t (initialCoordinates G v,0)
  have hi : QPPR.EC2.embedPair (Matrix.mulVecLin (Q G : Matrix N N ℂ))
      (initialCoordinates G v,0) = ((fun j => (Real.sqrt (v j) : ℂ)),0) := by
    simp [QPPR.EC2.embedPair, restore_initial]
  rw [hi] at h
  rw [h]

theorem diagonal_action (G : N → N → ℝ) (z : N → ℂ) (i : N) :
    (diagonal G *ᵥ z) i = ((K_hermitian G).eigenvalues i : ℂ) * z i := by
  simp [diagonal, Matrix.mulVec_diagonal]

end QPPR.Manuscript.DiscriminantComputation
