import QPPR.Core.Imports
import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.CStarAlgebra.ContinuousFunctionalCalculus.Basic
import Mathlib.LinearAlgebra.Eigenspace.Minpoly

namespace QPPR.Manuscript.FiniteSpectrum
open scoped Matrix Matrix.Norms.L2Operator
variable {J : Type*} [Fintype J] [DecidableEq J] [Nonempty J]

noncomputable instance matrixCStar : CStarAlgebra (Matrix J J ℂ) where

noncomputable instance spectrumFintype (T : Matrix J J ℂ) : Fintype (spectrum ℂ T) :=
  Fintype.ofFinite _

noncomputable def projector (T : Matrix J J ℂ) (z : ℂ) : Matrix J J ℂ :=
  by classical exact cfc (fun x : ℂ => if x = z then (1 : ℂ) else 0) T

theorem projector_continuous (T : Matrix J J ℂ) (z : ℂ) :
    ContinuousOn (fun x : ℂ => if x = z then (1 : ℂ) else 0) (spectrum ℂ T) :=
  (Matrix.finite_spectrum T).continuousOn _

theorem sum_projectors (T : Matrix J J ℂ) (hT : IsStarNormal T) :
    ∑ z : spectrum ℂ T, projector T z = 1 := by
  classical
  have hsum : cfc (∑ z : spectrum ℂ T, fun x : ℂ => if x = z then (1 : ℂ) else 0) T = 1 := by
    calc
      _ = cfc (fun _ : ℂ => 1) T := by
        apply cfc_congr
        intro x hx
        simp only [Finset.sum_apply]
        have h : (∑ z : spectrum ℂ T, if x = (z : ℂ) then (1 : ℂ) else 0) = 1 := by
          rw [Finset.sum_eq_single (⟨x,hx⟩ : spectrum ℂ T)]
          · simp
          · intro b _ hb
            have hn : x ≠ (b : ℂ) := fun he => hb (Subtype.ext he.symm)
            simp [hn]
          · simp
        exact h
      _ = 1 := cfc_const_one ℂ T hT
  have hmap := cfc_sum_univ
    (fun z : spectrum ℂ T => fun x : ℂ => if x = z then (1 : ℂ) else 0) T
    (fun z => by exact projector_continuous T z)
  exact hmap.symm.trans hsum

theorem projector_eigen (T : Matrix J J ℂ) (hT : IsStarNormal T) (z : ℂ) :
    T * projector T z = z • projector T z := by
  classical
  have hc := projector_continuous T z
  calc
    T * projector T z = cfc (fun x : ℂ => x * (if x = z then (1 : ℂ) else 0)) T := by
      have hm := cfc_mul (fun x : ℂ => x)
        (fun x : ℂ => if x = z then (1 : ℂ) else 0) T
        (by exact continuousOn_id) (by exact hc)
      rw [cfc_id' ℂ T hT] at hm
      exact hm.symm
    _ = cfc (fun x : ℂ => z * (if x = z then (1 : ℂ) else 0)) T := by
      apply cfc_congr
      intro x hx
      dsimp only
      split_ifs with h <;> simp_all
    _ = z • projector T z := by
      rw [cfc_const_mul z _ T (by exact hc)]
      rfl

theorem projector_idempotent (T : Matrix J J ℂ) (z : ℂ) :
    projector T z * projector T z = projector T z := by
  classical
  rw [projector, ← cfc_mul _ _ T (by exact projector_continuous T z) (by exact projector_continuous T z)]
  apply cfc_congr
  intro x hx
  dsimp only
  split_ifs <;> simp

theorem projector_selfadjoint (T : Matrix J J ℂ) (z : ℂ) :
    star (projector T z) = projector T z := by
  classical
  rw [projector, ← cfc_star]
  apply cfc_congr
  intro x hx
  dsimp only
  split_ifs <;> simp

theorem distinct_projectors_orthogonal (T : Matrix J J ℂ) (z w : ℂ) (hzw : z ≠ w) :
    projector T z * projector T w = 0 := by
  classical
  rw [projector, projector, ← cfc_mul _ _ T
    (by exact projector_continuous T z) (by exact projector_continuous T w)]
  calc
    _ = cfc (fun _ : ℂ => 0) T := by
      apply cfc_congr
      intro x hx
      dsimp only
      by_cases hz : x = z <;> by_cases hw : x = w <;> simp_all
    _ = 0 := cfc_const_zero ℂ T

theorem projector_commutes (T : Matrix J J ℂ) (hT : IsStarNormal T) (z : ℂ) :
    Commute (projector T z) T := by
  classical
  have h := cfc_commute_cfc (fun x : ℂ => if x = z then (1 : ℂ) else 0) (fun x : ℂ => x) T
  rwa [cfc_id' ℂ T hT] at h

/-- These are the complete eigenspace projectors: every vector in the
    eigenspace at z is fixed by the corresponding spectral projector. -/
theorem projector_fixes_eigenvector (T : Matrix J J ℂ) (hT : IsStarNormal T)
    (z : spectrum ℂ T) (x : J → ℂ) (hx : T *ᵥ x = (z : ℂ) • x) :
    projector T z *ᵥ x = x := by
  classical
  have hzero (w : spectrum ℂ T) (hw : w ≠ z) : projector T w *ᵥ x = 0 := by
    have hcomm := (projector_commutes T hT w).eq
    have hv := congrArg (fun M : Matrix J J ℂ => M *ᵥ x) hcomm
    rw [← Matrix.mulVec_mulVec, hx, Matrix.mulVec_smul,
      projector_eigen T hT w, Matrix.smul_mulVec] at hv
    have he : ((z : ℂ) - (w : ℂ)) • (projector T w *ᵥ x) = 0 := by
      rw [sub_smul, hv, sub_self]
    exact (smul_eq_zero.mp he).resolve_left (sub_ne_zero.mpr
      (fun h => hw (Subtype.ext h.symm)))
  have hsum := congrArg (fun M : Matrix J J ℂ => M *ᵥ x) (sum_projectors T hT)
  rw [Matrix.sum_mulVec, Matrix.one_mulVec] at hsum
  rw [Finset.sum_eq_single z] at hsum
  · exact hsum
  · intro w _ hw
    exact hzero w hw
  · simp

end QPPR.Manuscript.FiniteSpectrum
