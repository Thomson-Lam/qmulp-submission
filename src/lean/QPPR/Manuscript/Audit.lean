import QPPR
import Lean.Util.CollectAxioms

/-! Final statements and their transitive axiom dependencies. -/

-- P1: physical operator, spectral formula, existence and L1 rate
#check QPPR.Manuscript.ConcreteLimit.proposition1_concrete
#print axioms QPPR.Manuscript.ConcreteLimit.proposition1_concrete
#check QPPR.Manuscript.ConcreteLimit.proposition1_rate
#print axioms QPPR.Manuscript.ConcreteLimit.proposition1_rate
#check QPPR.Manuscript.ConcreteLimit.destination_convergence
#print axioms QPPR.Manuscript.ConcreteLimit.destination_convergence

-- Genuine complete orthogonal spectral projectors and unitarity
#check QPPR.Manuscript.FiniteSpectrum.projector_fixes_eigenvector
#print axioms QPPR.Manuscript.FiniteSpectrum.projector_fixes_eigenvector
#check QPPR.Manuscript.FiniteSpectrum.distinct_projectors_orthogonal
#print axioms QPPR.Manuscript.FiniteSpectrum.distinct_projectors_orthogonal
#check QPPR.Manuscript.UnitaryWalk.U_unitary
#print axioms QPPR.Manuscript.UnitaryWalk.U_unitary

-- Physical discriminant and exact computation from its eigendecomposition
#check QPPR.Manuscript.DiscriminantComputation.physical_discriminant
#print axioms QPPR.Manuscript.DiscriminantComputation.physical_discriminant
#check QPPR.Manuscript.DiscriminantComputation.discriminant_entries
#print axioms QPPR.Manuscript.DiscriminantComputation.discriminant_entries
#check QPPR.Manuscript.DiscriminantComputation.evolution_from_discriminant_eigendecomposition
#print axioms QPPR.Manuscript.DiscriminantComputation.evolution_from_discriminant_eigendecomposition

-- P2: both measurements, all times, finite and limiting averages
#check QPPR.Manuscript.PhysicalQuotient.proposition2_concrete_time
#print axioms QPPR.Manuscript.PhysicalQuotient.proposition2_concrete_time
#check QPPR.Manuscript.CompletePropositions.quotient_stochastic
#print axioms QPPR.Manuscript.CompletePropositions.quotient_stochastic
#check QPPR.Manuscript.CompletePropositions.proposition2_finite
#print axioms QPPR.Manuscript.CompletePropositions.proposition2_finite
#check QPPR.Manuscript.CompletePropositions.proposition2_limiting
#print axioms QPPR.Manuscript.CompletePropositions.proposition2_limiting
#check QPPR.Manuscript.CompletePropositions.proposition2_member_time
#print axioms QPPR.Manuscript.CompletePropositions.proposition2_member_time
#check QPPR.Manuscript.CompletePropositions.proposition2_member_finite
#print axioms QPPR.Manuscript.CompletePropositions.proposition2_member_finite
#check QPPR.Manuscript.CompletePropositions.proposition2_member_limiting
#print axioms QPPR.Manuscript.CompletePropositions.proposition2_member_limiting

-- P3: existence of BOTH limits and all stated symmetry equalities
#check QPPR.Manuscript.ClassicalLimit.classical_cesaro_exists
#print axioms QPPR.Manuscript.ClassicalLimit.classical_cesaro_exists
#check QPPR.Manuscript.CompletePropositions.proposition3_complete
#print axioms QPPR.Manuscript.CompletePropositions.proposition3_complete


#check QPPR.Manuscript.CompletePropositions.proposition1_discriminant
#print axioms QPPR.Manuscript.CompletePropositions.proposition1_discriminant

open Lean Elab Command in
run_cmd do
  let approved : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let targets : Array Name := #[
    ``QPPR.Manuscript.ConcreteLimit.proposition1_concrete,
    ``QPPR.Manuscript.ConcreteLimit.proposition1_rate,
    ``QPPR.Manuscript.ConcreteLimit.destination_convergence,
    ``QPPR.Manuscript.FiniteSpectrum.projector_fixes_eigenvector,
    ``QPPR.Manuscript.FiniteSpectrum.distinct_projectors_orthogonal,
    ``QPPR.Manuscript.UnitaryWalk.U_unitary,
    ``QPPR.Manuscript.DiscriminantComputation.physical_discriminant,
    ``QPPR.Manuscript.DiscriminantComputation.discriminant_entries,
    ``QPPR.Manuscript.DiscriminantComputation.evolution_from_discriminant_eigendecomposition,
    ``QPPR.Manuscript.PhysicalQuotient.proposition2_concrete_time,
    ``QPPR.Manuscript.CompletePropositions.quotient_stochastic,
    ``QPPR.Manuscript.CompletePropositions.proposition2_finite,
    ``QPPR.Manuscript.CompletePropositions.proposition2_limiting,
    ``QPPR.Manuscript.CompletePropositions.proposition2_member_time,
    ``QPPR.Manuscript.CompletePropositions.proposition2_member_finite,
    ``QPPR.Manuscript.CompletePropositions.proposition2_member_limiting,
    ``QPPR.Manuscript.ClassicalLimit.classical_cesaro_exists,
    ``QPPR.Manuscript.CompletePropositions.proposition3_complete,
    ``QPPR.Manuscript.CompletePropositions.proposition1_discriminant]
  for thm in targets do
    match (← getConstInfo thm) with
    | .thmInfo _ => pure ()
    | _ => throwError "Audit target is not a theorem: {thm}"
    let axioms ← collectAxioms thm
    for ax in axioms do
      unless approved.contains ax do
        throwError "Unapproved axiom {ax} in {thm}"
  logInfo m!"AXIOM AUDIT PASSED: {targets.size} theorems; only propext, Classical.choice and Quot.sound."

