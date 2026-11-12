import QPPR.Core.Imports

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript

/-- Born-rule marginal of the first (source) register. -/
noncomputable def sourceProbability
    {N : Type*} [Fintype N] (psi : N → N → ℂ) (j : N) : ℝ :=
  ∑ k : N, Complex.normSq (psi j k)

/-- Born-rule marginal of the second (destination) register. -/
noncomputable def destinationProbability
    {N : Type*} [Fintype N] (psi : N → N → ℂ) (k : N) : ℝ :=
  ∑ j : N, Complex.normSq (psi j k)

/-- Finite average over the paper's recorded times (zero gives 0). -/
noncomputable def finiteSourceScore
    {N : Type*} [Fintype N]
    (psi : ℕ → N → N → ℂ) (tau : ℕ) (j : N) : ℝ :=
  (∑ t ∈ Finset.range tau, sourceProbability (psi t) j) / (tau : ℝ)

/-- Spectral formula on the right of Proposition 1, with an index for
DISTINCT eigenvalues of U: phi eig = P_eig |Psi_0>.
This is a definition and does not construct the projectors. -/
noncomputable def spectralSourceScore
    {N I : Type*} [Fintype N] [Fintype I]
    (phi : I → N → N → ℂ) (j : N) : ℝ :=
  ∑ eig : I, sourceProbability (phi eig) j

/-- Proposition 1's two score-level conclusions: spectral limit and O(1/tau)
L1 bound. A future proof must instantiate psi via U^t A sqrt(v) and
phi with genuine spectral projections of the concrete unitary U. -/
def proposition1_score_target
    {N I : Type*} [Fintype N] [Fintype I]
    (psi : ℕ → N → N → ℂ) (phi : I → N → N → ℂ) : Prop :=
  (∀ j : N,
    Filter.Tendsto (fun tau : ℕ => finiteSourceScore psi tau j)
      Filter.atTop (nhds (spectralSourceScore phi j))) ∧
  (∃ c : ℝ, 0 ≤ c ∧ ∀ tau : ℕ, 0 < tau →
    (∑ j : N, |finiteSourceScore psi tau j - spectralSourceScore phi j|)
      ≤ c / (tau : ℝ))

/-- Class aggregation of an arbitrary node score vector. -/
noncomputable def blockAggregate
    {N R : Type*} [Fintype N]
    (block : R → Finset N) (p : N → ℝ) (a : R) : ℝ :=
  ∑ j ∈ block a, p j

/-- Proposition 2's conclusion, BEFORE its assumptions CG=Gbar C,
DB=B Dbar, block-constant v and the definition of the quotient walk
have been discharged. Both source and destination registers are included.
Full/reduced sequences should be instantiated with the physical walks. -/
def proposition2_measured_target
    {N R : Type*} [Fintype N] [Fintype R]
    (block : R → Finset N)
    (full : ℕ → N → N → ℂ) (reduced : ℕ → R → R → ℂ) : Prop :=
  ∀ t : ℕ, ∀ a : R,
    blockAggregate block (sourceProbability (full t)) a =
      sourceProbability (reduced t) a ∧
    blockAggregate block (destinationProbability (full t)) a =
      destinationProbability (reduced t) a

/-- Proposition 3's at-every-time conclusion for classical and quantum
scores at two exchangeable labels. Existence of limiting scores and
equality of their limits need to be attached to this in the full proof. -/
def proposition3_time_target
    {N : Type*} [Fintype N]
    (quantum : ℕ → N → N → ℂ)
    (classical : ℕ → N → ℝ) (i j : N) : Prop :=
  (∀ t : ℕ, sourceProbability (quantum t) i =
                 sourceProbability (quantum t) j) ∧
  (∀ t : ℕ, classical t i = classical t j)

/-- A verified deterministic consequence of pointwise quantum symmetry:
source-register equality also holds for every finite Cesaro average. -/
theorem proposition3_pointwise_implies_finite
    {N : Type*} [Fintype N]
    (quantum : ℕ → N → N → ℂ) (i j : N)
    (h : ∀ t : ℕ, sourceProbability (quantum t) i =
                     sourceProbability (quantum t) j)
    (tau : ℕ) :
    finiteSourceScore quantum tau i = finiteSourceScore quantum tau j := by
  have hsum :
      (∑ t ∈ Finset.range tau, sourceProbability (quantum t) i) =
        (∑ t ∈ Finset.range tau, sourceProbability (quantum t) j) := by
    apply Finset.sum_congr rfl
    intro t ht
    exact h t
  unfold finiteSourceScore
  rw [hsum]

end QPPR.Manuscript
