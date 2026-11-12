import QPPR.EC3.Symmetry
import QPPR.Manuscript.ProofTargets
import QPPR.EC3.LimitSymmetry

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.Manuscript.ConcreteSymmetry

open QPPR.Core

variable {N : Type*} [Fintype N]

/-- Finite sums are invariant under a bijective relabeling. -/
private theorem sum_perm {A : Type*} [AddCommMonoid A]
    (q : Equiv.Perm N) (f : N → A) :
    (∑ k : N, f (q k)) = ∑ k : N, f k := by
  classical
  simpa using (Equiv.Perm.sum_comp q Finset.univ f (by simp))

/-- Source j and destination k: the Szegedy transition amplitude. -/
noncomputable def amplitude (G : N → N → ℝ) (j k : N) : ℂ :=
  (Real.sqrt (G k j) : ℂ)

/-- Two-register complex amplitudes in the (source,destination) basis. -/
abbrev State (N : Type*) := N → N → ℂ

/-- Explicit A A† using the orthogonal source-labeled transition states. -/
noncomputable def project (G : N → N → ℝ) (psi : State N) : State N :=
  fun j k => amplitude G j k *
    (∑ m : N, star (amplitude G j m) * psi j m)

/-- 2 Π - I, with Π defined above. -/
noncomputable def reflect (G : N → N → ℝ) (psi : State N) : State N :=
  fun j k => 2 * project G psi j k - psi j k

/-- Swap the two quantum registers. -/
def swap (psi : State N) : State N := fun j k => psi k j

/-- W = S (2 Π - I). -/
noncomputable def oneWalk (G : N → N → ℝ) (psi : State N) : State N :=
  swap (reflect G psi)

/-- U = W², exactly two reflection-swap operations per recorded step. -/
noncomputable def recordedWalk (G : N → N → ℝ) (psi : State N) : State N :=
  oneWalk G (oneWalk G psi)

/-- Ψ₀ = A sqrt(v). -/
noncomputable def initial (G : N → N → ℝ) (v : N → ℝ) : State N :=
  fun j k => (Real.sqrt (v j) : ℂ) * amplitude G j k

/-- Simultaneous relabeling of both registers. -/
def permuteState (q : Equiv.Perm N) (psi : State N) : State N :=
  fun j k => psi (q j) (q k)

/-- Transition amplitudes are invariant when G is invariant. -/
theorem amplitude_invariant (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k) (j k : N) :
    amplitude G (q j) (q k) = amplitude G j k := by
  simp [amplitude, hG]

/-- The prepared initial two-register state is invariant. -/
theorem initial_invariant (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) :
    permuteState q (initial G v) = initial G v := by
  funext j k
  change (Real.sqrt (v (q j)) : ℂ) * amplitude G (q j) (q k) =
    (Real.sqrt (v j) : ℂ) * amplitude G j k
  rw [hv j, amplitude_invariant G q hG j k]

/-- Π = A A† commutes with the simultaneous register relabeling. -/
theorem project_equivariant (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k) (psi : State N) :
    permuteState q (project G psi) = project G (permuteState q psi) := by
  funext j k
  have hsum :
      (∑ m : N, star (amplitude G (q j) m) * psi (q j) m) =
        (∑ m : N, star (amplitude G j m) * psi (q j) (q m)) := by
    calc
      (∑ m : N, star (amplitude G (q j) m) * psi (q j) m) =
          (∑ m : N, star (amplitude G (q j) (q m)) * psi (q j) (q m)) :=
            (sum_perm q (fun m => star (amplitude G (q j) m) * psi (q j) m)).symm
      _ = (∑ m : N, star (amplitude G j m) * psi (q j) (q m)) := by
        apply Finset.sum_congr rfl
        intro m _
        rw [amplitude_invariant G q hG j m]
  change amplitude G (q j) (q k) *
        (∑ m : N, star (amplitude G (q j) m) * psi (q j) m) =
      amplitude G j k *
        (∑ m : N, star (amplitude G j m) * psi (q j) (q m))
  rw [amplitude_invariant G q hG j k, hsum]

/-- The reflection commutes with label permutations. -/
theorem reflect_equivariant (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k) (psi : State N) :
    permuteState q (reflect G psi) = reflect G (permuteState q psi) := by
  funext j k
  have hproj := congrFun (congrFun (project_equivariant G q hG psi) j) k
  change 2 * project G psi (q j) (q k) - psi (q j) (q k) =
    2 * project G (permuteState q psi) j k - psi (q j) (q k)
  change project G psi (q j) (q k) =
    project G (permuteState q psi) j k at hproj
  rw [hproj]

/-- The register swap commutes with simultaneous permutations. -/
theorem swap_equivariant (q : Equiv.Perm N) (psi : State N) :
    permuteState q (swap psi) = swap (permuteState q psi) := by
  rfl

/-- Full physical reflection-swap evolution is equivariant. -/
theorem oneWalk_equivariant (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k) (psi : State N) :
    permuteState q (oneWalk G psi) = oneWalk G (permuteState q psi) := by
  calc
    permuteState q (oneWalk G psi) =
      swap (permuteState q (reflect G psi)) := swap_equivariant q (reflect G psi)
    _ = swap (reflect G (permuteState q psi)) := by rw [reflect_equivariant G q hG]
    _ = oneWalk G (permuteState q psi) := rfl

/-- One recorded quantum step U = W² commutes with the exchange. -/
theorem recordedWalk_equivariant (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k) (psi : State N) :
    permuteState q (recordedWalk G psi) = recordedWalk G (permuteState q psi) := by
  calc
    permuteState q (recordedWalk G psi) =
      oneWalk G (permuteState q (oneWalk G psi)) :=
        oneWalk_equivariant G q hG (oneWalk G psi)
    _ = oneWalk G (oneWalk G (permuteState q psi)) := by
      rw [oneWalk_equivariant G q hG psi]
    _ = recordedWalk G (permuteState q psi) := rfl

/-- Every U^t Ψ₀ remains symmetric. -/
theorem trajectory_invariant (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (t : ℕ) :
    permuteState q (evolve (recordedWalk G) t (initial G v)) =
      evolve (recordedWalk G) t (initial G v) := by
  exact QPPR.EC3.symmetry_preserved_under_evolution
    (recordedWalk G) (permuteState q)
    (recordedWalk_equivariant G q hG) (initial G v)
    (initial_invariant G v q hG hv) t

/-- Relabeling a two-register state relabels its source Born probabilities. -/
theorem sourceProbability_equivariant (q : Equiv.Perm N) (psi : State N) (j : N) :
    sourceProbability (permuteState q psi) j = sourceProbability psi (q j) := by
  unfold sourceProbability
  change (∑ k : N, Complex.normSq (psi (q j) (q k))) =
      (∑ k : N, Complex.normSq (psi (q j) k))
  exact sum_perm q (fun k => Complex.normSq (psi (q j) k))

/-- Concrete QPPR: source probabilities of exchanged nodes are equal at EVERY t. -/
theorem quantum_scores_equal_at_time (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (t : ℕ) (j : N) :
    sourceProbability (evolve (recordedWalk G) t (initial G v)) j =
      sourceProbability (evolve (recordedWalk G) t (initial G v)) (q j) := by
  let psi := evolve (recordedWalk G) t (initial G v)
  have hstate : permuteState q psi = psi := trajectory_invariant G v q hG hv t
  calc
    sourceProbability psi j = sourceProbability (permuteState q psi) j := by rw [hstate]
    _ = sourceProbability psi (q j) := sourceProbability_equivariant q psi j

/-- Concrete source-register finite averages are equal within an exchange orbit. -/
theorem quantum_finite_scores_equal (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (tau : ℕ) (j : N) :
    finiteSourceScore (evolve (recordedWalk G) · (initial G v)) tau j =
      finiteSourceScore (evolve (recordedWalk G) · (initial G v)) tau (q j) := by
  exact QPPR.Manuscript.proposition3_pointwise_implies_finite
    (evolve (recordedWalk G) · (initial G v)) j (q j)
    (fun t => quantum_scores_equal_at_time G v q hG hv t j) tau

/-- The classical transition update Gp in the paper's column convention. -/
def classicalStep (G : N → N → ℝ) (p : N → ℝ) : N → ℝ :=
  fun i => ∑ j : N, G i j * p j

/-- Permute classical node probabilities. -/
def permuteVector (q : Equiv.Perm N) (p : N → ℝ) : N → ℝ :=
  fun j => p (q j)

/-- The concrete classical Markov update commutes with the exchange. -/
theorem classicalStep_equivariant (G : N → N → ℝ) (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k) (p : N → ℝ) :
    permuteVector q (classicalStep G p) = classicalStep G (permuteVector q p) := by
  funext i
  change (∑ j : N, G (q i) j * p j) =
    (∑ j : N, G i j * p (q j))
  calc
    (∑ j : N, G (q i) j * p j) =
      (∑ j : N, G (q i) (q j) * p (q j)) :=
        (sum_perm q (fun j => G (q i) j * p j)).symm
    _ = (∑ j : N, G i j * p (q j)) := by
      apply Finset.sum_congr rfl
      intro j _
      rw [hG i j]

/-- Each classical trajectory G^t v is invariant. -/
theorem classical_trajectory_invariant (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (t : ℕ) :
    permuteVector q (evolve (classicalStep G) t v) =
      evolve (classicalStep G) t v := by
  have hinit : permuteVector q v = v := by
    funext i
    exact hv i
  exact QPPR.EC3.symmetry_preserved_under_evolution
    (classicalStep G) (permuteVector q)
    (classicalStep_equivariant G q hG) v hinit t

/-- Concrete classical trajectories give exchanged nodes identical probabilities. -/
theorem classical_scores_equal_at_time (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (t : ℕ) (j : N) :
    evolve (classicalStep G) t v j = evolve (classicalStep G) t v (q j) := by
  have hinv := classical_trajectory_invariant G v q hG hv t
  exact (congrFun hinv j).symm

/-- Both statements of Proposition 3 at EVERY recorded time, from the
actual G-dependent Szegedy reflection-swap and classical update. -/
theorem proposition3_concrete_time (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (j : N) :
    proposition3_time_target
      (evolve (recordedWalk G) · (initial G v))
      (evolve (classicalStep G) · v) j (q j) := by
  constructor
  · intro t
    exact quantum_scores_equal_at_time G v q hG hv t j
  · intro t
    exact classical_scores_equal_at_time G v q hG hv t j

/-- The equality of concrete finite quantum averages passes to their limits
    whenever those limits are known to exist (Proposition 1 supplies existence
    in the intended finite-unitary setting). -/
theorem quantum_limits_equal (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (j : N)
    (lim_j lim_qj : ℝ)
    (hj : Filter.Tendsto
      (fun tau : ℕ => finiteSourceScore
         (evolve (recordedWalk G) · (initial G v)) tau j)
      Filter.atTop (nhds lim_j))
    (hqj : Filter.Tendsto
      (fun tau : ℕ => finiteSourceScore
         (evolve (recordedWalk G) · (initial G v)) tau (q j))
      Filter.atTop (nhds lim_qj)) :
    lim_j = lim_qj := by
  exact QPPR.EC3.symmetry_equal_limiting_scores
    (fun tau => finiteSourceScore (evolve (recordedWalk G) · (initial G v)) tau j)
    (fun tau => finiteSourceScore (evolve (recordedWalk G) · (initial G v)) tau (q j))
    lim_j lim_qj
    (fun tau => quantum_finite_scores_equal G v q hG hv tau j)
    hj hqj

/-- Classical finite averages also agree for concrete G^t v trajectories. -/
theorem classical_finite_scores_equal (G : N → N → ℝ) (v : N → ℝ)
    (q : Equiv.Perm N)
    (hG : ∀ j k, G (q j) (q k) = G j k)
    (hv : ∀ j, v (q j) = v j) (tau : ℕ) (j : N) :
    ((∑ t ∈ Finset.range tau, evolve (classicalStep G) t v j) / (tau : ℝ)) =
      ((∑ t ∈ Finset.range tau, evolve (classicalStep G) t v (q j)) / (tau : ℝ)) := by
  have hsum :
      (∑ t ∈ Finset.range tau, evolve (classicalStep G) t v j) =
        (∑ t ∈ Finset.range tau, evolve (classicalStep G) t v (q j)) := by
    apply Finset.sum_congr rfl
    intro t _
    exact classical_scores_equal_at_time G v q hG hv t j
  rw [hsum]

end QPPR.Manuscript.ConcreteSymmetry
