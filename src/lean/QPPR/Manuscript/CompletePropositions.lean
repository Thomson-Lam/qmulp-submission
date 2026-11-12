import QPPR.Manuscript.ConcreteLimit
import QPPR.Manuscript.PhysicalQuotient
import QPPR.Manuscript.ClassicalLimit
import QPPR.Manuscript.DiscriminantComputation
import QPPR.Manuscript.TransitiveBlocks

namespace QPPR.Manuscript.CompletePropositions
open QPPR.Core QPPR.Manuscript.ConcreteSymmetry QPPR.Manuscript.PhysicalWalk
open QPPR.Manuscript.PhysicalQuotient QPPR.Manuscript.ConcreteLimit
variable {N R : Type*} [Fintype N] [Fintype R] [DecidableEq N] [DecidableEq R]

noncomputable def discriminantTrajectory (G : N → N → ℝ) (v : N → ℝ) (t : ℕ) : State N :=
  pairState G (QPPR.EC2.embedPair (Matrix.mulVecLin
    (DiscriminantComputation.Q G : Matrix N N ℂ))
      (evolve (QPPR.EC2.recordedCoefficientStep
        (Matrix.mulVecLin (DiscriminantComputation.diagonal G))) t
        (DiscriminantComputation.initialCoordinates G v,0)))

theorem proposition1_discriminant [Nonempty N]
    (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ) :
    proposition1_score_target (discriminantTrajectory G v) (phi G v) := by
  have he : discriminantTrajectory G v = (fun t => evolve (recordedWalk G) t (initial G v)) := by
    funext t
    exact (DiscriminantComputation.evolution_from_discriminant_eigendecomposition G hG v t).symm
  rw [he]
  exact proposition1_concrete G hG v

theorem quotient_stochastic (c : N → R) (hc : Function.Surjective c)
    (G : N → N → ℝ) (Gb : R → R → ℝ) (hG : Stochastic G) (hL : Lumpable c G Gb) :
    Stochastic Gb := by
  constructor
  · intro a b
    obtain ⟨k,hk⟩ := hc b
    rw [← hk, ← hL a k]
    exact Finset.sum_nonneg (fun j _ => hG.1 j k)
  · intro b
    obtain ⟨k,hk⟩ := hc b
    rw [← hk]
    calc
      _ = ∑ a, ∑ j ∈ fiber c a, G j k :=
        Finset.sum_congr rfl (fun a _ => (hL a k).symm)
      _ = 1 := by rw [sum_by_blocks, hG.2 k]

theorem finite_aggregation {N R : Type*} [Fintype N] [Fintype R] [DecidableEq R]
    (c : N → R) (full : ℕ → N → ℝ) (small : ℕ → R → ℝ)
    (heq : ∀ t a, aggregate c (full t) a = small t a) (n : ℕ) (a : R) :
    aggregate c (fun j => (∑ t ∈ Finset.range n, full t j) / (n : ℝ)) a =
      (∑ t ∈ Finset.range n, small t a) / (n : ℝ) := by
  unfold aggregate at *
  rw [← Finset.sum_div, Finset.sum_comm]
  simp_rw [heq]

theorem proposition2_finite
    (c : N → R) (hc : Function.Surjective c) (G : N → N → ℝ) (Gb : R → R → ℝ) (v : N → ℝ)
    (hG : Stochastic G) (hL : Lumpable c G Gb)
    (hD : ∀ z, D G (embedding c (weight c) z) = embedding c (weight c) (D Gb z))
    (hconst : ∀ i j, c i = c j → v i = v j) (n : ℕ) (a : R) :
    aggregate c (finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n) a =
      finiteSourceScore (evolve (recordedWalk Gb) · (initial Gb (aggregate c v))) n a ∧
    aggregate c (fun j => (∑ t ∈ Finset.range n,
      destinationProbability (evolve (recordedWalk G) t (initial G v)) j) / (n : ℝ)) a =
      (∑ t ∈ Finset.range n,
        destinationProbability (evolve (recordedWalk Gb) t (initial Gb (aggregate c v))) a) / (n : ℝ) := by
  have ht := PhysicalQuotient.proposition2_concrete_time c hc G Gb v hG
    (quotient_stochastic c hc G Gb hG hL) hL hD hconst
  constructor
  · exact finite_aggregation c _ _ (fun t a => (ht t a).1) n a
  · exact finite_aggregation c _ _ (fun t a => (ht t a).2) n a

theorem proposition2_limiting [Nonempty N] [Nonempty R]
    (c : N → R) (hc : Function.Surjective c) (G : N → N → ℝ) (Gb : R → R → ℝ) (v : N → ℝ)
    (hG : Stochastic G) (hL : Lumpable c G Gb)
    (hD : ∀ z, D G (embedding c (weight c) z) = embedding c (weight c) (D Gb z))
    (hconst : ∀ i j, c i = c j → v i = v j) (a : R) :
    aggregate c (limitScore G v) a = limitScore Gb (aggregate c v) a ∧
    aggregate c (limitDestinationScore G v) a = limitDestinationScore Gb (aggregate c v) a := by
  have hGb := quotient_stochastic c hc G Gb hG hL
  have hs : Filter.Tendsto
      (fun n => aggregate c (finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n) a)
      Filter.atTop (nhds (aggregate c (limitScore G v) a)) := by
    exact tendsto_finsetSum (fiber c a) (fun j _ => proposition1_convergence G hG v j)
  have hd : Filter.Tendsto
      (fun n => aggregate c (finiteDestinationScore G v n) a)
      Filter.atTop (nhds (aggregate c (limitDestinationScore G v) a)) := by
    exact tendsto_finsetSum (fiber c a) (fun j _ => destination_convergence G hG v j)
  constructor
  · have he : (fun n => aggregate c (finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n) a) =
        (fun n => finiteSourceScore (evolve (recordedWalk Gb) · (initial Gb (aggregate c v))) n a) :=
      funext (fun n => (proposition2_finite c hc G Gb v hG hL hD hconst n a).1)
    rw [he] at hs
    exact tendsto_nhds_unique hs (proposition1_convergence Gb hGb (aggregate c v) a)
  · have he : (fun n => aggregate c (finiteDestinationScore G v n) a) =
        (fun n => finiteDestinationScore Gb (aggregate c v) n a) := by
      funext n
      exact (proposition2_finite c hc G Gb v hG hL hD hconst n a).2
    rw [he] at hd
    exact tendsto_nhds_unique hd (destination_convergence Gb hGb (aggregate c v) a)

theorem proposition3_quantum_limit [Nonempty N]
    (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ) (q : Equiv.Perm N)
    (hex : ∀ i j, G (q i) (q j) = G i j) (hv : ∀ i, v (q i) = v i) (j : N) :
    limitScore G v j = limitScore G v (q j) := by
  exact quantum_limits_equal G v q hex hv j _ _
    (proposition1_convergence G hG v j) (proposition1_convergence G hG v (q j))

/-- Proposition 3 including existence of BOTH classical and quantum Cesaro limits.
    The assumptions concern G and v only. No existence of any limit is assumed. -/
theorem proposition3_complete [Nonempty N]
    (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ) (q : Equiv.Perm N)
    (hex : ∀ i j, G (q i) (q j) = G i j) (hv : ∀ i, v (q i) = v i) :
    (∀ j, proposition3_time_target (evolve (recordedWalk G) · (initial G v))
      (evolve (classicalStep G) · v) j (q j)) ∧
    (∀ n j, finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n j =
      finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n (q j)) ∧
    (∀ j, Filter.Tendsto (fun n => finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n j)
      Filter.atTop (nhds (limitScore G v j))) ∧
    (∀ j, limitScore G v j = limitScore G v (q j)) ∧
    (∃ p : N → ℝ,
      (∀ j, Filter.Tendsto (fun n => ClassicalLimit.finiteScore G v n j) Filter.atTop (nhds (p j))) ∧
      (∀ n j, ClassicalLimit.finiteScore G v n j = ClassicalLimit.finiteScore G v n (q j)) ∧
      (∀ j, p j = p (q j))) := by
  refine ⟨proposition3_concrete_time G v q hex hv,
    quantum_finite_scores_equal G v q hex hv,
    proposition1_convergence G hG v,
    proposition3_quantum_limit G hG v q hex hv, ?_⟩
  obtain ⟨p,hp⟩ := ClassicalLimit.classical_cesaro_exists G hG v
  have hf (n : ℕ) (j : N) : ClassicalLimit.finiteScore G v n j = ClassicalLimit.finiteScore G v n (q j) :=
    classical_finite_scores_equal G v q hex hv n j
  refine ⟨p,hp,hf,fun j => ?_⟩
  have he : (fun n => ClassicalLimit.finiteScore G v n j) =
      (fun n => ClassicalLimit.finiteScore G v n (q j)) := funext (fun n => hf n j)
  exact tendsto_nhds_unique (hp j) (he.symm ▸ hp (q j))

def ExchangeTransitive (c : N → R) (G : N → N → ℝ) (v : N → ℝ) : Prop :=
  ∀ i j, c i = c j → ∃ q : Equiv.Perm N, q i = j ∧
    (∀ k m, G (q k) (q m) = G k m) ∧ (∀ k, v (q k) = v k)

theorem member_recovery (c : N → R) (G : N → N → ℝ) (v p : N → ℝ)
    (htrans : ExchangeTransitive c G v)
    (hinv : ∀ q : Equiv.Perm N, (∀ k m, G (q k) (q m) = G k m) →
      (∀ k, v (q k) = v k) → ∀ k, p (q k) = p k) (i : N) :
    p i = aggregate c p (c i) / ((fiber c (c i)).card : ℝ) := by
  apply member_score_eq_block_average p (fiber c (c i)) ?_ i (by simp [fiber])
  intro j hj k hk
  have he : c j = c k := (Finset.mem_filter.mp hj).2.trans (Finset.mem_filter.mp hk).2.symm
  obtain ⟨q,hq,hG,hv⟩ := htrans j k he
  exact ⟨q,hq,hinv q hG hv⟩

theorem proposition2_member_time (c : N → R) (G : N → N → ℝ) (v : N → ℝ)
    (htrans : ExchangeTransitive c G v) (t : ℕ) (i : N) :
    sourceProbability (evolve (recordedWalk G) t (initial G v)) i =
      aggregate c (sourceProbability (evolve (recordedWalk G) t (initial G v))) (c i) /
        ((fiber c (c i)).card : ℝ) := by
  exact member_recovery c G v _ htrans
    (fun q hG hv k => (quantum_scores_equal_at_time G v q hG hv t k).symm) i

theorem proposition2_member_finite (c : N → R) (G : N → N → ℝ) (v : N → ℝ)
    (htrans : ExchangeTransitive c G v) (n : ℕ) (i : N) :
    finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n i =
      aggregate c (finiteSourceScore (evolve (recordedWalk G) · (initial G v)) n) (c i) /
        ((fiber c (c i)).card : ℝ) := by
  exact member_recovery c G v _ htrans
    (fun q hG hv k => (quantum_finite_scores_equal G v q hG hv n k).symm) i

theorem proposition2_member_limiting [Nonempty N]
    (c : N → R) (G : N → N → ℝ) (hG : Stochastic G) (v : N → ℝ)
    (htrans : ExchangeTransitive c G v) (i : N) :
    limitScore G v i = aggregate c (limitScore G v) (c i) / ((fiber c (c i)).card : ℝ) := by
  exact member_recovery c G v _ htrans
    (fun q hex hv k => (proposition3_quantum_limit G hG v q hex hv k).symm) i

end QPPR.Manuscript.CompletePropositions
