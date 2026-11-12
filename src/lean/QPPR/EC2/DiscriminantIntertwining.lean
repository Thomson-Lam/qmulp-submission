import QPPR.EC2.Intertwining

/-! Auxiliary results retained for compatibility. The complete concrete proofs
are in QPPR.Manuscript.CompletePropositions and its imported modules.
See docs/PROPOSITION_STATUS.md for the final theorem map. -/

namespace QPPR.EC2

/-- EC.4: reflection-swap coefficient update. -/
def coefficientWalk
    {V : Type*} [AddCommGroup V] [Module ℂ V]
    (D : V →ₗ[ℂ] V) (pair : V × V) : V × V :=
  (-pair.2, pair.1 + (2 : ℂ) • D pair.2)

/-- Apply the class-uniform embedding to both coefficient vectors. -/
def embedPair
    {Small Full : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (B : Small →ₗ[ℂ] Full) (pair : Small × Small) : Full × Full :=
  (B pair.1, B pair.2)

/-- Under DB = B Dbar, the coefficient evolution commutes with embedding. -/
theorem coefficientWalk_intertwining
    {Small Full : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (D : Full →ₗ[ℂ] Full)
    (Dbar : Small →ₗ[ℂ] Small)
    (B : Small →ₗ[ℂ] Full)
    (hD : ∀ x, D (B x) = B (Dbar x))
    (p : Small × Small) :
    embedPair B (coefficientWalk Dbar p) =
      coefficientWalk D (embedPair B p) := by
  apply Prod.ext
  · change B (-p.2) = -(B p.2)
    exact map_neg B p.2
  · change B (p.1 + (2 : ℂ) • Dbar p.2) =
      B p.1 + (2 : ℂ) • D (B p.2)
    calc
      B (p.1 + (2 : ℂ) • Dbar p.2) =
          B p.1 + (2 : ℂ) • B (Dbar p.2) := by simp
      _ = B p.1 + (2 : ℂ) • D (B p.2) := by rw [hD p.2]

/-- The recorded step of QPPR is two coefficient-walk operations (U=W²). -/
def recordedCoefficientStep
    {V : Type*} [AddCommGroup V] [Module ℂ V]
    (D : V →ₗ[ℂ] V) (p : V × V) : V × V :=
  coefficientWalk D (coefficientWalk D p)

/-- Under DB=B Dbar, the recorded steps of full and quotient operators agree
    after embedding, not merely their single reflection-swap steps. -/
theorem recordedCoefficientStep_intertwining
    {Small Full : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (D : Full →ₗ[ℂ] Full) (Dbar : Small →ₗ[ℂ] Small)
    (B : Small →ₗ[ℂ] Full)
    (hD : ∀ x, D (B x) = B (Dbar x))
    (p : Small × Small) :
    embedPair B (recordedCoefficientStep Dbar p) =
      recordedCoefficientStep D (embedPair B p) := by
  exact intertwining_recorded_steps
    (coefficientWalk Dbar) (coefficientWalk D) (embedPair B)
    (coefficientWalk_intertwining D Dbar B hD) p

/-- Every recorded horizon of the embedded quotient coefficient trajectory
    agrees with the full coefficient trajectory (at the coefficient level). -/
theorem all_recorded_coefficient_times
    {Small Full : Type*}
    [AddCommGroup Small] [Module ℂ Small]
    [AddCommGroup Full] [Module ℂ Full]
    (D : Full →ₗ[ℂ] Full) (Dbar : Small →ₗ[ℂ] Small)
    (B : Small →ₗ[ℂ] Full)
    (hD : ∀ x, D (B x) = B (Dbar x))
    (t : ℕ) (p : Small × Small) :
    embedPair B (QPPR.Core.evolve (recordedCoefficientStep Dbar) t p) =
      QPPR.Core.evolve (recordedCoefficientStep D) t (embedPair B p) := by
  exact intertwining_iterates
    (recordedCoefficientStep Dbar) (recordedCoefficientStep D)
    (embedPair B) (recordedCoefficientStep_intertwining D Dbar B hD)
    t p

end QPPR.EC2
