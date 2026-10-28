"""Procedural 30-day bench-press training logs with accumulated physiological fatigue.

Each lifter (baseline from OpenPowerlifting) trains Mon/Wed/Fri for 30 days. A hidden
fatigue state F rises with each session's hard volume and decays exponentially
between sessions; it lowers the lifter's working capacity, which shows up in the
observable features (reps achieved per set, RPE, rep drop-off).

Targets are computed from the hidden state the network never sees directly:
  - next_target_load_kg: the load for a top set of 5 reps @ RPE 8 at the NEXT session,
    given the fatigue carried into that session.
  - volume_class: 0 Under-Stimulated, 1 Optimal Hypertrophy, 2 Excessive Fatigue / Junk Volume.
"""
import numpy as np
import pandas as pd

DAYS = 30
TRAINING_WEEKDAYS = (0, 2, 4)  # Mon / Wed / Fri
FATIGUE_DECAY_PER_DAY = 0.80
FATIGUE_CAPACITY_COST = 0.15  # fractional capacity lost per unit of fatigue
REPS_IN_RESERVE_TARGET = 2  # RPE 8
TARGET_REPS = 5
PROGRESSION_PER_SESSION = 0.002  # small training adaptation per completed session


def reps_to_failure(capacity_kg: float, load_kg: float) -> float:
    """Maximum reps at a load, from the inverse of the Epley relation (simulator ground truth)."""
    return max(0.0, 30.0 * (capacity_kg / load_kg - 1.0))


def load_for_reps(capacity_kg: float, reps: float) -> float:
    return capacity_kg / (1.0 + reps / 30.0)


def score_session(mean_rpe: float, drop_off: float, rel_volume: float, hidden_fatigue: float) -> int:
    """Label rule. Pass hidden_fatigue=0 to get the feature-only baseline used in evaluation."""
    if drop_off > 35 or mean_rpe >= 9.6 or hidden_fatigue > 1.4:
        return 2
    if mean_rpe < 7.2 or rel_volume < 6:
        return 0
    return 1


def round_to_plate(kg: float) -> float:
    return round(kg / 2.5) * 2.5


def simulate_lifter(lifter: pd.Series, lifter_id: int, rng: np.random.Generator) -> list[dict]:
    one_rm = float(lifter["bench_1rm_baseline"])
    training_days = [d for d in range(DAYS + 7) if d % 7 in TRAINING_WEEKDAYS]
    # Lifter "personality": how hard they push and how much volume they do.
    planned_reps = int(rng.integers(5, 11))
    base_sets = int(rng.integers(3, 6))
    aggressiveness = rng.uniform(-0.04, 0.06)
    intensity = rng.uniform(0.62, 0.78) + aggressiveness

    fatigue = rng.uniform(0.0, 0.4)
    rows: list[dict] = []
    sessions = [d for d in training_days if d < DAYS]
    for i, day in enumerate(sessions):
        progression = 1.0 + PROGRESSION_PER_SESSION * i
        capacity = one_rm * (1.0 - FATIGUE_CAPACITY_COST * fatigue) * progression
        load = round_to_plate(one_rm * intensity)
        n_sets = int(np.clip(base_sets + rng.integers(-1, 3), 2, 8))

        reps, rpes = [], []
        set_capacity = capacity
        for _ in range(n_sets):
            max_reps = reps_to_failure(set_capacity, load)
            done = int(np.clip(round(min(planned_reps, max_reps) + rng.normal(0, 0.5)), 1, 20))
            rpe = float(np.clip(10.0 - (max_reps - done) + rng.normal(0, 0.3), 6.0, 10.0))
            reps.append(done)
            rpes.append(rpe)
            set_capacity *= 1.0 - 0.006 * (rpe - 5.0)  # intra-session fatigue

        volume = float(load * sum(reps))
        mean_rpe = float(np.mean(rpes))
        drop_off = max(0.0, 100.0 * (reps[0] - reps[-1]) / reps[0])
        rel_volume = volume / one_rm

        fatigue_after = fatigue + 0.025 * rel_volume * (mean_rpe / 8.0) ** 2
        next_day = training_days[training_days.index(day) + 1]
        fatigue_next = fatigue_after * FATIGUE_DECAY_PER_DAY ** (next_day - day)
        next_capacity = one_rm * (1.0 - FATIGUE_CAPACITY_COST * fatigue_next) * (1.0 + PROGRESSION_PER_SESSION * (i + 1))
        target = load_for_reps(next_capacity, TARGET_REPS + REPS_IN_RESERVE_TARGET)

        volume_class = score_session(mean_rpe, drop_off, rel_volume, fatigue_after)

        rows.append(
            {
                "lifter_id": lifter_id,
                "day": day,
                "bodyweight_kg": float(lifter["bodyweight_kg"]),
                "age": float(lifter["age"]),
                "sex_binary": int(lifter["sex_binary"]),
                "bench_1rm_baseline": one_rm,
                "load_kg": load,
                "sets": n_sets,
                "reps": "-".join(map(str, reps)),
                "session_volume_kg": volume,
                "mean_rpe": mean_rpe,
                "fatigue_drop_off": drop_off,
                "hidden_fatigue": fatigue_after,
                "next_target_load_kg": target + rng.normal(0, 1.0),
                "volume_class": volume_class,
            }
        )

        # Autoregulating lifter: adjust next session's load from how this one felt.
        intensity += 0.02 * (8.0 - mean_rpe) + rng.normal(0, 0.01)
        intensity = float(np.clip(intensity, 0.55, 0.92))
        fatigue = fatigue_next
    return rows


def simulate(lifters: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for lifter_id, (_, lifter) in enumerate(lifters.iterrows()):
        records.extend(simulate_lifter(lifter, lifter_id, rng))
    return pd.DataFrame.from_records(records)
