import { atom, selector } from "recoil";

export interface LifterProfile {
  bodyweight_kg: number;
  age: number;
  sex_binary: 0 | 1;
  bench_1rm_baseline: number;
}

export interface WorkoutSet {
  id: number;
  weightKg: number;
  reps: number;
  rpe: number;
}

export interface SessionFeatures {
  session_volume_kg: number;
  mean_rpe: number;
  fatigue_drop_off: number;
}

// Mirrors WorkoutPayload in server/main.py
export type WorkoutPayload = LifterProfile & SessionFeatures;

// Mirrors PredictResponse in server/main.py
export interface PredictResponse {
  prescribed_load_kg: number;
  volume_class_id: number;
  volume_class_label: string;
  inference_time_ms: number;
}

export type NumericProfileField = Exclude<keyof LifterProfile, "sex_binary">;
export type NumericSetField = Exclude<keyof WorkoutSet, "id">;

interface FieldSpec<K> {
  key: K;
  label: string;
  step: number;
  min: number;
  max: number;
}

// Ranges mirror the Pydantic bounds in server/main.py (the model's training domain).
export const PROFILE_FIELDS: FieldSpec<NumericProfileField>[] = [
  { key: "bodyweight_kg", label: "Bodyweight (kg)", step: 0.5, min: 40, max: 200 },
  { key: "age", label: "Age", step: 1, min: 16, max: 80 },
  { key: "bench_1rm_baseline", label: "Bench 1RM baseline (kg)", step: 2.5, min: 20, max: 330 },
];

export const SET_FIELDS: FieldSpec<NumericSetField>[] = [
  { key: "weightKg", label: "Weight (kg)", step: 2.5, min: 1, max: 330 },
  { key: "reps", label: "Reps", step: 1, min: 1, max: 20 },
  { key: "rpe", label: "RPE", step: 0.5, min: 6, max: 10 },
];

export const MAX_SESSION_VOLUME_KG = 10000;
export const MAX_RELATIVE_VOLUME = 50;

export const profileState = atom<LifterProfile>({
  key: "profile",
  default: { bodyweight_kg: 85, age: 27, sex_binary: 1, bench_1rm_baseline: 120 },
});

export const setsState = atom<WorkoutSet[]>({
  key: "sets",
  default: [
    { id: 1, weightKg: 80, reps: 8, rpe: 7.5 },
    { id: 2, weightKg: 80, reps: 8, rpe: 8 },
    { id: 3, weightKg: 80, reps: 7, rpe: 8.5 },
    { id: 4, weightKg: 80, reps: 6, rpe: 9 },
  ],
});

// Derives the three session-level input features from the logged sets (same formulas as server/simulate.py).
// A cleared input is NaN, which propagates so the UI can show "—" instead of a silently wrong number.
export const sessionFeaturesState = selector<SessionFeatures>({
  key: "sessionFeatures",
  get: ({ get }) => {
    const sets = get(setsState);
    const volume = sets.reduce((sum, s) => sum + s.weightKg * s.reps, 0);
    const meanRpe = sets.reduce((sum, s) => sum + s.rpe, 0) / sets.length;
    const first = sets[0].reps;
    const last = sets[sets.length - 1].reps;
    const dropOff = Math.max(0, ((first - last) / first) * 100);
    return {
      session_volume_kg: volume,
      mean_rpe: Math.round(meanRpe * 100) / 100,
      fatigue_drop_off: Math.round(dropOff * 10) / 10,
    };
  },
});

export const payloadState = selector<WorkoutPayload>({
  key: "payload",
  get: ({ get }) => ({ ...get(profileState), ...get(sessionFeaturesState) }),
});

const outOfRange = (value: number, min: number, max: number): boolean =>
  Number.isNaN(value) || value < min || value > max;

// Client-side copy of the server's validation, so invalid input is explained before any request is sent.
export const validationErrorsState = selector<string[]>({
  key: "validationErrors",
  get: ({ get }) => {
    const profile = get(profileState);
    const sets = get(setsState);
    const errors: string[] = [];
    for (const f of PROFILE_FIELDS) {
      if (outOfRange(profile[f.key], f.min, f.max)) errors.push(`${f.label} must be between ${f.min} and ${f.max}.`);
    }
    sets.forEach((s, i) => {
      for (const f of SET_FIELDS) {
        if (outOfRange(s[f.key], f.min, f.max)) errors.push(`Set ${i + 1}: ${f.label} must be between ${f.min} and ${f.max}.`);
      }
    });
    if (errors.length > 0) return errors;

    const { session_volume_kg } = get(sessionFeaturesState);
    if (session_volume_kg > MAX_SESSION_VOLUME_KG) {
      errors.push(`Session volume must be at most ${MAX_SESSION_VOLUME_KG.toLocaleString("en-US")} kg.`);
    } else if (session_volume_kg / profile.bench_1rm_baseline > MAX_RELATIVE_VOLUME) {
      errors.push(`Session volume must be at most ${MAX_RELATIVE_VOLUME}× your bench 1RM.`);
    }
    return errors;
  },
});
