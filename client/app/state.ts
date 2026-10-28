import { atom, selector } from "recoil";

// The UI works in pounds; the model was trained in kilograms (OpenPowerlifting), so
// payloadState converts to kg before the request and the page converts the result back.
export const KG_PER_LB = 0.45359237;
export const kgToLb = (kg: number): number => kg / KG_PER_LB;

export interface LifterProfile {
  bodyweight_lb: number;
  age: number;
  sex_binary: 0 | 1;
  bench_1rm_lb: number;
}

export interface WorkoutSet {
  id: number;
  weightLb: number;
  reps: number;
  rpe: number;
}

export interface SessionFeatures {
  session_volume_lb: number;
  mean_rpe: number;
  fatigue_drop_off: number;
}

// Mirrors WorkoutPayload in server/main.py (kilograms)
export interface WorkoutPayload {
  bodyweight_kg: number;
  age: number;
  sex_binary: 0 | 1;
  bench_1rm_baseline: number;
  session_volume_kg: number;
  mean_rpe: number;
  fatigue_drop_off: number;
}

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

// Ranges mirror the Pydantic bounds in server/main.py (the model's training domain), converted to lb.
export const PROFILE_FIELDS: FieldSpec<NumericProfileField>[] = [
  { key: "bodyweight_lb", label: "Bodyweight (lb)", step: 1, min: 89, max: 440 },
  { key: "age", label: "Age", step: 1, min: 16, max: 80 },
  { key: "bench_1rm_lb", label: "Bench 1RM baseline (lb)", step: 5, min: 45, max: 725 },
];

export const SET_FIELDS: FieldSpec<NumericSetField>[] = [
  { key: "weightLb", label: "Weight (lb)", step: 5, min: 2.5, max: 725 },
  { key: "reps", label: "Reps", step: 1, min: 1, max: 20 },
  { key: "rpe", label: "RPE", step: 0.5, min: 6, max: 10 },
];

export const MAX_SESSION_VOLUME_LB = 22000; // server cap is 10,000 kg
export const MAX_RELATIVE_VOLUME = 50;

export const profileState = atom<LifterProfile>({
  key: "profile",
  default: { bodyweight_lb: 185, age: 27, sex_binary: 1, bench_1rm_lb: 265 },
});

export const setsState = atom<WorkoutSet[]>({
  key: "sets",
  default: [
    { id: 1, weightLb: 175, reps: 8, rpe: 7.5 },
    { id: 2, weightLb: 175, reps: 8, rpe: 8 },
    { id: 3, weightLb: 175, reps: 7, rpe: 8.5 },
    { id: 4, weightLb: 175, reps: 6, rpe: 9 },
  ],
});

// Derives the three session-level input features from the logged sets (same formulas as server/simulate.py).
// A cleared input is NaN, which propagates so the UI can show "—" instead of a silently wrong number.
export const sessionFeaturesState = selector<SessionFeatures>({
  key: "sessionFeatures",
  get: ({ get }) => {
    const sets = get(setsState);
    const volume = sets.reduce((sum, s) => sum + s.weightLb * s.reps, 0);
    const meanRpe = sets.reduce((sum, s) => sum + s.rpe, 0) / sets.length;
    const first = sets[0].reps;
    const last = sets[sets.length - 1].reps;
    const dropOff = Math.max(0, ((first - last) / first) * 100);
    return {
      session_volume_lb: volume,
      mean_rpe: Math.round(meanRpe * 100) / 100,
      fatigue_drop_off: Math.round(dropOff * 10) / 10,
    };
  },
});

export const payloadState = selector<WorkoutPayload>({
  key: "payload",
  get: ({ get }) => {
    const p = get(profileState);
    const f = get(sessionFeaturesState);
    return {
      bodyweight_kg: p.bodyweight_lb * KG_PER_LB,
      age: p.age,
      sex_binary: p.sex_binary,
      bench_1rm_baseline: p.bench_1rm_lb * KG_PER_LB,
      session_volume_kg: f.session_volume_lb * KG_PER_LB,
      mean_rpe: f.mean_rpe,
      fatigue_drop_off: f.fatigue_drop_off,
    };
  },
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

    const { session_volume_lb } = get(sessionFeaturesState);
    if (session_volume_lb > MAX_SESSION_VOLUME_LB) {
      errors.push(`Session volume must be at most ${MAX_SESSION_VOLUME_LB.toLocaleString("en-US")} lb.`);
    } else if (session_volume_lb / profile.bench_1rm_lb > MAX_RELATIVE_VOLUME) {
      errors.push(`Session volume must be at most ${MAX_RELATIVE_VOLUME}× your bench 1RM.`);
    }
    return errors;
  },
});
