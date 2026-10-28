"use client";

import { useEffect, useState } from "react";
import { useRecoilState, useRecoilValue } from "recoil";
import {
  type NumericSetField,
  type PredictResponse,
  PROFILE_FIELDS,
  SET_FIELDS,
  kgToLb,
  payloadState,
  profileState,
  sessionFeaturesState,
  setsState,
  validationErrorsState,
} from "./state";

const INFERENCE_URL = "http://localhost:8000/predict";
const INPUT_CLASS = "w-full rounded border border-slate-700 bg-slate-900 px-2 py-1.5";
const CARD_CLASS = "rounded-lg border border-slate-800 bg-slate-900/60 p-4";

// A cleared number input becomes NaN (not 0) so validation can flag it instead of sending a wrong value.
const parseNumber = (raw: string): number => (raw.trim() === "" ? Number.NaN : Number(raw));
const displayNumber = (value: number): number | string => (Number.isNaN(value) ? "" : value);
const formatFeature = (value: number, suffix = ""): string =>
  Number.isFinite(value) ? `${value.toLocaleString("en-US")}${suffix}` : "—";

function isPredictResponse(value: unknown): value is PredictResponse {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.prescribed_load_kg === "number" &&
    typeof v.volume_class_id === "number" &&
    typeof v.volume_class_label === "string" &&
    typeof v.inference_time_ms === "number"
  );
}

// FastAPI 422 bodies look like {"detail": [{"loc": [..., "field"], "msg": "..."}]}.
function describeError(status: number, body: unknown): string {
  if (typeof body === "object" && body !== null && Array.isArray((body as { detail?: unknown }).detail)) {
    const details = (body as { detail: { loc?: unknown[]; msg?: string }[] }).detail;
    return details.map((d) => `${String(d.loc?.[d.loc.length - 1] ?? "input")}: ${d.msg ?? "invalid"}`).join("; ");
  }
  return `Inference service returned ${status}`;
}

function ProfileCard() {
  const [profile, setProfile] = useRecoilState(profileState);
  return (
    <section className={CARD_CLASS}>
      <h2 className="mb-3 font-medium">Lifter profile</h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {PROFILE_FIELDS.map(({ key, label, step, min, max }) => (
          <label key={key} className="text-sm text-slate-300">
            {label}
            <input
              type="number"
              step={step}
              min={min}
              max={max}
              value={displayNumber(profile[key])}
              onChange={(e) => setProfile({ ...profile, [key]: parseNumber(e.target.value) })}
              className={`mt-1 ${INPUT_CLASS}`}
            />
          </label>
        ))}
        <label className="text-sm text-slate-300">
          Sex
          <select
            value={profile.sex_binary}
            onChange={(e) => setProfile({ ...profile, sex_binary: Number(e.target.value) === 1 ? 1 : 0 })}
            className={`mt-1 ${INPUT_CLASS}`}
          >
            <option value={1}>Male</option>
            <option value={0}>Female</option>
          </select>
        </label>
      </div>
    </section>
  );
}

function SetLogger() {
  const [sets, setSets] = useRecoilState(setsState);

  const updateSet = (id: number, key: NumericSetField, value: number) =>
    setSets(sets.map((s) => (s.id === id ? { ...s, [key]: value } : s)));

  const addSet = () => {
    const last = sets[sets.length - 1];
    const nextId = sets.reduce((max, s) => Math.max(max, s.id), 0) + 1;
    setSets([...sets, { ...last, id: nextId }]);
  };

  return (
    <section className={CARD_CLASS}>
      <h2 className="mb-3 font-medium">Bench press sets</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-slate-400">
            <tr>
              <th className="pb-2 pr-2">Set</th>
              {SET_FIELDS.map((f) => (
                <th key={f.key} className="pb-2 pr-2">{f.label}</th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {sets.map((s, i) => (
              <tr key={s.id}>
                <td className="py-1 pr-2 text-slate-400">{i + 1}</td>
                {SET_FIELDS.map(({ key, step, min, max }) => (
                  <td key={key} className="py-1 pr-2">
                    <input
                      type="number"
                      step={step}
                      min={min}
                      max={max}
                      value={displayNumber(s[key])}
                      onChange={(e) => updateSet(s.id, key, parseNumber(e.target.value))}
                      className={INPUT_CLASS}
                    />
                  </td>
                ))}
                <td className="py-1">
                  <button
                    type="button"
                    onClick={() => setSets(sets.filter((x) => x.id !== s.id))}
                    disabled={sets.length <= 1}
                    className="rounded px-2 py-1 text-slate-400 hover:bg-slate-800 hover:text-red-400 disabled:opacity-30"
                    aria-label={`Remove set ${i + 1}`}
                  >
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        type="button"
        onClick={addSet}
        className="mt-3 rounded border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800"
      >
        + Add set
      </button>
    </section>
  );
}

function DerivedFeatures() {
  const f = useRecoilValue(sessionFeaturesState);
  const items = [
    { label: "Session volume", value: formatFeature(f.session_volume_lb, " lb") },
    { label: "Mean RPE", value: Number.isFinite(f.mean_rpe) ? f.mean_rpe.toFixed(2) : "—" },
    { label: "Rep drop-off (set 1 → last)", value: formatFeature(f.fatigue_drop_off, "%") },
  ];
  return (
    <section className={CARD_CLASS}>
      <h2 className="mb-3 font-medium">Derived input features</h2>
      <dl className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {items.map((it) => (
          <div key={it.label}>
            <dt className="text-xs uppercase text-slate-400">{it.label}</dt>
            <dd className="text-lg font-semibold">{it.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export default function Home() {
  const payload = useRecoilValue(payloadState);
  const validationErrors = useRecoilValue(validationErrorsState);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Any input change invalidates the previous prescription.
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [payload]);

  async function runInference() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(INFERENCE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body: unknown = await res.json().catch(() => null);
      if (!res.ok) throw new Error(describeError(res.status, body));
      if (!isPredictResponse(body)) throw new Error("Unexpected response from inference service");
      setResult(body);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl space-y-4 px-4 py-8">
      <header>
        <h1 className="text-2xl font-semibold">Strength &amp; Volume Optimization Engine</h1>
        <p className="mt-1 text-sm text-slate-400">
          Log a bench session. The MLP function approximator maps your current state to a prescribed load for the
          next session, and the MLP classifier scores the session.
        </p>
      </header>

      <ProfileCard />
      <SetLogger />
      <DerivedFeatures />

      {validationErrors.length > 0 && (
        <ul className="list-inside list-disc rounded border border-amber-700 bg-amber-950 p-3 text-sm text-amber-200">
          {validationErrors.map((msg) => (
            <li key={msg}>{msg}</li>
          ))}
        </ul>
      )}

      <button
        type="button"
        onClick={runInference}
        disabled={loading || validationErrors.length > 0}
        className="w-full rounded bg-emerald-600 px-4 py-2 font-medium hover:bg-emerald-500 disabled:opacity-50"
      >
        {loading ? "Running inference..." : "Run inference"}
      </button>

      {error && <p className="rounded border border-red-700 bg-red-950 p-3 text-sm text-red-300">{error}</p>}

      {result && (
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className={CARD_CLASS}>
            <p className="text-xs uppercase text-slate-400">MLPRegressor: next-session target</p>
            <p className="mt-1 text-3xl font-bold">{kgToLb(result.prescribed_load_kg).toFixed(1)} lb</p>
            <p className="text-xs text-slate-400">top set of 5 reps @ RPE 8</p>
          </div>
          <div className={CARD_CLASS}>
            <p className="text-xs uppercase text-slate-400">MLPClassifier: workout score</p>
            <p className="mt-1 text-xl font-bold">{result.volume_class_label}</p>
            <p className="text-xs text-slate-400">class {result.volume_class_id}</p>
          </div>
          <p className="text-xs text-slate-500 sm:col-span-2">Inference time: {result.inference_time_ms} ms</p>
        </section>
      )}
    </main>
  );
}
