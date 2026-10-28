# AGENTS.md — Neural Network Strength & Volume Optimization Engine

## 1. Architecture & Repository Structure
This project is a decoupled full-stack monorepo:
- `/client`: Next.js (TypeScript, Tailwind CSS, Recoil, Prisma ORM). Handles UI and state management. Currently the browser calls the inference server directly at `http://localhost:8000/predict`; PostgreSQL persistence and a Next.js proxy route are planned (schema + seed exist but are not yet wired into the app).
- `/server`: Python 3.10+ (FastAPI, Uvicorn, Pydantic, Scikit-Learn, Pandas, NumPy, Joblib). Handles dataset preprocessing, synthetic fatigue generation, model training, `.pkl` serialization, and asynchronous inference endpoints.
- `/server/data`: Raw OpenPowerlifting CSV subsets and procedurally generated time-series fatigue logs.
- `/server/models`: Serialized Joblib artifacts (`mlp_regressor.pkl`, `mlp_classifier.pkl`, `scaler.pkl`).

## 2. Domain & Terminology Rules
- **NO "AI" Buzzwords:** Never label components, variables, or UI text as "AI". Use `inference`, `mlp`, `neural_network`, or `function_approximator`.
- **Function Approximation Framing:** The regression model maps current input vectors (`bodyweight_kg`, `age`, `sex_binary`, `bench_1rm_baseline`, `session_volume_kg`, `mean_rpe`, `fatigue_drop_off` — % rep loss from set 1 to the final set) to a continuous output (`prescribed_load_kg`: the load for a top set of 5 reps @ RPE 8 next session). The classifier maps the same vector to `volume_class_id` 0/1/2 (Under-Stimulated / Optimal Hypertrophy / Excessive Fatigue / Junk Volume).
- **Honest Metrics:** The target scales with 1RM, so report error as % of 1RM and R² on load/1RM, alongside the 1RM×k, linear, and feature-only-rule baselines — not absolute R².

## 3. Coding Standards
### Frontend (`/client`)
- **TypeScript:** Strict mode enabled. No `any` types. Match frontend interfaces exactly to backend Pydantic schemas.
- **State Management:** Use Recoil atoms and selectors for active workout logging (adding/removing sets, updating reps/weight/RPE dynamically).
- **Database:** Use Prisma ORM with PostgreSQL. Always maintain a working `prisma/seed.ts` that seeds 30 days of realistic workout history for the demo user.

### Backend (`/server`)
- **Validation:** Every FastAPI endpoint must use explicit `Pydantic` `BaseModel` schemas for request bodies and responses.
- **Concurrency:** Use `async def` for route handlers where appropriate and load `.pkl` models once at startup via FastAPI `lifespan` context managers—never re-load or re-train models inside a request cycle.
- **ML Pipeline:** Always export the fitted `StandardScaler` alongside the `MLPRegressor` and `MLPClassifier` via `joblib.dump()` so incoming JSON payloads are scaled identically to the training data.

## 4. Common Commands
### Backend (`cd server`)
- Create venv (first time): `python -m venv venv`
- Activate venv (Windows PowerShell): `.\venv\Scripts\Activate.ps1` (if blocked: `Set-ExecutionPolicy -Scope Process Bypass`)
- Activate venv (Git Bash / MINGW64): `source venv/Scripts/activate`
- Install deps: `pip install -r requirements.txt`
- Download + filter OpenPowerlifting (first time; ~170 MB download, cached in the temp dir): `python prepare_data.py`
  - Without this step `train_pipeline.py` falls back to a synthetic baseline and says so.
- Run simulation, training, evaluation & serialization (~30 s; writes `models/` and `reports/`): `python train_pipeline.py`
- Start FastAPI server: `uvicorn main:app --reload --port 8000`

### Frontend (`cd client`)
- Install deps: `npm install`
- Type-check: `npm run typecheck`
- Start Next.js dev server: `npm run dev`
- Database (not yet used by the app; requires a running PostgreSQL matching `DATABASE_URL` in `client/.env`):
  - Push Prisma schema: `npx prisma db push`
  - Seed 30-day demo data: `npx prisma db seed`