"""Load OpenPowerlifting baselines, simulate 30-day fatigue logs, train + evaluate both MLPs, serialize artifacts."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from simulate import REPS_IN_RESERVE_TARGET, TARGET_REPS, load_for_reps, score_session, simulate

ROOT = Path(__file__).parent
BASELINE_CSV = ROOT / "data" / "opl_raw_bench.csv"  # produced by prepare_data.py
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SEED = 42
N_LIFTERS = 3000

FEATURES = [
    "bodyweight_kg",
    "age",
    "sex_binary",
    "bench_1rm_baseline",
    "session_volume_kg",
    "mean_rpe",
    "fatigue_drop_off",
]
CLASS_LABELS = ["Under-Stimulated", "Optimal Hypertrophy", "Excessive Fatigue / Junk Volume"]


def load_baseline(rng: np.random.Generator, n: int = N_LIFTERS) -> pd.DataFrame:
    if BASELINE_CSV.exists():
        lifters = pd.read_csv(BASELINE_CSV)
        print(f"OpenPowerlifting: sampling {n:,} of {len(lifters):,} Raw bench lifters")
        return lifters.sample(n=n, random_state=SEED).reset_index(drop=True)
    print(f"{BASELINE_CSV.name} not found (run prepare_data.py); using synthetic baseline")
    sex = rng.integers(0, 2, n)
    bw = np.where(sex == 1, rng.normal(88, 15, n), rng.normal(64, 11, n)).clip(45, 160)
    age = rng.gamma(9.0, 3.0, n).clip(16, 70)
    ratio = np.where(sex == 1, rng.normal(1.25, 0.25, n), rng.normal(0.75, 0.18, n)).clip(0.3, 2.2)
    return pd.DataFrame({"bodyweight_kg": bw, "age": age, "sex_binary": sex, "bench_1rm_baseline": bw * ratio})


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, one_rm: np.ndarray) -> dict[str, float]:
    # The target scales with 1RM, so absolute R² is inflated; relative (load / 1RM) metrics are the honest ones.
    return {
        "mae_kg": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kg": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_pct_1rm": float(100 * np.mean(np.abs(y_true - y_pred) / one_rm)),
        "r2_relative": float(r2_score(y_true / one_rm, y_pred / one_rm)),
    }


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """The simulator's labeling rule applied to observable features only (hidden fatigue withheld)."""
    rel_volume = df["session_volume_kg"] / df["bench_1rm_baseline"]
    return np.array(
        [score_session(r, d, v, 0.0) for r, d, v in zip(df["mean_rpe"], df["fatigue_drop_off"], rel_volume)]
    )


def save_plots(regressor: MLPRegressor, y_true: np.ndarray, y_pred: np.ndarray, classifier, X_te, yc_te) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(regressor.loss_curve_, label="training loss (MSE/2)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(regressor.validation_scores_, color="tab:orange", label="validation R²")
    ax2.set_ylabel("validation R²")
    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(handles=handles, loc="center right")
    ax1.set_title("MLPRegressor training curve")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "regressor_loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("simulator target load (kg)")
    ax.set_ylabel("MLP prescribed load (kg)")
    ax.set_title("Held-out lifters: prescribed vs target")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "prescribed_vs_target.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        classifier, X_te, yc_te, display_labels=["Under", "Optimal", "Excessive"], ax=ax, colorbar=False
    )
    ax.set_title("MLPClassifier confusion matrix (held-out lifters)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    sessions = simulate(load_baseline(rng), seed=SEED)
    sessions.to_csv(ROOT / "data" / "simulated_sessions.csv", index=False)
    print(f"Simulated {len(sessions):,} sessions (30 days x {sessions['lifter_id'].nunique():,} lifters)")

    # Split by lifter so no lifter's sessions appear in both train and test.
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(split.split(sessions, groups=sessions["lifter_id"]))
    train, test = sessions.iloc[tr_idx], sessions.iloc[te_idx]

    scaler = StandardScaler().fit(train[FEATURES])
    X_tr, X_te = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
    yr_tr, yr_te = train["next_target_load_kg"].to_numpy(), test["next_target_load_kg"].to_numpy()
    yc_tr, yc_te = train["volume_class"].to_numpy(), test["volume_class"].to_numpy()

    regressor = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=2000,
        early_stopping=True, n_iter_no_change=25, random_state=SEED,
    ).fit(X_tr, yr_tr)
    classifier = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam", max_iter=2000,
        early_stopping=True, n_iter_no_change=25, random_state=SEED,
    ).fit(X_tr, yc_tr)

    # Regression baselines:
    #   static Epley  - 5 reps @ RPE 8 from the competition 1RM, ignoring fatigue (the classical formula)
    #   1RM x k       - single fitted ratio of target to 1RM (what scale alone explains)
    #   linear        - linear regression on the same 7 scaled features
    rm_tr, rm_te = train["bench_1rm_baseline"].to_numpy(), test["bench_1rm_baseline"].to_numpy()
    epley = np.array([load_for_reps(rm, TARGET_REPS + REPS_IN_RESERVE_TARGET) for rm in rm_te])
    k = float(np.sum(yr_tr * rm_tr) / np.sum(rm_tr**2))
    linear = LinearRegression().fit(X_tr, yr_tr)
    mlp_pred = regressor.predict(X_te)

    cls_pred = classifier.predict(X_te)
    rule_pred = rule_baseline(test)
    majority = np.bincount(yc_tr).argmax()

    metrics = {
        "data": {
            "lifters": int(sessions["lifter_id"].nunique()),
            "sessions": len(sessions),
            "train_sessions": len(train),
            "test_sessions": len(test),
            "class_balance": sessions["volume_class"].value_counts(normalize=True).sort_index().round(3).tolist(),
        },
        "regression": {
            "static_epley": regression_metrics(yr_te, epley, rm_te),
            "1rm_times_k": regression_metrics(yr_te, k * rm_te, rm_te),
            "linear_regression": regression_metrics(yr_te, linear.predict(X_te), rm_te),
            "mlp_regressor": regression_metrics(yr_te, mlp_pred, rm_te),
            # Targets carry N(0, 1 kg) label noise, so no model can beat MAE = sqrt(2/pi) kg.
            "noise_floor_mae_kg": float(np.sqrt(2 / np.pi)),
            "fitted_k": k,
            "mlp_epochs": regressor.n_iter_,
            "mlp_final_training_loss": float(regressor.loss_),
        },
        "classification": {
            "majority_class_accuracy": float(np.mean(yc_te == majority)),
            "feature_rule_accuracy": float(accuracy_score(yc_te, rule_pred)),
            "feature_rule_macro_f1": float(f1_score(yc_te, rule_pred, average="macro")),
            "mlp_accuracy": float(accuracy_score(yc_te, cls_pred)),
            "mlp_macro_f1": float(f1_score(yc_te, cls_pred, average="macro")),
            "mlp_recall_per_class": dict(
                zip(CLASS_LABELS, recall_score(yc_te, cls_pred, average=None).round(3).tolist())
            ),
            "mlp_epochs": classifier.n_iter_,
        },
    }

    r = metrics["regression"]
    print("\nPrescribed next-session load, held-out lifters:")
    print(f"  {'model':<20}{'MAE (kg)':>10}{'MAE (%1RM)':>12}{'R2 (load/1RM)':>15}")
    for name in ("static_epley", "1rm_times_k", "linear_regression", "mlp_regressor"):
        m = r[name]
        print(f"  {name:<20}{m['mae_kg']:>10.2f}{m['mae_pct_1rm']:>11.2f}%{m['r2_relative']:>15.3f}")
    print(f"  {'label-noise floor':<20}{r['noise_floor_mae_kg']:>10.2f}")
    c = metrics["classification"]
    print("\nWorkout score, held-out lifters:")
    print(f"  {'model':<20}{'accuracy':>10}{'macro F1':>10}")
    print(f"  {'majority class':<20}{c['majority_class_accuracy']:>10.3f}")
    print(f"  {'feature-only rule':<20}{c['feature_rule_accuracy']:>10.3f}{c['feature_rule_macro_f1']:>10.3f}")
    print(f"  {'mlp_classifier':<20}{c['mlp_accuracy']:>10.3f}{c['mlp_macro_f1']:>10.3f}")
    print(f"  MLP recall per class: {c['mlp_recall_per_class']}")

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(regressor, MODELS_DIR / "mlp_regressor.pkl")
    joblib.dump(classifier, MODELS_DIR / "mlp_classifier.pkl")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_plots(regressor, yr_te, mlp_pred, classifier, X_te, yc_te)
    print(f"\nSaved models to {MODELS_DIR.name}/ and metrics + plots to {REPORTS_DIR.name}/")


if __name__ == "__main__":
    main()
