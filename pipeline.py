import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

warnings.filterwarnings("ignore")

# ===========================================================================
# Config
# ===========================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_PATH = os.path.join(ROOT, "submissions", "submission.csv")

HORIZONS = [12, 24, 48, 72]
N_FOLDS = 5

GBSA_SEEDS = (
    123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
    279, 239, 70, 77, 31, 2024, 2077, 3077, 123456, 654321,
    4640, 841, 7755, 8525, 2701, 8817, 8864, 4085, 8919, 934,
    4746, 1699, 7401, 7826, 4098, 2921, 1204, 2752, 8384, 1284,
    11, 22, 33, 44, 55, 5505, 6606, 7707, 8808, 9909,
    101010, 202020, 303030, 404040, 505050, 606060, 707070, 808080, 909090, 111111,
)
LGB_SEEDS = (
    123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
    279, 239, 70, 77, 31, 2024, 2077, 3077, 123456, 654321,
    2034, 2035, 2036, 1984, 1991, 3255, 1011, 6241, 2790, 6847,
    8141, 7752, 432, 906, 6217, 7785, 1603, 7609, 965, 2506,
    3771, 7080, 4963, 7939, 2751, 473, 339, 3675, 5535, 4760,
    11, 22, 33, 44, 55, 888, 999, 1010, 1212, 1313,
    1414, 1515, 1616, 1717, 1818, 1919, 2121, 2222, 2323, 2424,
)
RSF_SEEDS = tuple(range(42, 72))
COX_SEEDS = tuple(range(100, 130))

GBSA_CONFIGS = [
    {"learning_rate": 0.010, "subsample": 0.70, "max_depth": 3, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.010, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 15, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.010, "subsample": 0.60, "max_depth": 3, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.005, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 2000},
    {"learning_rate": 0.010, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 20, "min_samples_split": 3, "n_estimators": 1400},
    {"learning_rate": 0.008, "subsample": 0.75, "max_depth": 2, "min_samples_leaf": 15, "min_samples_split": 4, "n_estimators": 1500},
    {"learning_rate": 0.015, "subsample": 0.70, "max_depth": 3, "min_samples_leaf": 10, "min_samples_split": 3, "n_estimators": 1000},
    {"learning_rate": 0.005, "subsample": 0.90, "max_depth": 3, "min_samples_leaf": 18, "min_samples_split": 5, "n_estimators": 2500},
    {"learning_rate": 0.010, "subsample": 0.80, "max_depth": 4, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.020, "subsample": 0.65, "max_depth": 3, "min_samples_leaf": 10, "min_samples_split": 3, "n_estimators": 800},
]
LGB_CONFIGS = {
    12: {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 200, "subsample": 0.7, "colsample_bytree": 0.7,
         "min_child_samples": 10, "reg_alpha": 1.0, "reg_lambda": 3.0, "num_leaves": 4},
    24: {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.7, "colsample_bytree": 0.7,
         "min_child_samples": 8, "reg_alpha": 0.5, "reg_lambda": 2.0, "num_leaves": 7},
    48: {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8,
         "min_child_samples": 5, "reg_alpha": 0.1, "reg_lambda": 1.0, "num_leaves": 4},
}
RSF_CONFIG = {"n_estimators": 500, "min_samples_leaf": 10, "min_samples_split": 6, "max_features": "sqrt", "n_jobs": -1}
COX_CONFIG = {"l1_ratio": 0.5, "alpha_min_ratio": 0.01, "n_alphas": 50,
              "fit_baseline_model": True, "max_iter": 10000, "tol": 1e-5}

# Hand-tuned reference blend weights, per horizon, over [GBSA, LGB, RSF, Cox]
REF_WEIGHTS = {12: [0.97, 0.03, 0, 0], 24: [0.95, 0.05, 0, 0], 48: [0.45, 0.55, 0, 0]}
POWER_CAL_24 = 1.1
N_RESTARTS = 80


# ===========================================================================
# Metric: Hybrid = 0.3 × C-index + 0.7 × (1 − weighted Brier)
# ===========================================================================
def c_index(time_, event, risk):
    concordant = comparable = 0
    for i in range(len(time_)):
        if event[i] != 1:
            continue
        for j in range(len(time_)):
            if i == j or time_[i] >= time_[j]:
                continue
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5
    return concordant / comparable if comparable > 0 else 0.5


def brier(time_, event, prob, horizon):
    # Observations censored before the horizon are excluded
    valid = ~((event == 0) & (time_ < horizon))
    if valid.sum() == 0:
        return 0.25
    y = ((event == 1) & (time_ <= horizon)).astype(float)[valid]
    return float(np.mean((np.clip(prob[valid], 0, 1) - y) ** 2))


def hybrid_score(time_, event, preds):
    p24, p48, p72 = preds[:, 1], preds[:, 2], preds[:, 3]
    c = c_index(time_, event, 0.3 * p24 + 0.4 * p48 + 0.3 * p72)
    wb = 0.3 * brier(time_, event, p24, 24) + 0.4 * brier(time_, event, p48, 48) + 0.3 * brier(time_, event, p72, 72)
    return 0.3 * c + 0.7 * (1 - wb), c, wb


# ===========================================================================
# Helpers
# ===========================================================================
def ipcw_weights(times, events, horizon):
    """Inverse probability of censoring weights (Kaplan-Meier on censoring)."""
    unique_t = np.sort(np.unique(times))
    surv = np.ones(len(unique_t))
    for i, t in enumerate(unique_t):
        at_risk = (times >= t).sum()
        if at_risk > 0:
            surv[i] = 1 - ((times == t) & (events == 0)).sum() / at_risk
        if i > 0:
            surv[i] *= surv[i - 1]

    def G(t):
        idx = np.searchsorted(unique_t, t, side="right") - 1
        return max(surv[idx], 0.01) if idx >= 0 else 1.0

    w = np.ones(len(times))
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon:
            w[i] = 1.0 / G(times[i])
        elif times[i] >= horizon:
            w[i] = 1.0 / G(horizon)
    return w


def monotone(preds):
    r = np.clip(preds, 0, 1)
    for i in range(1, r.shape[1]):
        r[:, i] = np.maximum(r[:, i], r[:, i - 1])
    return r


def surv_to_probs(model, X):
    """Survival function : P(hit <= h) for each horizon."""
    out = np.empty((len(X), len(HORIZONS)))
    for i, fn in enumerate(model.predict_survival_function(X)):
        t_min, t_max = fn.domain
        out[i] = 1.0 - fn(np.clip(HORIZONS, t_min, t_max))
    return out


def cv_survival(make_model, X, X_test, y, strat, seed):
    """One 5-fold CV run of a survival model: OOF and fold-averaged test preds."""
    oof, test = np.zeros((len(X), 4)), np.zeros((len(X_test), 4))
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, strat):
        m = make_model().fit(X.iloc[tr], y[tr])
        oof[va] = surv_to_probs(m, X.iloc[va])
        test += surv_to_probs(m, X_test) / N_FOLDS
    return oof, test


def create_features(df):
    r = df.copy()
    dist = r["dist_min_ci_0_5h"].clip(lower=1)
    speed = r["closing_speed_m_per_h"]
    perims = r["num_perimeters_0_5h"]
    area = r["area_first_ha"]

    # Distance transforms
    r["log_distance"] = np.log1p(dist)
    r["inv_distance"] = 1 / (dist / 1000 + 0.1)
    r["inv_distance_sq"] = r["inv_distance"] ** 2
    r["sqrt_distance"] = np.sqrt(dist)
    r["dist_km"] = dist / 1000
    r["dist_km_sq"] = (dist / 1000) ** 2
    r["dist_km_cb"] = (dist / 1000) ** 3
    r["dist_rank"] = dist.rank(pct=True)

    # Fire size relative to distance
    radius = np.sqrt(area * 10000 / np.pi)
    r["fire_radius_km"] = radius / 1000
    r["radius_to_dist"] = radius / dist
    r["area_to_dist_ratio"] = area / (dist / 1000 + 0.1)
    r["log_area_dist_ratio"] = np.log1p(area) - np.log1p(dist)

    # Time-to-arrival estimates
    r["has_movement"] = (perims > 1).astype(float)
    closing = speed.clip(lower=0)
    r["eta_hours"] = np.where(closing > 0.01, dist / closing, 9999).clip(max=9999)
    r["log_eta"] = np.log1p(r["eta_hours"].clip(0, 9999))
    eff_closing = closing + r["radial_growth_rate_m_per_h"].clip(lower=0)
    r["effective_closing_speed"] = eff_closing
    r["eta_effective"] = np.where(eff_closing > 0.01, dist / eff_closing, 9999).clip(max=9999)

    # Threat / intensity
    r["threat_score"] = r["alignment_abs"] * speed / np.log1p(dist)
    r["threat_score_sq"] = r["threat_score"] ** 2
    r["fire_urgency"] = perims * speed
    r["growth_intensity"] = r["area_growth_rate_ha_per_h"] * perims

    # Distance zones and time of occurrence
    r["zone_critical"] = (dist < 5000).astype(float)
    r["zone_warning"] = ((dist >= 5000) & (dist < 10000)).astype(float)
    r["zone_safe"] = (dist >= 10000).astype(float)
    r["is_summer"] = r["event_start_month"].isin([6, 7, 8]).astype(float)
    r["is_afternoon"] = ((r["event_start_hour"] >= 12) & (r["event_start_hour"] < 20)).astype(float)

    drop = ["relative_growth_0_5h", "projected_advance_m", "centroid_displacement_m", "centroid_speed_m_per_h",
            "closing_speed_abs_m_per_h", "area_growth_abs_0_5h", "dist_bin"]
    r = r.drop(columns=[c for c in drop if c in r.columns])
    r = r.drop(columns=r.select_dtypes(include=["category", "object"]).columns)
    r = r.replace([np.inf, -np.inf], np.nan)
    num = r.select_dtypes(include="number").columns
    r[num] = r[num].fillna(0)
    return r


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    print(f"Train: {train.shape} | Test: {test.shape}")

    event = train["event"].values
    t_hit = train["time_to_hit_hours"].values
    y_surv = Surv.from_arrays(event=train["event"].astype(bool), time=train["time_to_hit_hours"])

    X_surv = train.drop(columns=["event_id", "event", "time_to_hit_hours"], errors="ignore")
    X_surv_test = test.drop(columns=["event_id"], errors="ignore")
    feat_train, feat_test = create_features(train), create_features(test)
    X_feat = feat_train.drop(columns=["event_id", "event", "time_to_hit_hours", "dist_bin"], errors="ignore")
    X_feat_test = feat_test[X_feat.columns]

    # == 1. GBSA ensemble (raw features) ======================================
    print(f"\n[GBSA] {len(GBSA_CONFIGS)} configs × {len(GBSA_SEEDS)} seeds × {N_FOLDS}-fold CV")
    oof_gbsa, test_gbsa = np.zeros((len(train), 4)), np.zeros((len(test), 4))
    for cfg in GBSA_CONFIGS:
        cfg_oof, cfg_test = np.zeros((len(train), 4)), np.zeros((len(test), 4))
        for seed in GBSA_SEEDS:
            o, t = cv_survival(lambda: GradientBoostingSurvivalAnalysis(**cfg, random_state=seed),
                               X_surv, X_surv_test, y_surv, event, seed)
            cfg_oof += o / len(GBSA_SEEDS)
            cfg_test += t / len(GBSA_SEEDS)
        oof_gbsa += cfg_oof / len(GBSA_CONFIGS)
        test_gbsa += cfg_test / len(GBSA_CONFIGS)

    # == 2. LightGBM IPCW per horizon (engineered features) ===================
    print(f"[LGB-IPCW] 3 horizons × {len(LGB_SEEDS)} seeds × {N_FOLDS}-fold CV")
    oof_lgb, test_lgb = {}, {}
    for h in [12, 24, 48]:
        # Obs censored before h have an unknown label: excluded from training
        y_bin = ((event == 1) & (t_hit <= h)).astype(float)
        known = ~((event == 0) & (t_hit < h))
        known_idx, unknown_idx = np.where(known)[0], np.where(~known)[0]
        make = lambda seed: lgb.LGBMClassifier(**LGB_CONFIGS[h], objective="binary", random_state=seed, verbose=-1)
        oof_sum, test_sum = np.zeros(len(train)), np.zeros(len(test))

        for seed in LGB_SEEDS:
            seed_oof = np.zeros(len(train))
            for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(known_idx, y_bin[known]):
                tr, va = known_idx[tr], known_idx[va]
                m = make(seed).fit(X_feat.iloc[tr], y_bin[tr], sample_weight=ipcw_weights(t_hit[tr], event[tr], h))
                seed_oof[va] = m.predict_proba(X_feat.iloc[va])[:, 1]
            if len(unknown_idx):
                seed_oof[unknown_idx] = m.predict_proba(X_feat.iloc[unknown_idx])[:, 1]
            oof_sum += seed_oof

            m = make(seed).fit(X_feat.iloc[known_idx], y_bin[known_idx],
                               sample_weight=ipcw_weights(t_hit[known_idx], event[known_idx], h))
            test_sum += m.predict_proba(X_feat_test)[:, 1]

        oof_lgb[h], test_lgb[h] = oof_sum / len(LGB_SEEDS), test_sum / len(LGB_SEEDS)

    # == 3. Random Survival Forest (raw features) =============================
    print(f"[RSF] {len(RSF_SEEDS)} seeds × {N_FOLDS}-fold CV")
    oof_rsf, test_rsf = np.zeros((len(train), 4)), np.zeros((len(test), 4))
    for seed in RSF_SEEDS:
        o, t = cv_survival(lambda: RandomSurvivalForest(**RSF_CONFIG, random_state=seed),
                           X_surv, X_surv_test, y_surv, event, seed)
        oof_rsf += o / len(RSF_SEEDS)
        test_rsf += t / len(RSF_SEEDS)

    # == 4. CoxNet (scaled engineered features) ===============================
    print(f"[CoxNet] {len(COX_SEEDS)} seeds × {N_FOLDS}-fold CV")
    X_cox, X_cox_test = X_feat.astype(float), X_feat_test.astype(float)
    oof_cox, test_cox = np.zeros((len(train), 4)), np.zeros((len(test), 4))
    n_ok = 0
    for seed in COX_SEEDS:
        seed_oof, seed_test = np.zeros((len(train), 4)), np.zeros((len(test), 4))
        folds_ok = 0
        for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X_cox, event):
            try:  # CoxNet can fail to converge on some folds
                sc = StandardScaler().fit(X_cox.iloc[tr])
                m = CoxnetSurvivalAnalysis(**COX_CONFIG).fit(sc.transform(X_cox.iloc[tr]), y_surv[tr])
                seed_oof[va] = surv_to_probs(m, sc.transform(X_cox.iloc[va]))
                seed_test += surv_to_probs(m, sc.transform(X_cox_test)) / N_FOLDS
                folds_ok += 1
            except Exception:
                continue
        if folds_ok >= 3:
            oof_cox += seed_oof
            test_cox += seed_test
            n_ok += 1
    if n_ok:
        oof_cox /= n_ok
        test_cox /= n_ok

    # == 5. Blending ==========================================================
    for arr in (oof_gbsa, test_gbsa):
        arr[:, 1] = np.clip(arr[:, 1] ** POWER_CAL_24, 0, 1)

    oof_views = {h: [oof_gbsa[:, i], oof_lgb[h], oof_rsf[:, i], oof_cox[:, i]] for i, h in enumerate([12, 24, 48])}
    test_views = {h: [test_gbsa[:, i], test_lgb[h], test_rsf[:, i], test_cox[:, i]] for i, h in enumerate([12, 24, 48])}

    def blend(views, weights):
        out = np.ones((len(views[12][0]), 4))  # prob_72h = 1
        for i, h in enumerate([12, 24, 48]):
            out[:, i] = sum(w * v for w, v in zip(weights[h], views[h]))
        return monotone(out)

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def to_weights(params):
        return {h: softmax(params[4 * i:4 * i + 4]) for i, h in enumerate([12, 24, 48])}

    def objective(params):
        return -hybrid_score(t_hit, event, blend(oof_views, to_weights(params)))[0]

    s_ref = hybrid_score(t_hit, event, blend(oof_views, REF_WEIGHTS))[0]

    # Nelder-Mead on softmax-parametrized weights: one start at the reference weights + random restarts
    print(f"[OPT] Nelder-Mead, {N_RESTARTS} restarts")
    rng = np.random.default_rng(2026)
    x_ref = np.concatenate([[np.log(w[0]), np.log(w[1]), -5.0, -5.0] for w in REF_WEIGHTS.values()])
    starts = [(x_ref, 800)] + [(rng.normal(0, 1.5, size=12), 600) for _ in range(N_RESTARTS - 1)]
    best = min((minimize(objective, x0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": it})
                for x0, it in starts), key=lambda r: r.fun)
    w_opt = to_weights(best.x)

    if -best.fun > s_ref:
        weights, label = w_opt, "optimized"
        for h in [12, 24, 48]:
            print(f"  w{h} [GBSA, LGB, RSF, Cox] = {w_opt[h].round(3)}")
    else:
        weights, label = REF_WEIGHTS, "reference"

    # == 6. Results & submission ==============================================
    final_oof = blend(oof_views, weights)
    final_test = blend(test_views, weights)
    hyb, c, wb = hybrid_score(t_hit, event, final_oof)

    print("\n" + "=" * 50)
    print(f"  OOF Hybrid score : {hyb:.5f}  ({label} blend)")
    print(f"  C-index          : {c:.4f}")
    print(f"  Weighted Brier   : {wb:.5f}")
    print("=" * 50)

    sub = pd.DataFrame({"event_id": test["event_id"].values,
                        **{f"prob_{h}h": final_test[:, i] for i, h in enumerate(HORIZONS)}})
    sub = sample_sub[["event_id"]].merge(sub, on="event_id", how="left")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sub.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH} ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
