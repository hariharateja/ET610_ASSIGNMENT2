"""OULAD: preprocessing, feature engineering, temporal visualisation and ML.

Hypotheses
  O-H1  At-risk learners (Fail/Withdrawn) diverge from successful learners in
        weekly VLE engagement within the first weeks, so early clickstream
        features alone can flag them (AUC >= 0.75 by week 4).
  O-H2  Submission latency (days submitted relative to the deadline) is
        associated with failure even after controlling for assessment score.
  O-H3  Engagement *regularity* (active-day ratio, weekly variability, recency)
        adds predictive value beyond raw click volume.

Run:  .venv/bin/python code/oulad_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score,
                             silhouette_score, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import style

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "archive"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
style.apply()
RNG = 42
results = {}

# ---------------------------------------------------------------- load ----
print("loading ...")
info = pd.read_csv(RAW / "studentInfo.csv")
reg = pd.read_csv(RAW / "studentRegistration.csv")
ass = pd.read_csv(RAW / "assessments.csv")
sass = pd.read_csv(RAW / "studentAssessment.csv", na_values="?")
vle = pd.read_csv(RAW / "vle.csv")
svle = pd.read_csv(RAW / "studentVle.csv",
                   dtype={"id_student": "int32", "id_site": "int32",
                          "date": "int16", "sum_click": "int32"})
courses = pd.read_csv(RAW / "courses.csv")

KEY = ["code_module", "code_presentation", "id_student"]
n_raw = len(info)

# ----------------------------------------------------------- cleaning ----
info = info.drop_duplicates(KEY)
info = info.merge(reg, on=KEY, how="left")
# learners who unregistered before the module started never had a chance to
# engage; they are excluded from all analyses
early_unreg = info["date_unregistration"].notna() & (info["date_unregistration"] <= 0)
info = info[~early_unreg].copy()
info["enr"] = np.arange(len(info))                       # enrolment id
info["at_risk"] = info["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
info["year"] = info["code_presentation"].str[:4].astype(int)
info["imd_num"] = (info["imd_band"].str.extract(r"(\d+)")[0].astype(float))
edu_map = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
           "HE Qualification": 3, "Post Graduate Qualification": 4}
info["edu_num"] = info["highest_education"].map(edu_map)
info["age_num"] = info["age_band"].map({"0-35": 0, "35-55": 1, "55<=": 2})
info["female"] = (info["gender"] == "F").astype(int)
info["disab"] = (info["disability"] == "Y").astype(int)
info["imd_num"] = info["imd_num"].fillna(info["imd_num"].median())
info["date_registration"] = info["date_registration"].fillna(info["date_registration"].median())
results["cleaning"] = {
    "enrolments_raw": int(n_raw), "unregistered_before_start": int(early_unreg.sum()),
    "enrolments_used": int(len(info)),
    "outcome_counts": info["final_result"].value_counts().to_dict(),
    "vle_rows": int(len(svle)),
}

emap = info[KEY + ["enr"]]
svle = svle.merge(emap, on=KEY, how="inner")
svle = svle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
results["cleaning"]["vle_rows_used"] = int(len(svle))
# daily aggregation (one row per enrolment-day) - the unit for regularity features
daily = svle.groupby(["enr", "date"], as_index=False)["sum_click"].sum()
daily_type = svle.groupby(["enr", "date", "activity_type"], as_index=False)["sum_click"].sum()
del svle

# assessments: TMA/CMA only (exam dates are frequently missing / end-of-module)
ass = ass[ass["assessment_type"] != "Exam"].dropna(subset=["date"])
sub = sass[sass["is_banked"] == 0].merge(ass, on="id_assessment")
sub = sub.merge(info[KEY + ["enr"]], on=["code_module", "code_presentation", "id_student"])
sub["late_days"] = sub["date_submitted"] - sub["date"]
due = info[["enr", "code_module", "code_presentation"]].merge(
    ass[["code_module", "code_presentation", "id_assessment", "date", "assessment_type"]],
    on=["code_module", "code_presentation"])
due = due.merge(sub[["enr", "id_assessment", "date_submitted", "late_days", "score"]],
                on=["enr", "id_assessment"], how="left")

# ------------------------------------------------- feature engineering ----
DEMO = ["num_of_prev_attempts", "studied_credits", "imd_num", "edu_num", "age_num",
        "female", "disab", "date_registration"]
VOLUME = ["clicks_total", "clicks_prestart"]
REGULAR = ["active_days", "active_ratio", "weekly_cv", "active_weeks_ratio",
           "days_since_last", "type_entropy", "recent_trend"]
ASSESS = ["n_due", "submit_rate", "mean_late", "mean_score"]


def features_at(cutoff):
    """Features computable using only data observed before `cutoff` (day)."""
    d = daily[daily["date"] < cutoff]
    f = pd.DataFrame(index=info["enr"])
    g = d.groupby("enr")
    f["clicks_total"] = g["sum_click"].sum()
    f["clicks_prestart"] = d[d["date"] < 0].groupby("enr")["sum_click"].sum()
    f["active_days"] = d[d["date"] >= 0].groupby("enr").size()
    f["active_ratio"] = f["active_days"] / max(cutoff, 1)
    wk = d.assign(week=d["date"].clip(lower=0) // 7).groupby(["enr", "week"])["sum_click"].sum()
    nweeks = max(int(np.ceil(cutoff / 7)), 1)
    wk_full = wk.unstack(fill_value=0).reindex(columns=range(nweeks), fill_value=0)
    wk_full = wk_full.reindex(f.index, fill_value=0)
    f["weekly_cv"] = (wk_full.std(axis=1) / wk_full.mean(axis=1).replace(0, np.nan)).fillna(0)
    f["active_weeks_ratio"] = (wk_full > 0).mean(axis=1)
    last = g["date"].max()
    f["days_since_last"] = (cutoff - last).fillna(cutoff + 30)
    # recent trend: last-2-week clicks / earlier weekly mean (log ratio)
    if nweeks >= 3:
        recent = wk_full.iloc[:, -2:].mean(axis=1)
        earlier = wk_full.iloc[:, :-2].mean(axis=1)
        f["recent_trend"] = np.log1p(recent) - np.log1p(earlier)
    else:
        f["recent_trend"] = 0.0
    t = daily_type[daily_type["date"] < cutoff].groupby(["enr", "activity_type"])["sum_click"].sum()
    p = t / t.groupby(level=0).transform("sum")
    f["type_entropy"] = (-(p * np.log2(p))).groupby(level=0).sum()
    a = due[due["date"] < cutoff].copy()
    a.loc[a["date_submitted"] >= cutoff, ["date_submitted", "late_days", "score"]] = np.nan
    ga = a.groupby("enr")
    f["n_due"] = ga.size()
    f["submit_rate"] = ga["date_submitted"].apply(lambda s: s.notna().mean())
    f["mean_late"] = ga["late_days"].mean()
    f["mean_score"] = ga["score"].mean()
    f = f.fillna({c: 0 for c in VOLUME + REGULAR + ["n_due"]})
    f["submit_rate"] = f["submit_rate"].fillna(1.0)          # nothing due yet
    f["mean_late"] = f["mean_late"].fillna(0)
    f["mean_score"] = f["mean_score"].fillna(f["mean_score"].median() if f["mean_score"].notna().any() else 0)
    f[["clicks_total", "clicks_prestart"]] = np.log1p(f[["clicks_total", "clicks_prestart"]])
    return f.reset_index().merge(info[["enr", "at_risk", "year", "final_result",
                                       "date_unregistration"] + DEMO], on="enr")


# --------------------------------------------- temporal visualisations ----
print("temporal figures ...")
wk_all = daily.assign(week=daily["date"] // 7).groupby(["enr", "week"])["sum_click"].sum()
wk_all = wk_all.unstack(fill_value=0).reindex(columns=range(-3, 39), fill_value=0)
wk_all = wk_all.reindex(info["enr"], fill_value=0)
wk_all.index = info["enr"].values
res_by_enr = info.set_index("enr")["final_result"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for oc in style.OUTCOME_ORDER:
    m = wk_all[res_by_enr.reindex(wk_all.index).values == oc]
    mu = m.mean(); se = m.sem() * 1.96
    ax.plot(mu.index, mu.values, color=style.OUTCOME_COLORS[oc], label=oc)
    ax.fill_between(mu.index, mu - se, mu + se, color=style.OUTCOME_COLORS[oc], alpha=0.15, lw=0)
ax.axvline(0, color=style.MUTED, lw=0.8, ls="--")
ax.text(0.3, ax.get_ylim()[1] * 0.92, "module start", color=style.INK2, fontsize=7)
ax.set_xlabel("Week relative to module start"); ax.set_ylabel("Mean clicks per learner")
ax.set_title("a. Weekly VLE clicks by final outcome")
ax.legend(ncol=2, loc="upper right")
# share of learners active each week
ax = axes[1]
for oc in style.OUTCOME_ORDER:
    m = wk_all[res_by_enr.reindex(wk_all.index).values == oc]
    ax.plot(m.columns, (m > 0).mean().values * 100, color=style.OUTCOME_COLORS[oc], label=oc)
ax.set_xlabel("Week"); ax.set_ylabel("% learners active")
ax.set_title("b. Weekly active share")
ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig(FIG / "oulad_weekly_clicks.png"); plt.close(fig)

# divergence statistic by week (H1): effect size at-risk vs success
eff = []
risk = info.set_index("enr")["at_risk"].reindex(wk_all.index).values
for w in range(0, 11):
    a_, b_ = wk_all.loc[risk == 1, w], wk_all.loc[risk == 0, w]
    u, pval = stats.mannwhitneyu(a_, b_)
    eff.append({"week": w, "median_at_risk": float(a_.median()), "median_success": float(b_.median()),
                "rank_biserial": float(1 - 2 * u / (len(a_) * len(b_))), "p": float(pval)})
results["H1_weekly_divergence"] = eff

# submission latency (H2)
sub_l = sub.merge(info[["enr", "final_result", "at_risk"]], on="enr")
stud_late = sub_l.groupby("enr").agg(mean_late=("late_days", "mean"),
                                    mean_score=("score", "mean"),
                                    n_sub=("late_days", "size"))
nd = due.groupby("enr").agg(n_due=("id_assessment", "size"),
                            n_done=("date_submitted", lambda s: s.notna().sum()))
stud_late = stud_late.join(nd, how="right").join(info.set_index("enr")[["final_result", "at_risk"]])
stud_late["submit_rate"] = stud_late["n_done"] / stud_late["n_due"]
stud_late["late_bucket"] = pd.cut(stud_late["mean_late"], [-1e9, -3, 0, 7, 1e9],
                                  labels=["Early (>3d)", "Just in time", "Late 1-7d", "Late >7d"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
ax = axes[0]
# on-time submission rate by assessment sequence number
due_s = due.merge(info[["enr", "final_result"]], on="enr").sort_values(["enr", "date"])
due_s["seq"] = due_s.groupby("enr").cumcount() + 1
due_s["on_time"] = (due_s["late_days"] <= 0).astype(float)
due_s = due_s[due_s["seq"] <= 10]
for oc in style.OUTCOME_ORDER:
    s = due_s[due_s["final_result"] == oc].groupby("seq")["on_time"].mean() * 100
    ax.plot(s.index, s.values, marker="o", ms=3.5, color=style.OUTCOME_COLORS[oc], label=oc)
ax.set_xlabel("Assessment # in the module (chronological)"); ax.set_ylabel("% submitted on time")
ax.set_title("a. On-time submission across the term")
ax.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.set_ylim(0, 100)
ax = axes[1]
tab = (stud_late.dropna(subset=["late_bucket"])
       .groupby("late_bucket", observed=True)["final_result"].value_counts(normalize=True)
       .unstack()[style.OUTCOME_ORDER] * 100)
left = np.zeros(len(tab))
for oc in style.OUTCOME_ORDER:
    ax.barh(tab.index.astype(str), tab[oc], left=left, color=style.OUTCOME_COLORS[oc],
            edgecolor="white", linewidth=1, label=oc, height=0.65)
    for i, (l, v) in enumerate(zip(left, tab[oc])):
        if v > 9:
            ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white")
    left += tab[oc].values
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.set_xlabel("% of learners"); ax.set_title("b. Outcome by mean submission latency")
fig.tight_layout(); fig.savefig(FIG / "oulad_latency.png"); plt.close(fig)

# H2 logistic regression: at_risk ~ latency + score + submit_rate (completers of >=1 TMA/CMA)
h2 = stud_late.dropna(subset=["mean_late", "mean_score"]).copy()
h2["mean_late_c"] = h2["mean_late"].clip(-30, 60)
Xh = h2[["mean_late_c", "mean_score", "submit_rate"]]
Xh = (Xh - Xh.mean()) / Xh.std()
lr_h2 = sm.Logit(h2["at_risk"], sm.add_constant(Xh)).fit(disp=0)
ci = np.exp(lr_h2.conf_int())
results["H2_logit"] = {k: {"OR_per_SD": float(np.exp(lr_h2.params[k])), "ci_lo": float(ci.loc[k, 0]),
                           "ci_hi": float(ci.loc[k, 1]), "p": float(lr_h2.pvalues[k])}
                       for k in Xh.columns}
results["H2_logit"]["n"] = int(len(h2))
results["H2_bucket_table"] = tab.round(1).to_dict()
results["H2_spearman_late_vs_risk"] = float(stats.spearmanr(h2["mean_late"], h2["at_risk"])[0])

# --------------------------------------------------- machine learning ----
print("ML ...")
CUTOFFS = [7, 14, 28, 42, 56, 84, 112, 140, 182]
SETS = {"Demographics + volume": DEMO + VOLUME,
        "+ regularity": DEMO + VOLUME + REGULAR,
        "+ regularity + assessment": DEMO + VOLUME + REGULAR + ASSESS}


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                               class_weight="balanced_subsample", n_jobs=-1, random_state=RNG),
    }


curve = []
store = {}
for c in CUTOFFS:
    F = features_at(c)
    # only learners still enrolled at the cutoff are predicted (early warning)
    F = F[~(F["date_unregistration"] < c)]
    tr, te = F[F["year"] == 2013], F[F["year"] == 2014]
    for sname, cols in SETS.items():
        for mname, mdl in models().items():
            mdl.fit(tr[cols], tr["at_risk"])
            p = mdl.predict_proba(te[cols])[:, 1]
            curve.append({"cutoff_day": c, "week": c // 7, "set": sname, "model": mname,
                          "auc": roc_auc_score(te["at_risk"], p), "n_test": len(te),
                          "prev_test": float(te["at_risk"].mean())})
            if c == 56 and sname == "+ regularity + assessment":
                store[mname] = (mdl, te, p, cols)
    print(" cutoff", c, "done")
curve = pd.DataFrame(curve)
curve.to_csv(OUT / "oulad_auc_by_cutoff.csv", index=False)
results["auc_curve"] = curve.round(4).to_dict(orient="records")

# detailed metrics at week 8
det = {}
for mname, (mdl, te, p, cols) in store.items():
    yhat = (p >= 0.5).astype(int)
    det[mname] = {"auc": roc_auc_score(te["at_risk"], p), "f1": f1_score(te["at_risk"], yhat),
                  "precision": precision_score(te["at_risk"], yhat),
                  "recall": recall_score(te["at_risk"], yhat),
                  "balanced_acc": balanced_accuracy_score(te["at_risk"], yhat),
                  "confusion": confusion_matrix(te["at_risk"], yhat).tolist(),
                  "n_test": int(len(te)), "n_train_cols": len(cols)}
results["week8_metrics"] = det

# 5-fold CV inside training years for the week-8 RF (internal validation)
from sklearn.model_selection import cross_val_score, StratifiedKFold
F8 = features_at(56); F8 = F8[~(F8["date_unregistration"] < 56)]
tr8 = F8[F8["year"] == 2013]
cols = SETS["+ regularity + assessment"]
cv = cross_val_score(models()["RandomForest"], tr8[cols], tr8["at_risk"],
                     cv=StratifiedKFold(5, shuffle=True, random_state=RNG), scoring="roc_auc")
results["week8_rf_cv_auc"] = {"mean": float(cv.mean()), "sd": float(cv.std())}

rf, te, p, cols = store["RandomForest"]
pi = permutation_importance(rf, te[cols], te["at_risk"], scoring="roc_auc", n_repeats=5,
                            random_state=RNG, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
results["week8_perm_importance"] = imp.round(4).to_dict()

lr = store["LogReg"][0]
coef = pd.Series(lr[-1].coef_[0], index=cols).sort_values()
results["week8_logreg_coef"] = coef.round(3).to_dict()

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
set_col = {"Demographics + volume": style.C2, "+ regularity": style.C3, "+ regularity + assessment": style.C1}
for sname in SETS:
    for mname, ls in [("RandomForest", "-"), ("LogReg", ":")]:
        s = curve[(curve["set"] == sname) & (curve["model"] == mname)]
        ax.plot(s["week"], s["auc"], ls=ls, marker="o" if ls == "-" else None, ms=3.5,
                color=set_col[sname], label=f"{sname} ({'RF' if mname == 'RandomForest' else 'LR'})")
ax.axhline(0.75, color=style.MUTED, lw=0.8, ls="--")
ax.set_xlabel("Prediction cut-off (week of module)"); ax.set_ylabel("Test ROC-AUC (2014 cohorts)")
ax.set_title("a. Early-warning accuracy vs. time")
ax.legend(fontsize=6.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
ax = axes[1]
top = imp.head(10)[::-1]
PRETTY = {"mean_score": "Mean assessment score", "submit_rate": "Submission rate", "mean_late": "Mean latency (days)",
          "edu_num": "Prior education", "active_weeks_ratio": "Active-week ratio", "days_since_last": "Days since last login",
          "type_entropy": "Activity-type entropy", "recent_trend": "Recent click trend", "imd_num": "IMD band",
          "active_days": "Active days", "active_ratio": "Active-day ratio", "date_registration": "Registration day",
          "n_due": "Assessments due", "clicks_total": "Total clicks (log)", "studied_credits": "Studied credits",
          "weekly_cv": "Weekly burstiness (CV)", "clicks_prestart": "Pre-start clicks", "num_of_prev_attempts": "Previous attempts"}
ax.barh([PRETTY.get(i, i) for i in top.index], top.values, color=style.C1, height=0.6)
ax.set_xlabel("Permutation importance (Δ AUC)"); ax.grid(axis="y", visible=False)
ax.set_title("b. RF feature importance, week 8")
fig.tight_layout(); fig.savefig(FIG / "oulad_ml.png"); plt.close(fig)

# ----------------------------------------- clustering of trajectories ----
print("clustering ...")
traj = np.log1p(wk_all.loc[:, 0:19])            # first 20 weeks
ks = {}
sample = traj.sample(8000, random_state=RNG)
for k in range(2, 8):
    km = KMeans(k, n_init=10, random_state=RNG).fit(sample)
    ks[k] = float(silhouette_score(sample, km.labels_, sample_size=4000, random_state=RNG))
results["kmeans_silhouette"] = ks
K = 4
km = KMeans(K, n_init=20, random_state=RNG).fit(traj)
lab = pd.Series(km.labels_, index=traj.index)
# order clusters by overall engagement
order = traj.groupby(lab).mean().mean(axis=1).sort_values(ascending=False).index
names = ["High & sustained", "Moderate & steady", "Early drop-off", "Minimal / absent"]
rename = {old: names[i] for i, old in enumerate(order)}
lab = lab.map(rename)
ct = pd.crosstab(lab, res_by_enr.reindex(lab.index), normalize="index")[style.OUTCOME_ORDER] * 100
ct = ct.loc[names]
results["kmeans_clusters"] = {"k": K, "sizes": lab.value_counts().reindex(names).to_dict(),
                              "outcomes_pct": ct.round(1).to_dict(orient="index")}

clu_col = dict(zip(names, [style.C1, style.C3, style.C2, style.C7]))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
for n in names:
    m = wk_all.loc[lab.index[lab == n], 0:19].mean()
    ax.plot(m.index, m.values, color=clu_col[n], label=f"{n} (n={int((lab == n).sum()):,})")
ax.set_xlabel("Week"); ax.set_ylabel("Mean clicks per learner")
ax.set_title("a. K-Means (k=4) engagement trajectories")
ax.legend(fontsize=6.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
ax = axes[1]
left = np.zeros(len(ct))
for oc in style.OUTCOME_ORDER:
    ax.barh(ct.index, ct[oc], left=left, color=style.OUTCOME_COLORS[oc], edgecolor="white",
            linewidth=1, label=oc, height=0.65)
    for i, (l, v) in enumerate(zip(left, ct[oc])):
        if v > 9:
            ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white")
    left += ct[oc].values
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.set_xlabel("% of learners"); ax.set_title("b. Final outcome per cluster")
ax.legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.3))
fig.tight_layout(); fig.savefig(FIG / "oulad_clusters.png"); plt.close(fig)

(OUT / "oulad_results.json").write_text(json.dumps(results, indent=2, default=float))
print("done")
