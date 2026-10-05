"""Moodle IT-2507 log: cleaning, sessionisation, features, temporal views, ML.

Outcome (no grades in the log): completion of the *Final* course-project
assignment ("A submission has been submitted." in Assignment: Final).

Hypotheses
  M-H1  Self-regulated, distributed engagement during the first 10 weeks
        (more active weeks, more sessions, self-monitoring of grades/feedback)
        distinguishes learners who go on to complete the Final project.
  M-H2  Procrastination: learners who submit the first milestone (Synopsis)
        close to its deadline, or not at all, are less likely to complete
        the Final project; self-testing on practice quizzes is higher among
        completers.
  M-H3  Learners segment into distinct behavioural profiles (e.g. steady
        self-testers vs. exam-window crammers vs. disengaged) that differ in
        completion.

Run:  .venv/bin/python code/moodle_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             balanced_accuracy_score, silhouette_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import style

ROOT = Path(__file__).resolve().parents[1]
FIG, OUT = ROOT / "figures", ROOT / "outputs"
style.apply()
RNG = 42
R = {}

# ----------------------------------------------------------- cleaning ----
d = pd.read_csv(ROOT / "MOODLE Data.csv", encoding="utf-8-sig")
R["rows_raw"] = int(len(d))
# The 'User ID' column is truncated ("e id '3") for SCORM events; the true id
# is always present in the free-text Description, so recover it from there.
d["uid"] = d["Description"].str.extract(r"user with (?:the )?id '(\d+)'")[0]
R["user_id_repaired"] = int((~d["User ID"].str.match(r"^'\d+'$")).sum())
d["t"] = pd.to_datetime(d["Time"], format="%d/%m/%y, %H:%M:%S")
d = d.drop_duplicates(["uid", "t", "Event context", "Event name", "Description"])
R["rows_dedup"] = int(len(d))
d["ctx"] = d["Event context"].str.replace(r"\s+", " ", regex=True).str.strip()
d["code"] = d["Content code"].fillna("Navigation")
d = d.sort_values(["uid", "t"]).reset_index(drop=True)

# keep genuine learners: >= 30 events in the term
cnt = d.groupby("uid").size()
keep = cnt[cnt >= 30].index
R["users_raw"] = int(cnt.size); R["users_kept"] = int(len(keep))
d = d[d["uid"].isin(keep)].copy()

START = pd.Timestamp("2025-10-20")           # Monday of first logged week
d["week"] = ((d["t"] - START).dt.days // 7).astype(int)
d["hour"] = d["t"].dt.hour

# ------------------------------------------------------ sessionisation ----
GAP = pd.Timedelta(minutes=30)
d["new_sess"] = (d.groupby("uid")["t"].diff() > GAP) | d.groupby("uid")["t"].diff().isna()
d["sess"] = d.groupby("uid")["new_sess"].cumsum()
sess = d.groupby(["uid", "sess"]).agg(start=("t", "min"), end=("t", "max"), n=("t", "size"))
sess["dur_min"] = (sess["end"] - sess["start"]).dt.total_seconds() / 60
# single-event sessions get a nominal 1 minute of time on task
sess.loc[sess["dur_min"] == 0, "dur_min"] = 1.0
sess = sess.reset_index()
R["sessions"] = {"n": int(len(sess)), "median_dur_min": float(sess["dur_min"].median()),
                 "median_events": float(sess["n"].median())}

# --------------------------------------------------------------- outcome --
final_sub = d[(d["ctx"] == "Assignment: Final") & (d["Event name"] == "A submission has been submitted.")]
syn_sub = d[(d["ctx"] == "Assignment: Synopsis") & (d["Event name"] == "A submission has been submitted.")]
users = pd.DataFrame(index=sorted(keep))
users["final_done"] = users.index.isin(final_sub["uid"]).astype(int)
R["final_done_n"] = int(users["final_done"].sum())
FINAL_OPEN = final_sub["t"].min().normalize()            # 29 Dec 2025
SYN_DEADLINE = syn_sub["t"].max().ceil("D")             # last submission day, end of day
R["final_first_submission"] = str(final_sub["t"].min()); R["synopsis_deadline_proxy"] = str(SYN_DEADLINE)

# ------------------------------------------------- feature engineering ----
def build(df, sdf, horizon_weeks):
    f = pd.DataFrame(index=users.index)
    act = df[df["Event name"] != "Quiz attempt auto-saved"]   # auto-saves inflate counts
    g = act.groupby("uid")
    f["events"] = np.log1p(g.size())
    gs = sdf.groupby("uid")
    f["sessions"] = gs.size()
    f["mean_sess_min"] = gs["dur_min"].mean()
    f["time_on_task_h"] = gs["dur_min"].sum() / 60
    f["active_days"] = df.groupby("uid")["t"].apply(lambda s: s.dt.normalize().nunique())
    wk = df.groupby(["uid", "week"]).size().unstack(fill_value=0).reindex(columns=range(horizon_weeks), fill_value=0)
    wk = wk.reindex(users.index, fill_value=0)
    f["active_weeks_ratio"] = (wk > 0).mean(axis=1)
    f["weekly_cv"] = (wk.std(axis=1) / wk.mean(axis=1).replace(0, np.nan))
    gaps = sdf.sort_values("start").groupby("uid")["start"].apply(
        lambda s: s.diff().dt.total_seconds().div(86400).std())
    f["gap_sd_days"] = gaps
    f["night_ratio"] = df.assign(n=df["hour"].lt(6)).groupby("uid")["n"].mean()
    f["weekend_ratio"] = df.assign(n=df["t"].dt.dayofweek.ge(5)).groupby("uid")["n"].mean()
    pq = df[df["code"] == "Practice Quiz"]
    f["practice_attempts"] = pq[pq["Event name"] == "Quiz attempt started"].groupby("uid").size()
    f["practice_share"] = act.assign(p=act["code"].eq("Practice Quiz")).groupby("uid")["p"].mean()
    content = ["Self-Learning Material", "Digital Storyboard", "Additional Learning Material",
               "Must Know Concepts Videos", "Audiobyte", "Live Session Recordings", "Class PPT",
               "Faculty Video", "Reference Material"]
    f["content_views"] = np.log1p(act[act["code"].isin(content)].groupby("uid").size())
    f["content_diversity"] = act[act["code"] != "Navigation"].groupby("uid")["ctx"].nunique()
    sm = df[df["Event name"].isin(["Course user report viewed", "Feedback viewed",
                                    "The status of the submission has been viewed."])]
    f["self_monitoring"] = sm.groupby("uid").size()
    f["forum_glossary"] = df[df["code"].isin(["Discussion Forum", "Glossary"])].groupby("uid").size()
    sc = df[df["Event name"] == "Submitted SCORM raw score"]["Description"].str.extract(r"value of '([\d.]+)'")[0].astype(float)
    f["scorm_mean_score"] = sc.groupby(df.loc[sc.index, "uid"]).mean()
    s1 = syn_sub[syn_sub["t"] < df["t"].max() + pd.Timedelta(seconds=1)].groupby("uid")["t"].min()
    f["synopsis_submitted"] = f.index.isin(s1.index).astype(int)
    f["synopsis_lead_days"] = ((SYN_DEADLINE - s1).dt.total_seconds() / 86400).reindex(f.index)
    zero = ["sessions", "time_on_task_h", "active_days", "practice_attempts", "practice_share",
            "content_diversity", "self_monitoring", "forum_glossary", "events", "content_views"]
    f[zero] = f[zero].fillna(0)
    f["mean_sess_min"] = f["mean_sess_min"].fillna(0)
    f["weekly_cv"] = f["weekly_cv"].fillna(f["weekly_cv"].max())
    f["gap_sd_days"] = f["gap_sd_days"].fillna(f["gap_sd_days"].max())
    for c in ["night_ratio", "weekend_ratio"]:
        f[c] = f[c].fillna(f[c].median())
    f["scorm_mean_score"] = f["scorm_mean_score"].fillna(f["scorm_mean_score"].median())
    f["synopsis_lead_days"] = f["synopsis_lead_days"].fillna(-5)   # not submitted -> below 0
    return f


# early-warning window: everything before the Final project submissions open,
# with Final-assignment events removed (no outcome leakage)
early = d[(d["t"] < FINAL_OPEN) & (d["ctx"] != "Assignment: Final")]
early_s = sess[sess["start"] < FINAL_OPEN]
EW = int((FINAL_OPEN - START).days // 7)
Fe = build(early, early_s, EW)
Fe["final_done"] = users["final_done"]
Fe.to_csv(OUT / "moodle_features_early.csv")
R["early_window_weeks"] = EW

FEATS = [c for c in Fe.columns if c != "final_done"]

# M-H1 / M-H2: group comparison (Mann-Whitney U, rank-biserial r)
comp = []
for c in FEATS:
    a, b = Fe.loc[Fe.final_done == 1, c], Fe.loc[Fe.final_done == 0, c]
    u, p = stats.mannwhitneyu(a, b)
    comp.append({"feature": c, "median_completed": float(a.median()), "median_not": float(b.median()),
                 "r_rb": float(2 * u / (len(a) * len(b)) - 1), "p": float(p)})
comp = pd.DataFrame(comp).sort_values("p")
from statsmodels.stats.multitest import multipletests
comp["p_holm"] = multipletests(comp["p"], method="holm")[1]
comp.to_csv(OUT / "moodle_group_tests.csv", index=False)
R["group_tests"] = comp.round(4).to_dict(orient="records")

# synopsis timing vs completion
syn_tab = pd.crosstab(pd.cut(Fe["synopsis_lead_days"], [-10, -0.01, 1, 7, 100],
                             labels=["Not submitted", "<1 day before", "1-7 days", ">7 days"]),
                      Fe["final_done"])
R["synopsis_table"] = {str(k): v for k, v in syn_tab.to_dict(orient="index").items()}
chi = stats.chi2_contingency(syn_tab)
R["synopsis_chi2"] = {"chi2": float(chi[0]), "p": float(chi[1]), "dof": int(chi[2])}

# --------------------------------------------------- temporal figures ----
grp = users["final_done"].map({1: "Completed Final", 0: "Did not complete"})
wk_ev = d[d["Event name"] != "Quiz attempt auto-saved"].groupby(["uid", "week"]).size().unstack(fill_value=0)
wk_ev = wk_ev.reindex(columns=range(0, 28), fill_value=0)
wk_time = sess.assign(week=((sess["start"] - START).dt.days // 7)).groupby(["uid", "week"])["dur_min"].sum().unstack(fill_value=0)
wk_time = wk_time.reindex(columns=range(0, 28), fill_value=0).reindex(users.index, fill_value=0)
GCOL = {"Completed Final": style.C1, "Did not complete": style.C2}

def wk_label(ax):
    marks = {"Synopsis due": SYN_DEADLINE, "Final opens": FINAL_OPEN,
             "CE quizzes open": pd.Timestamp("2026-01-26")}
    for i, (k, v) in enumerate(marks.items()):
        w = (v - START).days / 7
        ax.axvline(w, color=style.MUTED, lw=0.8, ls="--")
        ax.text(w + 0.2, 1 - 0.08 * (i + 1), k, transform=ax.get_xaxis_transform(), fontsize=6.5, color=style.INK2)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, M, ttl, yl in [(axes[0], wk_ev.reindex(users.index, fill_value=0), "a. Weekly actions per learner", "Mean actions (excl. auto-saves)"),
                       (axes[1], wk_time / 60, "b. Weekly time-on-task (sessionised)", "Mean hours per learner")]:
    for gname, col in GCOL.items():
        m = M[grp.reindex(M.index).values == gname]
        mu, se = m.mean(), m.sem() * 1.96
        ax.plot(mu.index, mu.values, color=col, label=f"{gname} (n={len(m)})")
        ax.fill_between(mu.index, mu - se, mu + se, color=col, alpha=0.15, lw=0)
    wk_label(ax); ax.set_xlabel("Course week (from 20 Oct 2025)"); ax.set_ylabel(yl); ax.set_title(ttl)
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.legend(h, l, ncol=2, fontsize=6.8, loc="lower center")
fig.savefig(FIG / "moodle_weekly.png"); plt.close(fig)

# component mix over time + synopsis ECDF
mix_map = {"Quiz": "Quiz (CE + practice)", "SCORM package": "Content", "Book": "Content", "Page": "Content",
           "File": "Content", "Assignment": "Assignment", "File submissions": "Assignment",
           "Online text submissions": "Assignment", "Forum": "Forum / glossary", "Glossary": "Forum / glossary",
           "System": "Course navigation"}
dd = d[d["Event name"] != "Quiz attempt auto-saved"].assign(cat=lambda x: x["Component"].map(mix_map))
mix = dd.groupby(["week", "cat"]).size().unstack(fill_value=0).reindex(range(0, 28), fill_value=0)
mix_pct = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0) * 100
cats = ["Content", "Quiz (CE + practice)", "Assignment", "Course navigation", "Forum / glossary"]
ccol = [style.C1, style.C2, style.C3, style.AXIS, style.C7]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.4, 1]})
ax = axes[0]
ax.bar(mix.index, mix.sum(axis=1), color=style.GRID, width=0.85, label="_")
ax2 = ax
bottom = np.zeros(len(mix))
for c, col in zip(cats, ccol):
    ax.bar(mix.index, mix[c], bottom=bottom, color=col, width=0.85, edgecolor="white", linewidth=0.5, label=c)
    bottom += mix[c].values
ax.set_xlabel("Course week"); ax.set_ylabel("Actions (all learners)")
ax.set_title("a. What learners do, week by week")
ax.legend(fontsize=6.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
wk_label(ax)
ax = axes[1]
s1 = syn_sub.groupby("uid")["t"].min()
lead = (SYN_DEADLINE - s1).dt.total_seconds() / 86400
for gname, col in GCOL.items():
    v = np.sort(lead[lead.index.isin(grp[grp == gname].index)].values)[::-1]
    v = np.concatenate([[30], v, [-0.2]])
    cum = np.concatenate([[0], np.arange(1, len(v) - 1), [len(v) - 2]]) / len(grp[grp == gname]) * 100
    ax.step(v, cum, where="post", color=col, label=gname)
ax.set_xlim(30, -0.2); ax.set_ylim(0, 100)
ax.set_xlabel("Days before Synopsis deadline (submitted)"); ax.set_ylabel("Cumulative % of group")
ax.set_title("b. Synopsis submission timing"); ax.legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "moodle_mix_synopsis.png"); plt.close(fig)

# ---------------------------------------------------- supervised model ----
X, y = Fe[FEATS], Fe["final_done"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RNG)
mods = {"LogReg (L2)": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                                               class_weight="balanced_subsample", random_state=RNG, n_jobs=-1)}
ml = {}
for name, m in mods.items():
    rows = []
    for tr, te in cv.split(X, y):
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]; yh = (p >= 0.5).astype(int)
        rows.append([roc_auc_score(y.iloc[te], p), f1_score(y.iloc[te], yh),
                     precision_score(y.iloc[te], yh, zero_division=0), recall_score(y.iloc[te], yh),
                     balanced_accuracy_score(y.iloc[te], yh)])
    rows = np.array(rows)
    ml[name] = {k: {"mean": float(rows[:, i].mean()), "sd": float(rows[:, i].std())}
                for i, k in enumerate(["auc", "f1", "precision", "recall", "balanced_acc"])}
# baseline: activity volume only
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
b_auc = []
for tr, te in cv.split(X, y):
    base.fit(X.iloc[tr][["events"]], y.iloc[tr]); b_auc.append(roc_auc_score(y.iloc[te], base.predict_proba(X.iloc[te][["events"]])[:, 1]))
ml["Baseline LogReg (events only)"] = {"auc": {"mean": float(np.mean(b_auc)), "sd": float(np.std(b_auc))}}
# robustness: purely behavioural features (no assignment-related signals)
BEH = [c for c in FEATS if c not in ("synopsis_submitted", "synopsis_lead_days", "self_monitoring")]
rf_b = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                              class_weight="balanced_subsample", random_state=RNG, n_jobs=-1)
bh = [roc_auc_score(y.iloc[te], rf_b.fit(X.iloc[tr][BEH], y.iloc[tr]).predict_proba(X.iloc[te][BEH])[:, 1])
      for tr, te in cv.split(X, y)]
ml["RandomForest (behaviour only)"] = {"auc": {"mean": float(np.mean(bh)), "sd": float(np.std(bh))}}
R["ml"] = ml

rf = mods["RandomForest"].fit(X, y)
lr = mods["LogReg (L2)"].fit(X, y)
# out-of-fold permutation importance (importance measured on held-out folds)
pis = []
for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=RNG).split(X, y):
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                               class_weight="balanced_subsample", random_state=RNG, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
    pis.append(permutation_importance(m, X.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=10,
                                      random_state=RNG).importances_mean)
imp = pd.Series(np.mean(pis, axis=0), index=FEATS).sort_values(ascending=False)
R["rf_perm_importance_insample"] = imp.round(4).to_dict()
coef = pd.Series(lr[-1].coef_[0], index=FEATS).sort_values()
R["logreg_coef"] = coef.round(3).to_dict()

# ------------------------------------------------------- clustering ----
Ff = build(d[d["ctx"] != "Assignment: Final"], sess, 28)
CL = ["events", "sessions", "mean_sess_min", "active_weeks_ratio", "weekly_cv", "night_ratio",
      "practice_share", "content_views", "self_monitoring", "synopsis_lead_days"]
# exam-window concentration: share of actions in weeks 14-15 (26 Jan - 8 Feb, CE quizzes)
ew = d[d["Event name"] != "Quiz attempt auto-saved"]
Ff["exam_window_share"] = (ew[ew["week"].isin([14, 15])].groupby("uid").size() / ew.groupby("uid").size()).reindex(Ff.index).fillna(0)
CL.append("exam_window_share")
Z = StandardScaler().fit_transform(Ff[CL])
sil = {k: float(silhouette_score(Z, KMeans(k, n_init=50, random_state=RNG).fit_predict(Z))) for k in range(2, 8)}
R["moodle_silhouette"] = sil
K = 4
km = KMeans(K, n_init=100, random_state=RNG).fit(Z)
Ff["cluster"] = km.labels_
Ff["final_done"] = users["final_done"]
# name clusters by their profile (deterministic given the seed)
cm = Ff.groupby("cluster")[CL + ["final_done"]].mean()
nm = {}
nm[cm["sessions"].idxmax()] = "Intensive self-testers"
rest = cm.drop(index=list(nm))
nm[rest["final_done"].idxmax()] = "Planful submitters"
rest = rest.drop(index=rest["final_done"].idxmax())
nm[rest["exam_window_share"].idxmax()] = "Exam-only / last-minute"
rest = rest.drop(index=rest["exam_window_share"].idxmax())
nm[rest.index[0]] = "Passive content consumers"
Ff["cluster"] = Ff["cluster"].map(nm)
CORDER = ["Intensive self-testers", "Planful submitters", "Passive content consumers", "Exam-only / last-minute"]
prof = Ff.groupby("cluster")[CL + ["final_done"]].mean()
prof["n"] = Ff.groupby("cluster").size()
prof = prof.loc[CORDER]
zprof = pd.DataFrame(km.cluster_centers_, columns=CL).rename(index=nm).loc[CORDER]
labels = {"events": "Actions (log)", "sessions": "Sessions", "mean_sess_min": "Mean session length",
          "active_weeks_ratio": "Active-week ratio", "weekly_cv": "Weekly burstiness (CV)",
          "night_ratio": "Night-time share", "practice_share": "Practice-quiz share",
          "content_views": "Content views (log)", "self_monitoring": "Self-monitoring views",
          "synopsis_lead_days": "Synopsis lead time", "exam_window_share": "Exam-window share", "content_diversity": "Distinct items accessed",
          "practice_attempts": "Practice-quiz attempts", "time_on_task_h": "Time on task (h)",
          "active_days": "Active days", "gap_sd_days": "Inter-session gap SD", "synopsis_submitted": "Synopsis submitted",
          "forum_glossary": "Forum/glossary views", "scorm_mean_score": "SCORM score", "weekend_ratio": "Weekend share"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [2.3, 1]})
ax = axes[0]
zz = zprof.clip(-2, 2)
im = ax.imshow(zz.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(CL))); ax.set_xticklabels([labels[c] for c in CL], rotation=40, ha="right", fontsize=6.8)
ax.set_yticks(range(K)); ax.set_yticklabels([f"{c} (n={int(prof.loc[c,'n'])})" for c in CORDER], fontsize=7)
for i in range(K):
    for j in range(len(CL)):
        ax.text(j, i, f"{zprof.values[i, j]:+.1f}", ha="center", va="center", fontsize=6,
                color="white" if abs(zz.values[i, j]) > 1.1 else style.INK)
ax.grid(False); ax.set_title("a. K-Means centroids (z-scores, full term)")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.ax.tick_params(labelsize=6)
ax = axes[1]
rate = prof["final_done"] * 100
ax.barh(range(K), rate.values, color=[style.C1, style.C3, style.C2, style.C7], height=0.6)
for i, v in enumerate(rate.values):
    ax.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=7, color=style.INK)
ax.set_yticks(range(K)); ax.set_yticklabels([]); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False); ax.set_xlabel("% completed Final project"); ax.set_title("b. Completion by profile")
fig.tight_layout(); fig.savefig(FIG / "moodle_clusters.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
top = imp.head(10)[::-1]
ax.barh([labels.get(i, i) for i in top.index], top.values, color=style.C1, height=0.6)
ax.set_xlabel("Permutation importance (Δ AUC)"); ax.grid(axis="y", visible=False)
ax.set_title("RF importance, held-out folds (weeks 0-9)")
fig.tight_layout(); fig.savefig(FIG / "moodle_importance.png"); plt.close(fig)
prof.to_csv(OUT / "moodle_cluster_profiles.csv")
R["cluster_profiles"] = prof.round(3).to_dict(orient="index")
R["cluster_final_rate_chi2"] = float(stats.chi2_contingency(pd.crosstab(Ff["cluster"], Ff["final_done"]))[1])
Ff.to_csv(OUT / "moodle_features_full.csv")

(OUT / "moodle_results.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps({k: R[k] for k in ["users_kept", "final_done_n", "sessions", "ml", "moodle_silhouette", "synopsis_chi2"]}, indent=1))
print(prof.round(2).T)
print(comp.round(3).to_string())
