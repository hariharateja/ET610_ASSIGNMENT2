"""Builds report/ET610_LA_Assignment2_Report.pdf from figures + outputs/*.json.

Run after oulad_analysis.py and moodle_analysis.py:
    .venv/bin/python code/build_report.py
"""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FIG, OUT = ROOT / "figures", ROOT / "outputs"
(ROOT / "report").mkdir(exist_ok=True)
PDF = ROOT / "report" / "ET610_LA_Assignment2_Report.pdf"

# ---- team details: EDIT THESE ------------------------------------------
TEAM = [("Hari Hara Teja", "24b0977"), ("T Ganesh", "24b2250")]
REPO = "https://github.com/hariharateja/ET610_ASSIGNMENT2"
COURSE = "ET 610 Learning Analytics — Group Assignment 2"
# ------------------------------------------------------------------------

O = json.loads((OUT / "oulad_results.json").read_text())
M = json.loads((OUT / "moodle_results.json").read_text())
auc = pd.DataFrame(O["auc_curve"])


def a(week, s, m="RandomForest"):
    return auc[(auc.week == week) & (auc.set == s) & (auc.model == m)].auc.iloc[0]


VOL, REG, ALL = "Demographics + volume", "+ regularity", "+ regularity + assessment"
w8 = O["week8_metrics"]
h2 = O["H2_logit"]
oc = O["cleaning"]["outcome_counts"]
mml = M["ml"]
gt = {r["feature"]: r for r in M["group_tests"]}
cp = M["cluster_profiles"]
ocl = O["kmeans_clusters"]
syn = M["synopsis_table"]

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
FD = Path("/System/Library/Fonts/Supplemental")
for name, f in [("Arial", "Arial.ttf"), ("Arial-Bold", "Arial Bold.ttf"),
                ("Arial-Italic", "Arial Italic.ttf"), ("Arial-BoldItalic", "Arial Bold Italic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FD / f)))
for b, i, n in [(0, 0, "Arial"), (1, 0, "Arial-Bold"), (0, 1, "Arial-Italic"), (1, 1, "Arial-BoldItalic")]:
    addMapping("Arial", b, i, n)
ss = getSampleStyleSheet()
INK = colors.HexColor("#0b0b0b"); INK2 = colors.HexColor("#52514e"); RULE = colors.HexColor("#c3c2b7")
ACC = colors.HexColor("#2a78d6")
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Arial", fontSize=9.2, leading=12.2,
                      alignment=TA_JUSTIFY, textColor=INK, spaceAfter=4)
small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=9.8, textColor=INK2, alignment=TA_JUSTIFY)
cap = ParagraphStyle("c", parent=small, spaceBefore=1, spaceAfter=7)
h1 = ParagraphStyle("h1", parent=body, fontName="Arial-Bold", fontSize=12, leading=15,
                    spaceBefore=8, spaceAfter=4, textColor=INK, alignment=0)
h2s = ParagraphStyle("h2", parent=h1, fontSize=10, leading=13, spaceBefore=5, spaceAfter=2)
hyp = ParagraphStyle("hyp", parent=body, leftIndent=10, spaceAfter=3)
title = ParagraphStyle("t", parent=h1, fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=4)
sub = ParagraphStyle("st", parent=body, alignment=TA_CENTER, textColor=INK2, fontSize=9.5)
cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.6, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Arial-Bold")

W = A4[0] - 3.6 * cm
story = []
P = lambda t, s=body: story.append(Paragraph(t, s))


def fig(name, caption, width=W):
    from PIL import Image as PImage
    w, h = PImage.open(FIG / name).size
    story.append(KeepTogether([Image(str(FIG / name), width=width, height=width * h / w),
                               Paragraph(caption, cap)]))


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.6),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6), ("LEFTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE))
    t.setStyle(TableStyle(st))
    story.append(t)


f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
pm = lambda d: f"{d['mean']:.3f} ± {d['sd']:.3f}"

# ================================================================ title ==
P("Modelling Learner Behaviour Over Time:<br/>Engagement, Self-Regulation and At-Risk Prediction in OULAD and Moodle", title)
story.append(Spacer(1, 4))
team_rows = [["Team member", "Roll number"]] + [list(t) for t in TEAM]
tt = Table([[Paragraph(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(team_rows)],
           colWidths=[6 * cm, 4 * cm], hAlign="CENTER")
tt.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
story.append(tt)
story.append(Spacer(1, 4))
P(f"Code & notebooks: <link href='{REPO}' color='#2a78d6'>{REPO}</link> &nbsp;·&nbsp; Submitted 5 October 2026", sub)
story.append(Spacer(1, 6))
P(f"<b>Abstract.</b> We analyse two learning-analytics datasets. One is the Open University Learning Analytics "
  f"Dataset (OULAD; {O['cleaning']['enrolments_used']:,} enrolments, {O['cleaning']['vle_rows_used']/1e6:.1f} M "
  f"daily click records). The other is a {M['rows_raw']:,}-event Moodle log from a blended IT course "
  f"({M['users_kept']} learners). For each dataset we test hypotheses about early engagement, regularity, "
  f"self-monitoring and procrastination. We build weekly behavioural trajectories and evaluate supervised and "
  f"unsupervised models. In OULAD, a Random Forest trained on 2013 cohorts flags at-risk learners in the 2014 "
  f"cohorts with ROC-AUC {a(4, ALL):.2f} at week 4, {a(8, ALL):.2f} at week 8 and {a(26, ALL):.2f} at week 26. "
  f"Regularity features add up to +{(auc[(auc.set==REG)&(auc.model=='RandomForest')].set_index('week').auc - auc[(auc.set==VOL)&(auc.model=='RandomForest')].set_index('week').auc).max():.2f} AUC over click volume. "
  f"Submission latency, however, has no effect once score and submission rate are controlled for. In Moodle, "
  f"behaviour from the first ten weeks predicts completion of the final project (AUC {mml['RandomForest']['auc']['mean']:.2f}). "
  f"The strongest signals are early Synopsis submission and self-monitoring of grades and feedback, not raw "
  f"activity. K-Means separates four profiles whose completion rates range from "
  f"{min(v['final_done'] for v in cp.values())*100:.0f}% to {max(v['final_done'] for v in cp.values())*100:.0f}%.", body)

# ====================================================== 1. introduction ==
P("1. Introduction &amp; Hypotheses", h1)
P("Learning-management-system logs record what learners do and when they do it. Self-regulated learning theory "
  "(Zimmerman, 2002) holds that successful learners plan, monitor and adjust their study. Procrastination research "
  "(Steel, 2007) links delay to worse outcomes. Both theories predict that <i>how</i> and <i>when</i> a learner engages "
  "matters, not only <i>how much</i>. We ask three questions. <b>RQ1:</b> How early do the trajectories of "
  "successful and unsuccessful learners diverge? <b>RQ2:</b> Do timing behaviours (submission latency, "
  "regularity, self-monitoring) carry information beyond activity volume? <b>RQ3:</b> Can learners be segmented "
  "into interpretable behavioural profiles that differ in outcome?")
P("<b>OULAD hypotheses.</b>", h2s)
P("<b>O-H1 (early at-risk divergence).</b> Learners who later fail or withdraw already show lower and declining VLE "
  "engagement in the first weeks. Early clickstream features alone reach AUC ≥ 0.75 by week 4.", hyp)
P("<b>O-H2 (submission latency).</b> Later submission relative to the deadline is associated with failing or "
  "withdrawing, even after controlling for assessment score and submission rate.", hyp)
P("<b>O-H3 (regularity over volume).</b> Engagement regularity (active-day and active-week ratios, weekly "
  "burstiness, recency, activity diversity) adds predictive value beyond total clicks and demographics.", hyp)
P("<b>Moodle hypotheses.</b>", h2s)
P("<b>M-H1 (distributed, self-monitored engagement).</b> In weeks 0–9, learners who go on to submit the Final "
  "project log in on more distinct days and weeks, have more regular gaps between sessions, and check their own "
  "grades and feedback more often.", hyp)
P("<b>M-H2 (procrastination &amp; self-testing).</b> Learners who submit the first milestone (Synopsis) close to its "
  "deadline, or not at all, are less likely to complete the Final project. Completers also use practice quizzes more. "
  "(The deadline is inferred from the data, so we test how early learners submit, not whether they submit late.)", hyp)
P("<b>M-H3 (behavioural profiles).</b> Unsupervised clustering yields distinct, interpretable profiles "
  "(e.g. steady self-regulators vs. exam-window-only users) with different completion rates.", hyp)

# ========================================================= 2. data ==
P("2. Datasets &amp; Preprocessing", h1)
P("2.1 OULAD", h2s)
P(f"OULAD (Kuzilek et al., 2017) covers 22 presentations of 7 modules (2013–2014). It contains demographics, "
  f"registration dates, assessment deadlines and submissions, and daily click counts per VLE resource. "
  f"Cleaning steps: (i) we de-duplicated enrolments on (module, presentation, student). (ii) We removed "
  f"{O['cleaning']['unregistered_before_start']:,} enrolments that unregistered on or before day 0, because "
  f"they never had a chance to engage. This leaves {O['cleaning']['enrolments_used']:,} enrolments "
  f"(Pass {oc['Pass']:,}; Fail {oc['Fail']:,}; Withdrawn {oc['Withdrawn']:,}; Distinction {oc['Distinction']:,}). "
  f"(iii) We joined click records to activity types and aggregated them to one row per enrolment and day. "
  f"(iv) We excluded exam rows from the assessment features, since exam dates are often missing. Banked "
  f"(carried-over) results were also dropped. (v) Missing IMD bands and registration dates were imputed with the median. "
  f"The binary target is <b>at-risk = Fail or Withdrawn</b>. To mimic a real early-warning system, every feature at "
  f"cut-off day <i>c</i> uses only events dated before <i>c</i>. Learners who had already withdrawn by <i>c</i> are "
  f"not predicted at that cut-off.")
P("2.2 Moodle interaction log", h2s)
P(f"The Moodle export has {M['rows_raw']:,} events for course IT-2507 (23 Oct 2025 – 29 Jul 2026). Each event "
  f"records a user, a timestamp, an event context, a component, a content code and a description. "
  f"<b>Clean-up:</b> (i) for {M['user_id_repaired']:,} SCORM events the <i>User ID</i> column was truncated "
  f"(e.g. <font face='Courier'>e id '3</font>). We recovered the true id from the description text with a regular "
  f"expression. (ii) We parsed the dd/mm/yy timestamps and checked for exact duplicates (none found). (iii) We "
  f"dropped {M['users_raw'] - M['users_kept']} accounts with fewer than 30 events, leaving {M['users_kept']} learners. "
  f"(iv) We excluded <i>Quiz attempt auto-saved</i> events (37k rows) from action counts, because they are "
  f"system-generated during quiz attempts. They are kept for session timing. "
  f"<b>Sessionisation:</b> events from the same user less than 30 minutes apart belong to one session "
  f"(Kovanović et al., 2015). This gives {M['sessions']['n']:,} sessions (median {M['sessions']['median_events']:.0f} "
  f"events, {M['sessions']['median_dur_min']:.1f} min). A single-event session is credited with 1 minute of time "
  f"on task. "
  f"<b>Outcome &amp; leakage control:</b> the log contains no grades. Our outcome is therefore <b>submission of the "
  f"Final course-project assignment</b> ({M['final_done_n']} of {M['users_kept']} learners, "
  f"{M['final_done_n']/M['users_kept']*100:.0f}%). Predictive features use only events before 29 Dec 2025, the day of "
  f"the first Final submission. That gives an early window of weeks 0–9, from which all Final-assignment events "
  f"are removed. The Synopsis deadline is inferred as end of day 30 Nov 2025, the date of the last Synopsis submission. "
  f"Because of this, no submission can be late by construction (the closest one came 0.2 days before the proxy deadline).")

# ============================================== 3. features ==
P("3. Feature Extraction &amp; Engineering", h1)
table([
    ["Construct", "OULAD features (computed before cut-off <i>c</i>)", "Moodle features (weeks 0–9)"],
    ["Volume", "log total clicks; log pre-start clicks", "log actions; log content views (book, page, SCORM, video, PPT…)"],
    ["Regularity / time management",
     "active days; active-day ratio (= active days / <i>c</i>); active-week ratio; weekly coefficient of variation "
     "(burstiness); days since last login; recent trend (log clicks of last 2 weeks vs. earlier)",
     "sessions; active days; active-week ratio; weekly CV; SD of inter-session gaps; night (00–06 h) and weekend share"],
    ["Time on task", "—", "total sessionised hours; mean session length (min)"],
    ["Breadth / strategy", "Shannon entropy of clicks across 20 activity types",
     "distinct items accessed; practice-quiz attempts &amp; share of actions; forum/glossary views"],
    ["Submission behaviour", "assessments due; submission rate; mean latency (days submitted − deadline); mean score",
     "Synopsis submitted (0/1); Synopsis lead time (days before deadline; −5 if not submitted)"],
    ["Self-monitoring", "—", "views of own grade report, assignment feedback and submission status"],
    ["Performance / context", "prior attempts, credits, IMD band, education, age, gender, disability, registration day",
     "mean SCORM self-check score"],
], [2.9 * cm, 7.2 * cm, W - 10.1 * cm])
P("Table 1. Engineered features, grouped by behavioural construct. Clustering also uses an <i>exam-window share</i> "
  "feature: the share of a learner's actions in weeks 14–15 (26 Jan – 8 Feb 2026, when the continuous-evaluation "
  "quizzes ran).", cap)

# ================================================ 4. temporal ==
P("4. Temporal Behavioural Visualisations", h1)
P("4.1 OULAD", h2s)
fig("oulad_weekly_clicks.png",
    "Figure 1. (a) Mean weekly VLE clicks per learner by final outcome (bands = 95% CI). (b) Share of learners active "
    "in each week. Week 0 is the module start.")
e = {r["week"]: r for r in O["H1_weekly_divergence"]}
P(f"All four groups engage strongly in the first fortnight, but the curves separate almost immediately (Fig. 1). "
  f"In week 0 the median at-risk learner logs {e[0]['median_at_risk']:.0f} clicks, against "
  f"{e[0]['median_success']:.0f} for successful learners (rank-biserial r = {e[0]['rank_biserial']:.2f}). By week 4 the "
  f"medians are {e[4]['median_at_risk']:.0f} vs. {e[4]['median_success']:.0f} (r = {e[4]['rank_biserial']:.2f}). "
  f"By week 6 the median at-risk learner is no longer active at all (r = {e[6]['rank_biserial']:.2f}). Two "
  f"patterns stand out. Failing learners do not drop out: their activity stays low and gently declining. Withdrawn "
  f"learners decay steadily. Successful learners follow a periodic rhythm, with peaks around weeks 2, 20 and 30 "
  f"(assessment dates differ between modules, so these are pooled peaks). Both Distinction and Pass learners dip in week 11, which looks like a scheduled break "
  f"rather than disengagement.")
fig("oulad_latency.png",
    "Figure 2. (a) Share of each assessment submitted on or before its deadline, in chronological order within the module. "
    "(b) Final-outcome mix by a learner's mean submission latency (submitted assessments only).")
bt = O["H2_bucket_table"]
risk = {k: bt["Fail"][k] + bt["Withdrawn"][k] for k in bt["Fail"]}
P(f"Submission punctuality (Fig. 2a) shows the same early separation. On the first assessment, at-risk groups are "
  f"already 26–42 points behind. By the third assessment fewer than 20% of their assessments are submitted on time, partly "
  f"because many are never submitted. Mean latency is clearly graded with outcome (Fig. 2b): "
  f"{risk['Early (>3d)']:.0f}% of learners who submit more than 3 days early are at risk, against "
  f"{risk['Late >7d']:.0f}% of those who are on average more than a week late (Spearman ρ = "
  f"{O['H2_spearman_late_vs_risk']:.2f}).")
P("4.2 Moodle", h2s)
fig("moodle_weekly.png",
    "Figure 3. Weekly actions (a) and sessionised time-on-task (b) for learners who did and did not complete the Final "
    "project (bands = 95% CI). Dashed lines mark the Synopsis deadline, the opening of Final submissions and the "
    "continuous-evaluation (CE) quiz window.")
P(f"Completers are about twice as active as non-completers through the first ten weeks (Fig. 3). In weeks 2–5 they "
  f"spend roughly 0.5 h per week on the platform, against about 0.2 h. About a week after the Synopsis deadline "
  f"(week 7), completers produce a second burst of activity. The gap then closes. In the CE quiz window (weeks "
  f"14–15) both groups converge on the same exam-driven peak, and after week 16 their trajectories are "
  f"indistinguishable. In short, the groups differ in when they engage, not in their exam-time effort: "
  f"non-completers engage mainly when assessment forces them to.")
fig("moodle_mix_synopsis.png",
    "Figure 4. (a) Weekly composition of all learner actions by component (auto-saves excluded). (b) Cumulative share of "
    "each group that had submitted the Synopsis, by days before its deadline (read left to right as time passes).")
P(f"Figure 4a shows how activity moves through the course: content reading in weeks 3–13, assignment work around "
  f"each milestone, and a sharp switch to quizzes in weeks 14–15. Figure 4b shows the strongest temporal signal. "
  f"About a third (33%) of eventual completers submitted the Synopsis more than 24 days early, and about 90% submitted it "
  f"before the deadline. Only about 11% of non-completers ever submitted it.")

# ================================================== 5. ML ==
P("5. Machine Learning &amp; Results", h1)
P("5.1 OULAD — early-warning classification (O-H1, O-H3) and trajectory clustering", h2s)
P(f"<b>Setup.</b> We compare L2 logistic regression (standardised, class-balanced) with a Random Forest (300 trees, "
  f"min leaf 10, balanced class weights; Breiman, 2001). Both use scikit-learn (Pedregosa et al., 2011). "
  f"Validation is <b>temporal</b>: we train on all 2013 presentations and test on all 2014 presentations, so the "
  f"model always predicts a later cohort. We repeat this at nine cut-offs (weeks 1–26), each with three nested "
  f"feature sets (Table 1). We also ran 5-fold stratified cross-validation within the 2013 data at week 8 "
  f"(RF AUC {O['week8_rf_cv_auc']['mean']:.3f} ± {O['week8_rf_cv_auc']['sd']:.3f}). It is close to the temporal "
  f"test score, so the model generalises across years.")
fig("oulad_ml.png",
    "Figure 5. (a) Test ROC-AUC on 2014 cohorts by prediction cut-off and feature set (solid = Random Forest, dotted = "
    "logistic regression; dashed line = 0.75 target). (b) Permutation importance (drop in AUC) for the week-8 RF.")
table([
    ["Model (week 8, 2014 test, n = " + f"{w8['RandomForest']['n_test']:,})", "ROC-AUC", "F1", "Precision", "Recall", "Balanced acc."],
    ["Logistic regression (all features)", f3(w8["LogReg"]["auc"]), f3(w8["LogReg"]["f1"]), f3(w8["LogReg"]["precision"]),
     f3(w8["LogReg"]["recall"]), f3(w8["LogReg"]["balanced_acc"])],
    ["Random Forest (all features)", f3(w8["RandomForest"]["auc"]), f3(w8["RandomForest"]["f1"]),
     f3(w8["RandomForest"]["precision"]), f3(w8["RandomForest"]["recall"]), f3(w8["RandomForest"]["balanced_acc"])],
    ["RF, demographics + volume only", f3(a(8, VOL)), "", "", "", ""],
    ["RF, + regularity (no assessment)", f3(a(8, REG)), "", "", "", ""],
], [6.4 * cm] + [(W - 6.4 * cm) / 5] * 5)
P(f"Table 2. Week-8 early-warning performance (threshold 0.5; at-risk prevalence among learners still enrolled = "
  f"{auc[auc.week==8].prev_test.iloc[0]*100:.0f}%).", cap)
P(f"<b>Results.</b> Prediction accuracy rises steadily with time (Fig. 5a). With demographics and click volume only, "
  f"the RF reaches AUC {a(4, VOL):.2f} at week 4 and passes 0.75 only around week 16. Adding regularity features "
  f"lifts AUC to {a(4, REG):.2f} at week 4, {a(8, REG):.2f} at week 8 and {a(26, REG):.2f} at week 26. "
  f"Adding assessment behaviour gives {a(4, ALL):.2f}, {a(8, ALL):.2f} and {a(26, ALL):.2f}. "
  f"At week 8 the RF identifies {w8['RandomForest']['recall']*100:.0f}% of at-risk learners at "
  f"{w8['RandomForest']['precision']*100:.0f}% precision. Logistic regression is nearly as good, which suggests "
  f"the signal is largely monotonic. The most important week-8 features (Fig. 5b) are mean score, submission rate "
  f"and latency, followed by prior education, active-week ratio, recency and activity diversity. Total clicks matter much less once "
  f"regularity is known.")
P(f"<b>O-H2 test.</b> We fitted a logistic regression of at-risk status on standardised mean latency, mean score and "
  f"submission rate (n = {h2['n']:,} enrolments with ≥ 1 submission). Latency has <b>no independent effect</b> "
  f"(OR per SD = {h2['mean_late_c']['OR_per_SD']:.2f}, 95% CI {h2['mean_late_c']['ci_lo']:.2f}–{h2['mean_late_c']['ci_hi']:.2f}, "
  f"p = {h2['mean_late_c']['p']:.2f}). Score (OR {h2['mean_score']['OR_per_SD']:.2f}) and submission rate "
  f"(OR {h2['submit_rate']['OR_per_SD']:.2f}) are both highly significant (p &lt; 0.001).")
fig("oulad_clusters.png",
    "Figure 6. (a) Centroid trajectories from K-Means (k = 4) on log weekly clicks over weeks 0–19. (b) Final-outcome mix "
    "within each cluster.")
cl = ocl["outcomes_pct"]
P(f"<b>Clustering.</b> We ran K-Means on log-transformed 20-week click trajectories. Silhouette favours k = 2 "
  f"({O['kmeans_silhouette']['2']:.2f}), a simple engaged/disengaged split. We report k = 4 "
  f"(silhouette {O['kmeans_silhouette']['4']:.2f}) because it separates the <i>timing</i> of disengagement. "
  f"The four profiles are <i>high &amp; sustained</i> ({cl['High & sustained']['Fail'] + cl['High & sustained']['Withdrawn']:.0f}% at risk), "
  f"<i>moderate &amp; steady</i> ({cl['Moderate & steady']['Fail'] + cl['Moderate & steady']['Withdrawn']:.0f}%), "
  f"<i>early drop-off</i> ({cl['Early drop-off']['Fail'] + cl['Early drop-off']['Withdrawn']:.0f}%) and "
  f"<i>minimal/absent</i> ({cl['Minimal / absent']['Fail'] + cl['Minimal / absent']['Withdrawn']:.0f}%) (Fig. 6). "
  f"The early drop-off cluster starts at a similar level to the moderate cluster but falls away after week 3. "
  f"These are the learners an intervention could still reach.")

P("5.2 Moodle — completion prediction (M-H1, M-H2) and profile discovery (M-H3)", h2s)
P(f"<b>Setup.</b> With n = {M['users_kept']}, a single train/test split would be unstable. We therefore use "
  f"<b>repeated stratified 5-fold cross-validation (10 repeats, 50 folds)</b> and report mean ± SD. The models are "
  f"an L2 logistic regression (C = 0.3) and a Random Forest (400 trees, min leaf 3), both class-balanced. Two "
  f"baselines put the results in context: logistic regression on action count alone, and an RF without any "
  f"submission-related or self-monitoring features. Univariate group differences use Mann–Whitney U tests with "
  f"Holm correction across {len(M['group_tests'])} features.")
table([
    ["Model (weeks 0–9 features → Final submitted)", "ROC-AUC", "F1", "Precision", "Recall", "Balanced acc."],
    ["Logistic regression (all 19 features)", pm(mml["LogReg (L2)"]["auc"]), pm(mml["LogReg (L2)"]["f1"]),
     pm(mml["LogReg (L2)"]["precision"]), pm(mml["LogReg (L2)"]["recall"]), pm(mml["LogReg (L2)"]["balanced_acc"])],
    ["Random Forest (all 19 features)", pm(mml["RandomForest"]["auc"]), pm(mml["RandomForest"]["f1"]),
     pm(mml["RandomForest"]["precision"]), pm(mml["RandomForest"]["recall"]), pm(mml["RandomForest"]["balanced_acc"])],
    ["RF, behaviour only (no Synopsis / self-monitoring)", pm(mml["RandomForest (behaviour only)"]["auc"]), "", "", "", ""],
    ["Baseline LR, action count only", pm(mml["Baseline LogReg (events only)"]["auc"]), "", "", "", ""],
], [5.6 * cm] + [(W - 5.6 * cm) / 5] * 5)
P("Table 3. Moodle cross-validated performance (mean ± SD over 50 folds).", cap)
imp_rows = []
for k in ["synopsis_lead_days", "self_monitoring", "active_days", "gap_sd_days", "active_weeks_ratio", "sessions",
          "time_on_task_h", "practice_share", "scorm_mean_score"]:
    r = gt[k]
    nm = {"synopsis_lead_days": "Synopsis lead time (days)", "self_monitoring": "Self-monitoring views",
          "active_days": "Active days", "gap_sd_days": "Inter-session gap SD (days)",
          "active_weeks_ratio": "Active-week ratio", "sessions": "Sessions", "time_on_task_h": "Time on task (h)",
          "practice_share": "Practice-quiz share", "scorm_mean_score": "SCORM self-check score"}[k]
    imp_rows.append([nm, f"{r['median_completed']:.2f}", f"{r['median_not']:.2f}", f"{r['r_rb']:+.2f}",
                     "&lt; 0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.2f}"])
fig("moodle_importance.png",
    "Figure 7. Random-Forest permutation importance, measured on held-out folds. Importance is costly to compute, so it "
    "uses 20 folds (5-fold × 4 repeats) rather than the 50 folds (5-fold × 10 repeats) behind Table 3.", width=W * 0.5)
table([["Feature (weeks 0–9)", "Median: completed", "Median: not", "Rank-biserial r", "Holm p"]] + imp_rows,
      [5.6 * cm] + [(W - 5.6 * cm) / 4] * 4)
P("Table 4. Selected univariate comparisons between completers (n = 58) and non-completers (n = 103). Note: learners "
  "who never submitted the Synopsis are assigned a lead time of −5 days. The non-completer median of −5.00 is therefore "
  "this imputed value (most non-completers never submitted), not an observed lead time.", cap)
P(f"<b>Results.</b> Both models predict Final completion from week-0–9 behaviour with AUC ≈ "
  f"{mml['RandomForest']['auc']['mean']:.2f} (Table 3). However, Fig. 7 and the behaviour-only ablation show that "
  f"the signal is concentrated in the Synopsis features (whether and how early it was submitted) and in self-monitoring. Removing them drops AUC to "
  f"{mml['RandomForest (behaviour only)']['auc']['mean']:.2f}, barely above the action-count baseline "
  f"({mml['Baseline LogReg (events only)']['auc']['mean']:.2f}). The univariate tests (Table 4) still support "
  f"M-H1. Completers were active on twice as many days, had more regular gaps between sessions "
  f"(r = {gt['gap_sd_days']['r_rb']:+.2f}) and checked their grades and feedback far more often "
  f"(r = {gt['self_monitoring']['r_rb']:+.2f}). M-H2 is split. We grouped Synopsis timing into four categories and "
  f"counted Final completers in each: not submitted, {syn['Not submitted']['1']} of {sum(syn['Not submitted'].values())}; "
  f"less than 1 day before the deadline, {syn['<1 day before']['1']} of {sum(syn['<1 day before'].values())}; "
  f"1–7 days before, {syn['1-7 days']['1']} of {sum(syn['1-7 days'].values())}; more than 7 days before, "
  f"{syn['>7 days']['1']} of {sum(syn['>7 days'].values())}. The 4 × 2 table gives "
  f"χ²({M['synopsis_chi2']['dof']}) = {M['synopsis_chi2']['chi2']:.0f}, p &lt; 0.001. Practice-quiz use, by "
  f"contrast, did not differ between groups (Holm p = {gt['practice_share']['p_holm']:.2f}).")
fig("moodle_clusters.png",
    "Figure 8. (a) K-Means (k = 4) centroids on 11 standardised whole-term features (red = above cohort mean). "
    "(b) Share of each profile that completed the Final project. The Synopsis lead-time column includes the −5-day value assigned to non-submitters. Features cover the whole term, including the period after the outcome, so the profiles describe behaviour and do not predict it.")
cpn = {k: v for k, v in cp.items()}
P(f"<b>Profiles (M-H3).</b> K-Means on 11 standardised whole-term features (Fig. 8) produced four interpretable "
  f"groups. <i>Intensive self-testers</i> (n = {cpn['Intensive self-testers']['n']}) have the most sessions, the "
  f"most active weeks and the heaviest practice-quiz use ({cpn['Intensive self-testers']['final_done']*100:.0f}% "
  f"completed). <i>Planful submitters</i> (n = {cpn['Planful submitters']['n']}) are only moderately active but "
  f"submit very early ({cpn['Planful submitters']['synopsis_lead_days']:.0f} days ahead) and check feedback "
  f"({cpn['Planful submitters']['final_done']*100:.0f}% completed). <i>Passive content consumers</i> "
  f"(n = {cpn['Passive content consumers']['n']}) read as much material as anyone, in longer sessions, but rarely "
  f"submit or self-monitor; only {cpn['Passive content consumers']['final_done']*100:.0f}% completed. <i>Exam-only / last-minute</i> "
  f"learners (n = {cpn['Exam-only / last-minute']['n']}) put {cpn['Exam-only / last-minute']['exam_window_share']*100:.0f}% "
  f"of all their activity into the two CE weeks; only {cpn['Exam-only / last-minute']['final_done']*100:.0f}% completed the Final. "
  f"Completion differs strongly across clusters (χ² p &lt; 0.001). Silhouette values are low, however "
  f"(between {min(M['moodle_silhouette'].values()):.2f} and {max(M['moodle_silhouette'].values()):.2f} for every k from 2 to 7). The profiles are best read as regions of a "
  f"continuum rather than sharply separated types.")

# ============================================ 6. discussion ==
P("6. Discussion &amp; Conclusion", h1)
table([
    ["Hypothesis", "Verdict", "Key evidence"],
    ["O-H1 early divergence", "Supported (partly)",
     f"Trajectories separate from week 0 (r = {e[0]['rank_biserial']:.2f}, rising to {e[6]['rank_biserial']:.2f} by week 6). "
     f"Without assessment data (demographics + clickstream), AUC at week 4 = {a(4, REG):.2f} (< 0.75); 0.75 is reached by week 8 ({a(8, REG):.2f}), "
     f"or by week 4 once assessment data are added ({a(4, ALL):.2f})."],
    ["O-H2 latency", "Not supported (independently)",
     f"Strong bivariate gradient ({risk['Early (>3d)']:.0f}% → {risk['Late >7d']:.0f}% at risk), but OR = "
     f"{h2['mean_late_c']['OR_per_SD']:.2f} after controlling for score and submission rate."],
    ["O-H3 regularity", "Supported",
     f"+{a(8, REG) - a(8, VOL):.2f} AUC at week 8 and +{a(16, REG) - a(16, VOL):.2f} at week 16 over volume. "
     f"Active-week ratio and recency outrank total clicks."],
    ["M-H1 distributed / self-monitoring", "Supported",
     "Active days, session regularity and self-monitoring all differ (Holm p < 0.001). Self-monitoring is the "
     "top predictor."],
    ["M-H2 procrastination / self-testing", "Partly supported",
     f"Early Synopsis submission strongly predicts completion and non-submission predicts failure to complete "
     f"(χ²(3) p < 0.001). Lateness cannot be tested (inferred deadline). Practice-quiz use does not differ "
     f"(p = {gt['practice_share']['p_holm']:.2f})."],
    ["M-H3 profiles", "Supported (weak structure)",
     f"Four interpretable profiles; completion ranges from {min(v['final_done'] for v in cp.values())*100:.0f}% "
     f"to {max(v['final_done'] for v in cp.values())*100:.0f}%. Silhouette ≈ {M['moodle_silhouette']['4']:.2f}."],
], [3.6 * cm, 2.9 * cm, W - 6.5 * cm])
P("Table 5. Summary of hypothesis tests.", cap)
P("<b>Interpretation.</b> The two datasets tell the same story. <i>When</i> and <i>how regularly</i> learners "
  "engage reveals more than how much they click, and the most informative behaviours involve meeting early "
  "milestones and monitoring one's own progress. These are the behaviours that self-regulated learning theory "
  "points to. The OULAD latency result is a useful caution. Late submission looks like a risk factor, but it adds "
  "nothing once the model knows whether the learner submits at all and how well they score. Lateness appears to be "
  "a symptom of the same underlying disengagement, not a separate cause.")
P("<b>Implications for course design and automated intervention.</b> (1) <i>Intervene early, and use assessment "
  "data.</i> An OULAD-style alert at week 4 that combines regularity with the first assessment already reaches "
  f"AUC ≈ {a(4, ALL):.2f}. Making the first assessment small and early therefore gives the alert system data as "
  "well as giving learners practice. (2) <i>Monitor regularity, not volume.</i> Dashboards should show active weeks "
  "and days since the last login. A learner with many clicks crammed into one week is not on track. (3) "
  "<i>Treat the first milestone as a sensor.</i> In Moodle, not submitting the Synopsis identified 91 of 103 eventual non-completers by 30 Nov, while flagging only 5 completers, "
  "four weeks before the Final opened. A deadline-plus-48-hours nudge is a cheap, "
  "explainable trigger. (4) <i>Prompt self-monitoring.</i> Completers checked feedback and grade reports far more "
  "often. Pushing feedback to learners and adding progress widgets may help the passive and exam-only groups adopt "
  "the habit. Whether this would change outcomes is causal and should be tested with an A/B design.")
P("<b>Limitations.</b> Our analyses are correlational. The Moodle outcome is project completion, not a grade, and "
  "the cohort is a single course of 161 learners. Synopsis and self-monitoring features are close in kind to the "
  "outcome itself, and the Synopsis deadline is inferred rather than taken from the course calendar. Time-on-task depends on the 30-minute session threshold. OULAD click counts do not record "
  "dwell time. Future work could add sequence models (e.g. HMMs over weekly states), calibrate the alert "
  "thresholds to intervention capacity, and audit fairness across IMD and disability groups before deployment.")

P("References", h2s)
for r in [
    "Breiman, L. (2001). Random forests. <i>Machine Learning</i>, 45(1), 5–32.",
    "Kovanović, V., Gašević, D., Dawson, S., Joksimović, S., Baker, R. S., &amp; Hatala, M. (2015). Penetrating the black box "
    "of time-on-task estimation. <i>Proc. LAK '15</i>, 184–193.",
    "Kuzilek, J., Hlosta, M., &amp; Zdrahal, Z. (2017). Open University Learning Analytics dataset. <i>Scientific Data</i>, 4, 170171.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. <i>JMLR</i>, 12, 2825–2830.",
    "Steel, P. (2007). The nature of procrastination. <i>Psychological Bulletin</i>, 133(1), 65–94.",
    "Zimmerman, B. J. (2002). Becoming a self-regulated learner: An overview. <i>Theory Into Practice</i>, 41(2), 64–70.",
]:
    P(r, small)
P(f"<b>Appendix — code.</b> All analysis is reproducible from <link href='{REPO}' color='#2a78d6'>{REPO}</link>: "
  "<font face='Courier'>code/oulad_analysis.py</font> (OULAD pipeline, ≈ 1 min), "
  "<font face='Courier'>code/moodle_analysis.py</font> (Moodle pipeline), <font face='Courier'>code/build_report.py</font> "
  "(this PDF); intermediate results are in <font face='Courier'>outputs/</font>.", small)


def on_page(c, doc):
    c.saveState()
    c.setFont("Arial", 7.5); c.setFillColor(INK2)
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"{doc.page}")
    c.restoreState()


doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                        title="Modelling Learner Behaviour Over Time", author="ET610 Group")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
from pypdf import PdfReader  # noqa: E402
print(PDF, "pages:", len(PdfReader(str(PDF)).pages))
