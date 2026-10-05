# ET610 Learning Analytics — Group Assignment 2

Temporal behaviour analysis and ML on **OULAD** and a **Moodle** interaction log.

## Layout
- `code/oulad_analysis.py` — OULAD cleaning, early-warning features at 9 cut-offs, temporal figures, LogReg/Random Forest (train 2013 → test 2014), K-Means trajectory clustering
- `code/moodle_analysis.py` — Moodle cleaning (user-id repair), 30-min sessionisation, features, group tests, LogReg/RF with repeated CV, K-Means profiles
- `code/build_report.py` — builds `report/ET610_LA_Assignment2_Report.pdf` (edit `TEAM` and `REPO` at the top)
- `figures/`, `outputs/` — generated figures and result files (JSON/CSV)

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/oulad_analysis.py
.venv/bin/python code/moodle_analysis.py
.venv/bin/python code/build_report.py
```
