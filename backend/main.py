import os
import re
import json
import traceback
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = os.getenv(
    "ER_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "CLAUDE Dataset with ML.csv"),
)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
WAIT_ALERT_THRESHOLD_MINUTES = 150  # 2.5 hours

# ── Load & prep data ─────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH, low_memory=False)

# Normalise date column
df_raw["date"] = pd.to_datetime(df_raw["date"], dayfirst=True, errors="coerce")

# Ensure numeric columns are numeric
_numeric_cols = [
    "patient_waittime",
    "AI_Waiting_Time_Minutes",
    "Treatment_Duration_Minutes",
    "patient_age",
    "patient_sat_score",
    "AI_Satisfaction_Score",
    "Triage_Level",
    "Clinical_Risk_Score",
    "Resource_Intensity_Score",
    "Admission_Probability",
    "Estimated_Cost_USD",
    "Arrival_Hour",
    "Is_Weekend",
]
for col in _numeric_cols:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Boolean-ish columns
if "patient_admin_flag" in df_raw.columns:
    df_raw["patient_admin_flag"] = df_raw["patient_admin_flag"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )

print(f"[startup] Loaded {len(df_raw):,} rows × {len(df_raw.columns)} cols")
print(f"[startup] Date range: {df_raw['date'].min()} → {df_raw['date'].max()}")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Hospital ER NL Query API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anthropic client ─────────────────────────────────────────────────────────
_client: Optional[anthropic.Anthropic] = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")
        _client = anthropic.Anthropic(api_key=key)
    return _client

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data analyst assistant for a UK National Health Service (NHS) Emergency Room (A&E) department.

DOMAIN KNOWLEDGE:
- The NHS 4-hour target: 95% of patients should be seen, treated, and either admitted or discharged within 4 hours (240 minutes) of arrival.
- Triage levels (1–5): 1 = most critical (Resuscitation), 2 = Emergency, 3 = Urgent, 4 = Semi-urgent, 5 = Non-urgent.
- Wait time is measured in minutes from arrival to first clinical contact.
- Departments include Walk-In / Self-Referred, General Practice, physiotherapy, orthopedics, neurology, cardiology, gastroenterology, renal, and others.
- A wait time alert threshold is 150 minutes (2.5 hours) at the department level.

DATASET COLUMNS (exact names — use these verbatim):
- Source Row Number : original row index
- patient_id        : unique patient identifier
- patient_first_inital : first initial of patient first name
- patient_last_name : patient last name
- date              : datetime of arrival (parsed as datetime64)
- Arrival_DayOfWeek : day of week string ('Monday', 'Tuesday', …)
- Is_Weekend        : 1 if weekend, 0 otherwise
- Time              : arrival time string (HH:MM:SS)
- Arrival_Hour      : integer hour of arrival (0–23)
- patient_gender    : 'M' or 'F'
- patient_age       : integer age in years
- patient_sat_score : patient satisfaction score (1–10, may be null)
- patient_race      : racial/ethnic category string
- patient_admin_flag: True/False — whether the patient has an administrative flag
- patient_waittime  : wait time in minutes (primary wait-time column)
- department_referral : department the patient was referred to
- Insurance_Type    : 'Public', 'Private', or 'None'
- Estimated_Cost_USD: estimated treatment cost in USD
- AI_Satisfaction_Score : AI-predicted satisfaction (float)
- AI_Age            : AI-estimated age
- Triage_Level      : triage score 1–5
- AI_Waiting_Time_Minutes : AI-predicted wait time
- Treatment_Duration_Minutes : actual treatment duration
- Resource_Intensity_Score : resource usage score (float)
- Clinical_Risk_Score : clinical risk score (float)
- Admitted          : 'Yes' / 'No' — whether the patient was admitted
- Admission_Probability : model probability of admission (0–1)
- Predicted_Admission : 0/1 binary predicted admission

TASK:
Given a natural-language question about this dataset, generate a single Python code block that:
1. Uses the pre-loaded pandas DataFrame named `df` (already loaded, do NOT re-read any file).
2. Produces a variable called `result` which is one of:
   - A scalar (int/float/string) for single-value answers
   - A pandas Series or DataFrame for tabular/chart data
3. Also produces a variable called `chart_type` (string): one of "bar", "line", "pie", "table", "scalar". Choose based on what best visualises the result.
4. Also produces a variable called `chart_title` (string): a short descriptive title for the chart.
5. Handles NaN/null values gracefully (use .dropna(), fillna(), or .notna() as appropriate).
6. Does NOT import any libraries — pandas (pd) and numpy (np) are already available.
7. Does NOT modify `df` in place — only read from it.
8. Is safe: no file I/O, no exec/eval, no os/sys calls.

Return ONLY the Python code block, no explanation, no markdown fences.
"""

# ── Safe code execution ──────────────────────────────────────────────────────
_FORBIDDEN = re.compile(
    r"\b(import|__import__|open|exec|eval|os\.|sys\.|subprocess|shutil|pathlib|builtins)\b"
)

def safe_exec(code: str, df: pd.DataFrame) -> dict:
    """Execute LLM-generated pandas code in a restricted namespace."""
    if _FORBIDDEN.search(code):
        raise ValueError("Generated code contains forbidden operations.")

    namespace: dict[str, Any] = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<llm_code>", "exec"), namespace)  # noqa: S102

    result = namespace.get("result")
    chart_type = namespace.get("chart_type", "scalar")
    chart_title = namespace.get("chart_title", "Result")

    return {"result": result, "chart_type": chart_type, "chart_title": chart_title}


def serialise_result(result: Any) -> dict:
    """Convert pandas/numpy result to JSON-serialisable chart-friendly dict."""
    if result is None:
        return {"type": "scalar", "value": None}

    if isinstance(result, (int, float, np.integer, np.floating)):
        return {"type": "scalar", "value": float(result)}

    if isinstance(result, str):
        return {"type": "scalar", "value": result}

    if isinstance(result, pd.Series):
        result = result.dropna()
        # Convert index to str for JSON safety
        labels = [str(i) for i in result.index.tolist()]
        values = result.tolist()
        values = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in values]
        return {"type": "series", "labels": labels, "values": values}

    if isinstance(result, pd.DataFrame):
        result = result.copy()
        # Replace NaN with None for JSON
        result = result.where(pd.notnull(result), other=None)
        # Convert datetime columns to strings
        for col in result.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
            result[col] = result[col].astype(str)
        return {
            "type": "dataframe",
            "columns": result.columns.tolist(),
            "rows": result.head(500).values.tolist(),
        }

    # Fallback
    return {"type": "scalar", "value": str(result)}


# ── Pydantic models ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    chart_type: str
    chart_title: str
    data: dict
    code: str


# ── /query endpoint ──────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client = get_client()

    # Step 1: Ask Claude to generate pandas code
    code_response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}"}],
    )
    raw_code = code_response.content[0].text.strip()

    # Strip markdown fences if present
    raw_code = re.sub(r"^```(?:python)?\n?", "", raw_code)
    raw_code = re.sub(r"\n?```$", "", raw_code)

    # Step 2: Execute code
    try:
        exec_result = safe_exec(raw_code, df_raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Code execution error: {exc}\n\nGenerated code:\n{raw_code}",
        )

    result_obj = exec_result["result"]
    chart_type = exec_result["chart_type"]
    chart_title = exec_result["chart_title"]
    serialised = serialise_result(result_obj)

    # Step 3: Ask Claude for a natural language answer
    # Build a compact summary of the result for the answer prompt
    if serialised["type"] == "scalar":
        result_summary = f"The result is: {serialised['value']}"
    elif serialised["type"] == "series":
        top = list(zip(serialised["labels"][:10], serialised["values"][:10]))
        result_summary = f"Top results: {top}"
    else:  # dataframe
        result_summary = f"Returned {len(serialised.get('rows', []))} rows with columns {serialised.get('columns', [])}"

    answer_response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=400,
        system=(
            "You are a concise ER analyst. Given a question and its numerical result, "
            "provide a clear 1-3 sentence plain-English answer. Include specific numbers. "
            "Mention NHS context where relevant (e.g. 4-hour target). Be direct."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nResult: {result_summary}",
            }
        ],
    )
    answer_text = answer_response.content[0].text.strip()

    return QueryResponse(
        question=question,
        answer=answer_text,
        chart_type=chart_type,
        chart_title=chart_title,
        data={**serialised, "chart_type": chart_type, "chart_title": chart_title},
        code=raw_code,
    )


# ── /patients endpoint ───────────────────────────────────────────────────────
@app.get("/patients")
def get_patients(
    department: Optional[str] = Query(None),
    max_wait: Optional[float] = Query(None, description="Max wait time in minutes"),
    min_wait: Optional[float] = Query(None, description="Min wait time in minutes"),
    triage: Optional[int] = Query(None),
    admitted: Optional[str] = Query(None, description="'Yes' or 'No'"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    filtered = df_raw.copy()

    if department:
        filtered = filtered[
            filtered["department_referral"].str.lower() == department.lower()
        ]
    if max_wait is not None:
        filtered = filtered[filtered["patient_waittime"] <= max_wait]
    if min_wait is not None:
        filtered = filtered[filtered["patient_waittime"] >= min_wait]
    if triage is not None:
        filtered = filtered[filtered["Triage_Level"] == triage]
    if admitted:
        filtered = filtered[
            filtered["Admitted"].str.lower() == admitted.lower()
        ]

    total = len(filtered)
    start = (page - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]

    cols = [
        "patient_id",
        "patient_first_inital",
        "patient_last_name",
        "date",
        "patient_gender",
        "patient_age",
        "patient_waittime",
        "department_referral",
        "Triage_Level",
        "Admitted",
        "patient_sat_score",
        "Insurance_Type",
        "Clinical_Risk_Score",
    ]
    cols = [c for c in cols if c in page_df.columns]
    out = page_df[cols].copy()
    out["date"] = out["date"].astype(str)

    # Pandas 3.x-safe NaN → None replacement
    records = out.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (v != v):  # NaN check
                rec[k] = None
            elif hasattr(v, 'item'):  # numpy scalar
                rec[k] = v.item()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": records,
    }


# ── /kpis endpoint ───────────────────────────────────────────────────────────
@app.get("/kpis")
def get_kpis():
    wt = df_raw["patient_waittime"].dropna()
    total = len(df_raw)
    avg_wait = round(float(wt.mean()), 1) if len(wt) else 0
    longest_wait = round(float(wt.max()), 1) if len(wt) else 0
    over_4h = int((wt > 120).sum())
    pct_within_4h = round(float((wt <= 120).sum() / len(wt) * 100), 1) if len(wt) else 0

    # Department breakdown
    dept_avg = (
        df_raw.groupby("department_referral")["patient_waittime"]
        .mean()
        .dropna()
        .round(1)
        .to_dict()
    )

    return {
        "total_patients": total,
        "avg_wait_minutes": avg_wait,
        "longest_wait_minutes": longest_wait,
        "patients_over_4h": over_4h,
        "pct_within_4h_target": pct_within_4h,
        "dept_avg_wait": dept_avg,
    }


# ── /charts endpoint ─────────────────────────────────────────────────────────
@app.get("/charts")
def get_charts():
    # By hour of day
    by_hour: dict = {}
    if "Arrival_Hour" in df_raw.columns:
        counts = df_raw["Arrival_Hour"].dropna().astype(int).value_counts()
        by_hour = {str(h): int(counts.get(h, 0)) for h in range(24)}

    # By day of week
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow: dict = {}
    if "date" in df_raw.columns:
        dow_counts = df_raw["date"].dt.dayofweek.dropna().astype(int).value_counts()
        by_dow = {dow_names[i]: int(dow_counts.get(i, 0)) for i in range(7)}

    # By triage level
    triage_labels = {1: "1 – Critical", 2: "2 – Emergency", 3: "3 – Urgent",
                     4: "4 – Semi-urgent", 5: "5 – Non-urgent"}
    by_triage: dict = {}
    if "Triage_Level" in df_raw.columns:
        tc = df_raw["Triage_Level"].dropna().astype(int).value_counts().sort_index()
        by_triage = {triage_labels.get(k, str(k)): int(v) for k, v in tc.items()}

    # By date (for calendar heatmap)
    by_date: dict = {}
    if "date" in df_raw.columns:
        dc = df_raw["date"].dt.date.value_counts()
        by_date = {str(k): int(v) for k, v in dc.items()}

    return {
        "by_hour": by_hour,
        "by_dow": by_dow,
        "by_triage": by_triage,
        "by_date": by_date,
    }


# ── /alerts endpoint ─────────────────────────────────────────────────────────
@app.get("/alerts")
def get_alerts():
    dept_avg = (
        df_raw.groupby("department_referral")["patient_waittime"]
        .mean()
        .dropna()
        .round(1)
    )
    alerts = dept_avg[dept_avg > WAIT_ALERT_THRESHOLD_MINUTES]
    return {
        "threshold_minutes": WAIT_ALERT_THRESHOLD_MINUTES,
        "alerts": [
            {"department": dept, "avg_wait_minutes": float(avg)}
            for dept, avg in alerts.items()
        ],
    }


# ── /departments endpoint ────────────────────────────────────────────────────
@app.get("/departments")
def get_departments():
    depts = sorted(df_raw["department_referral"].dropna().unique().tolist())
    return {"departments": depts}


# ── /schema endpoint (debug) ─────────────────────────────────────────────────
@app.get("/schema")
def get_schema():
    return {
        "columns": list(df_raw.columns),
        "dtypes": {col: str(dtype) for col, dtype in df_raw.dtypes.items()},
        "shape": list(df_raw.shape),
        "date_range": {
            "min": str(df_raw["date"].min()),
            "max": str(df_raw["date"].max()),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
