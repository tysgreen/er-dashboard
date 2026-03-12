# Hospital ER Intelligence Dashboard

A natural-language query interface for Emergency Room analytics, powered by the Anthropic Claude API. Ask plain-English questions about ER data and get instant answers, auto-generated charts, and operational summaries.

▶ **[Click to watch demo video](https://github.com/tysgreen/er-dashboard/blob/main/demo%20video.mp4)**

---

## Features

- **Natural Language Queries** — ask anything about the data and get instant answers + charts
- **Live KPI Cards** — total patients, average wait time, longest wait, % seen within 2-hour target
- **Interactive Chart Panel** — 5 tabs: dept wait times, arrivals by hour, triage breakdown, day-of-week volume, and a GitHub-style calendar heatmap (hover for daily counts)
- **Chat-Driven Charts** — query results automatically populate a "Last Query" tab in the chart panel
- **Department Alerts** — red highlights for any department averaging > 2.5 hours (150 min)
- **Filterable Patient Table** — filter by department, wait time range, triage level, and admission status
- **Auto Chart Selection** — bar, line, doughnut, or table depending on the result type

---

## Setup

### 1. Prerequisites

- Python 3.10+ (tested on 3.14)
- An [Anthropic API key](https://console.anthropic.com/)
- The ER dataset CSV file (not included — see Dataset section below)

### 2. Clone the repo

```bash
git clone https://github.com/tysgreen/er-dashboard.git
cd er-dashboard
```

### 3. Install dependencies

```bash
pip install --only-binary :all: -r requirements.txt
```

> **Note:** If you're on Python 3.13+ use `--only-binary :all:` to avoid compilation errors on packages that don't yet have source builds for newer Python versions.

### 4. Set environment variables

You must set two environment variables — your Anthropic API key, and the path to your CSV dataset.

**Windows (PowerShell — session only)**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:ER_DATA_PATH = "C:\path\to\CLAUDE Dataset with ML.csv"
```

**Windows (permanent via System Settings)**
1. Search "Environment Variables" in the Start menu
2. Click "Edit the system environment variables" → "Environment Variables"
3. Under User variables, add:
   - `ANTHROPIC_API_KEY` = your key
   - `ER_DATA_PATH` = full path to your CSV

**macOS / Linux**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export ER_DATA_PATH="/path/to/CLAUDE Dataset with ML.csv"
```

### 5. Start the backend

```bash
cd backend
python main.py
```

You should see:
```
[startup] Loaded 9,216 rows × 28 cols
[startup] Date range: 2019-04-01 → 2020-10-30
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Leave this terminal open — closing it stops the server.

### 6. Open the frontend

Open a second terminal and run:

```bash
cd frontend
python -m http.server 3000
```

Then open **http://localhost:3000** in your browser.

> Opening `index.html` directly as a file (`file://`) may cause CORS errors — always serve it via `http.server`.

---

## Dataset

The dataset (`CLAUDE Dataset with ML.csv`) is included in the repo inside the `data/` folder. The backend finds it automatically — no configuration needed. If you want to use a different file, set the `ER_DATA_PATH` environment variable to its full path.

### Columns

| Column | Description |
|---|---|
| `patient_id` | Unique patient identifier |
| `date` | Arrival datetime |
| `Arrival_DayOfWeek` | Day name (Monday…) |
| `Is_Weekend` | 1 = weekend, 0 = weekday |
| `Arrival_Hour` | Hour of arrival (0–23) |
| `patient_gender` | M / F |
| `patient_age` | Age in years |
| `patient_sat_score` | Satisfaction score 1–10 |
| `patient_race` | Race / ethnicity |
| `patient_waittime` | **Primary wait time in minutes** |
| `department_referral` | Referring department |
| `Insurance_Type` | Public / Private / None |
| `Estimated_Cost_USD` | Estimated treatment cost |
| `Triage_Level` | 1 (critical) – 5 (non-urgent) |
| `AI_Waiting_Time_Minutes` | AI-predicted wait time |
| `Treatment_Duration_Minutes` | Actual treatment duration |
| `Clinical_Risk_Score` | Clinical risk score (float) |
| `Resource_Intensity_Score` | Resource usage score |
| `Admitted` | Yes / No |
| `Admission_Probability` | Model probability 0–1 |
| `Predicted_Admission` | 0 / 1 binary |

---

## Example Questions

```
What's the average wait time by department?
Which days of the week are busiest?
How many patients waited over 2 hours?
Show me the distribution of triage levels
What percentage of patients were seen within the 2-hour target?
Average wait time by hour of day?
Which department has the highest clinical risk score on average?
How does wait time vary by insurance type?
What's the admission rate by triage level?
Average satisfaction score by department?
How many patients arrived each month?
What's the average cost by insurance type?
Compare wait times on weekdays vs weekends
Which triage level has the most patients?
Show me patients with a severity score above 3
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /kpis` | GET | KPI summary + per-department averages |
| `GET /charts` | GET | Hour, day-of-week, triage, and calendar heatmap data |
| `GET /alerts` | GET | Departments over 2.5-hour wait threshold |
| `GET /patients` | GET | Paginated + filtered patient records |
| `GET /departments` | GET | List of all departments |
| `POST /query` | POST | Natural language query → answer + chart data |
| `GET /schema` | GET | Column names and data types |

### POST /query

**Request:**
```json
{ "question": "What is the average wait time by department?" }
```

**Response:**
```json
{
  "question": "What is the average wait time by department?",
  "answer": "The average wait time varies significantly by department...",
  "chart_type": "bar",
  "chart_title": "Average Wait Time by Department",
  "data": {
    "type": "series",
    "labels": ["Cardiology", "Neurology", "..."],
    "values": [142.3, 138.7, "..."]
  },
  "code": "result = df.groupby('department_referral')['patient_waittime'].mean()..."
}
```

---

## Architecture

```
er-dashboard/
├── backend/
│   └── main.py          # FastAPI app + Claude API integration
├── data/
│   └── CLAUDE Dataset with ML.csv
├── frontend/
│   └── index.html       # Single-file dashboard (no build step)
├── requirements.txt
├── .gitignore
└── README.md
```

**How a natural language query works:**
1. User types a question → `POST /query`
2. Backend sends the question + ER domain system prompt to Claude (`claude-sonnet-4-20250514`)
3. Claude returns pandas code to answer the question
4. Code is safety-checked and executed against the DataFrame
5. Result is sent back to Claude for a plain-English summary
6. Frontend renders the answer as text + a Chart.js visualisation
7. Chart also populates the "Last Query ✦" tab in the right panel

---

## Tech Stack

- **Backend:** Python, FastAPI, pandas, Anthropic Python SDK
- **Frontend:** Vanilla HTML/CSS/JS, Chart.js
- **AI:** Claude (`claude-sonnet-4-20250514`) via Anthropic API
