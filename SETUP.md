# WSN-01 Local Setup Guide

## Overview

this is a full-stack AI-Driven node scheduling in wireless sensor networks built with a FastAPI backend and a static HTML/CSS/JavaScript frontend. The current build focuses on:

- true-proportion field rendering for any width × height
- Voronoi-based redundant-node shutdown
- backup/off nodes and round-wise fault tracking
- sensor-aware energy and coverage behaviour
- dedicated energy-saving dashboard
- initial and final algorithm comparison
- separate ML Lab page for model memory, history, and previews
- local ML training for AI-assisted scheduling
- CSV result export

## Recommended environment

- Python 3.10 to 3.13
- Windows, macOS, or Linux
- Modern browser (Chrome / Edge / Firefox)

## Project structure

```text
src/
  algorithms/
  config.py
  experiments/
  ml/
  simulation/
web/
  backend/
  frontend/
tests/
SETUP.md
```

## 1) Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2) Install dependencies

```bash
pip install -r web/backend/requirements_web.txt
```

## 3) Run the app locally

From the project root:

```bash
python -m uvicorn web.backend.main:app --reload --port 8000
```

Open in your browser:

```text
http://127.0.0.1:8000
```

## 4) Run tests

```bash
python -m pytest tests/ -q
```

## 5) Recommended localhost workflow

1. Start the server.
2. Open the app in the browser.
3. Run a standard Voronoi simulation.
4. Review the **Run** tab for initial scheduling and final post-failure state.
5. Review the **Energy** tab for savings and lifetime impact.
6. Run **Compare** to inspect both initial and final algorithm comparison.
7. Train the local model using **Train ML Model** in the sidebar.
8. Open **ML Lab** to inspect model status, history, and stored datasets.
9. Enable **AI Scheduling** and rerun.
10. Run the density experiment and export data if needed.
11. Export the run CSV when you need the current run data.

## Important notes about ML / AI scheduling

- The **Train ML Model** button builds a **local RandomForest model**.
- Training data is **synthetic WSN simulation data**, not real sensor history.
- Retraining appends new synthetic samples to the saved dataset, rebuilds a candidate model from the combined history, and keeps the strongest saved model seen so far.
- If no ML model exists yet, **AI Scheduling** automatically falls back to the Voronoi scheduler.

## Main API endpoints

- `GET /api/health`
- `GET /api/ml/status`
- `GET /api/ml/memory`
- `POST /api/run`
- `POST /api/compare`
- `POST /api/experiment/density`
- `POST /api/export/run.csv`
- `POST /api/ml/train`
- `POST /api/ml/predict`

## Troubleshooting

### `ModuleNotFoundError`

Run all commands from the project root folder.

### Port 8000 already in use

Run on another port:

```bash
python -m uvicorn web.backend.main:app --reload --port 8080
```

Then open `http://127.0.0.1:8080`.

### AI Scheduling does not look different

Train the ML model first. Also compare multiple sensor types and threshold coefficients so the energy and coverage trade-offs become visible.

### Failure probability feels too strong

This build is tuned for small values. Keep **Per-round failure probability** between **0.001 and 0.020** for review-friendly behaviour.

### Tests or UI feel stale after code edits

Stop the server, clear browser cache, restart Uvicorn, and rerun pytest.

## Verified baseline

This packaged project was validated with automated tests before packaging. Your final browser pass should focus on:

- true field aspect ratio
- backup nodes vs failed nodes
- initial vs final comparison clarity
- energy dashboard plots
- ML Lab page content
- AI scheduling run / compare behaviour
