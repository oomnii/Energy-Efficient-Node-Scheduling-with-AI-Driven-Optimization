# WSN-01 — Voronoi WSN Scheduler

A FastAPI + HTML/CSS/JS simulator for Voronoi-based node scheduling in wireless sensor networks.

🚀 **Live Demo:** [https://voronoi-wsn-scheduler.onrender.com/](https://voronoi-wsn-scheduler.onrender.com/)
*(Note: If you ended up creating a new Render service with a new name, remember to update this link!)*

## Core features
- Voronoi-threshold scheduling with backup nodes
- Fault tolerance and recovery simulation
- Initial and final algorithm comparison on the same generated field
- AI-assisted scheduling using a local RandomForest classifier
- ML-based run-level metric prediction (coverage, energy saving, lifetime)
- Multi-sensor deployment modes: homogeneous, heterogeneous, temperature, humidity, motion, mixed
- Dedicated ML Lab page for model status, history, and dataset previews
- Density experiments and CSV export

## Mid-review scope
This mid-review package keeps the project focused on the core WSN scheduling story:
- energy-aware scheduling
- backup nodes
- failure and recovery simulation
- algorithm comparison
- ML / AI assistance

Map mode is intentionally left out of the visible frontend workflow in this build.

## Run locally
```bash
pip install -r web/backend/requirements_web.txt
python -m uvicorn web.backend.main:app --reload --port 8000
```

Open `http://localhost:8000`.
