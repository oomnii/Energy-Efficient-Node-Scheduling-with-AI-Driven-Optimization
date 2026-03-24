Energy-Efficient-Node-Scheduling-with-AI-Driven-Optimization

A FastAPI + HTML/CSS/JS simulator for AI-Driven node scheduling in wireless sensor networks.

<a href="https://voronoi-wsn-scheduler.onrender.com/" target="_blank">
  <button>🚀 Live Demo</button>
</a>

## Core features

- Voronoi-threshold scheduling with backup nodes
- Fault tolerance and recovery simulation
- Initial and final algorithm comparison on the same generated field
- AI-assisted scheduling using a local RandomForest classifier
- ML-based run-level metric prediction (coverage, energy saving, lifetime)
- Multi-sensor deployment modes: homogeneous, heterogeneous, temperature, humidity, motion, mixed
- Dedicated ML Lab page for model status, history, and dataset previews
- Density experiments and CSV export

## Review

This package keeps the project focused on the core WSN scheduling story:

- energy-aware scheduling
- backup nodes
- failure and recovery simulation
- algorithm comparison
- ML / AI assistance

## Future Scope

- Map mode for visible frontend workflow in this build.
- Enhance AI/ML model for better performance.
- Add more features to the project.

## Run locally

```bash
pip install -r web/backend/requirements_web.txt
python -m uvicorn web.backend.main:app --reload --port 8000
```

Open `http://localhost:8000`.
