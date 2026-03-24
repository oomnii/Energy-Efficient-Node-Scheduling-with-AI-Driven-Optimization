import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SimConfig
from src.ml.dataset import generate_metric_training_data, generate_training_data
from src.ml.model import (
    current_ml_status,
    load_metric_model,
    load_model,
    predict_run_metrics,
    save_classifier,
    save_regressor,
    train_metric_model_with_score,
    train_model_with_score,
    update_ml_status,
)
from src.ml.storage import (
    METRIC_DATASET_PATH,
    NODE_DATASET_PATH,
    RUN_FEATURE_NAMES,
    RUN_TARGET_NAMES,
    NODE_FEATURE_NAMES,
    NODE_TARGET_NAMES,
    append_dataset,
    dataset_preview,
)
from src.simulation.simulator import experiment_density, run_compare, run_voronoi, serialize_run_payload

FRONTEND_DIR = ROOT / "web" / "frontend"

app = FastAPI(title="Voronoi WSN Premium Dashboard", version="4.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


class SimRequest(BaseModel):
    n_nodes: int = Field(default=120, ge=1, le=5000)
    width: float = Field(default=100.0, gt=0)
    height: float = Field(default=100.0, gt=0)
    sensing_radius: float = Field(default=15.0, gt=0)
    threshold_coeff: float = Field(default=0.02, ge=0.0, le=0.02)
    seed: int = 7
    enable_fault_tolerance: bool = True
    failure_prob_per_round: float = Field(default=0.01, ge=0.0, le=1.0)
    failure_model: str = "random"
    recovery_model: str = "greedy_coverage"
    n_rounds: int = Field(default=50, ge=1, le=1000)
    sensor_type: str = "homogeneous"
    map_mode: bool = False
    enable_ai: bool = False
    map_bounds: list[list[float]] | None = None


class PredictRequest(BaseModel):
    features: list[float]


class MLPredictPayload(BaseModel):
    features: list[float] | None = None
    sim_request: SimRequest | None = None


VALID_SENSOR_TYPES = {"homogeneous", "heterogeneous", "temperature", "humidity", "motion", "mixed"}
VALID_FAILURE_MODELS = {"random", "clustered", "region", "periodic"}
VALID_RECOVERY_MODELS = {"greedy_coverage", "nearest_backup"}


def _cfg(req: SimRequest) -> SimConfig:
    if req.sensor_type not in VALID_SENSOR_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported sensor_type: {req.sensor_type}")
    if req.failure_model not in VALID_FAILURE_MODELS:
        raise HTTPException(status_code=422, detail=f"Unsupported failure_model: {req.failure_model}")
    if req.recovery_model not in VALID_RECOVERY_MODELS:
        raise HTTPException(status_code=422, detail=f"Unsupported recovery_model: {req.recovery_model}")

    return SimConfig(
        n_nodes=req.n_nodes,
        width=req.width,
        height=req.height,
        sensing_radius=req.sensing_radius,
        threshold_coeff=req.threshold_coeff,
        seed=req.seed,
        enable_fault_tolerance=req.enable_fault_tolerance,
        failure_prob_per_round=req.failure_prob_per_round,
        failure_model=req.failure_model,
        recovery_model=req.recovery_model,
        n_rounds=req.n_rounds,
        sensor_type=req.sensor_type,
        map_mode=req.map_mode,
        map_bounds=req.map_bounds,
        enable_ai=req.enable_ai,
    )


def _predict_features(features: list[float]):
    X = np.array(features, dtype=float).reshape(1, -1)
    if X.shape[1] == 6:
        model = load_model()
        if model is None:
            raise HTTPException(status_code=400, detail="Node-level classifier not trained")
        pred = model.predict(X)
        return {"prediction": int(pred[0]), "prediction_type": "node_state", "model_loaded": True}
    metric_model = load_metric_model()
    if metric_model is None:
        raise HTTPException(status_code=400, detail="Run-metric regressor not trained")
    pred = predict_run_metrics(metric_model, X)
    return {
        "prediction_type": "run_metrics",
        "prediction": {
            "coverage_scheduled": float(pred[0][0]),
            "energy_saved_pct": float(pred[0][1]),
            "estimated_network_lifetime_rounds": float(pred[0][2]),
        },
        "model_loaded": True,
    }


def _predict_simulation(req: SimRequest):
    cfg = _cfg(req)
    cfg.enable_ai = True
    model_loaded = load_model() is not None
    payload = serialize_run_payload(*run_voronoi(cfg))
    payload["ml_status"] = "model_loaded" if model_loaded else "fallback_voronoi"
    payload["model_loaded"] = model_loaded
    payload["ml_training"] = current_ml_status()
    return payload


def _nodes_csv_bytes(points, active_mask, unavailable_mask, metrics) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["key", "value"])
    for key, value in metrics.items():
        writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])
    writer.writerow([])
    writer.writerow(["node_index", "x", "y", "state", "sensor_label", "sensing_radius", "active_energy_cost"])
    sensor_labels = metrics.get("sensor_labels") or [metrics.get("sensor_type", "homogeneous")] * len(points)
    sensing_radii = metrics.get("sensing_radii") or [metrics.get("sensing_radius", 0.0)] * len(points)
    active_costs = metrics.get("active_energy_costs") or [metrics.get("energy_active_cost", 1.0)] * len(points)
    for i, (x, y) in enumerate(points.tolist()):
        state = "FAILED" if bool(unavailable_mask[i]) else ("ACTIVE" if bool(active_mask[i]) else "BACKUP_OFF")
        writer.writerow([i, x, y, state, sensor_labels[i], sensing_radii[i], active_costs[i]])
    return buf.getvalue().encode("utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/ml/status")
def api_ml_status():
    return current_ml_status()


@app.get("/api/ml/memory")
def api_ml_memory():
    return {
        "status": current_ml_status(),
        "node_dataset": dataset_preview(NODE_DATASET_PATH, NODE_FEATURE_NAMES, NODE_TARGET_NAMES),
        "run_dataset": dataset_preview(METRIC_DATASET_PATH, RUN_FEATURE_NAMES, RUN_TARGET_NAMES),
    }


@app.post("/api/run")
def api_run(req: SimRequest):
    return serialize_run_payload(*run_voronoi(_cfg(req)))


@app.post("/api/compare")
def api_compare(req: SimRequest):
    return run_compare(_cfg(req))


@app.post("/api/experiment/density")
def api_density(req: SimRequest):
    df = experiment_density(_cfg(req))
    return {"rows": df.to_dict(orient="records")}


@app.post("/api/export/run.csv")
def export_run_csv(req: SimRequest):
    points, _boundary, active_mask, _backups, unavailable_mask, metrics, _fault_logs = run_voronoi(_cfg(req))
    csv_bytes = _nodes_csv_bytes(points, active_mask, unavailable_mask, metrics)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wsn_run_export.csv"},
    )


@app.post("/api/ml/train")
def api_ml_train():
    try:
        previous = current_ml_status()

        new_X_nodes, new_y_nodes = generate_training_data(num_samples=20)
        all_X_nodes, all_y_nodes = append_dataset(NODE_DATASET_PATH, new_X_nodes, new_y_nodes)
        clf, classifier_accuracy = train_model_with_score(all_X_nodes, all_y_nodes, persist=False)

        new_X_metrics, new_y_metrics = generate_metric_training_data(num_samples=80)
        all_X_metrics, all_y_metrics = append_dataset(METRIC_DATASET_PATH, new_X_metrics, new_y_metrics)
        reg, regressor_r2 = train_metric_model_with_score(all_X_metrics, all_y_metrics, persist=False)

        best_classifier = previous.get("best_classifier_accuracy")
        best_regressor = previous.get("best_regressor_r2")
        keep_classifier = best_classifier is not None and classifier_accuracy + 1e-9 < float(best_classifier)
        keep_regressor = best_regressor is not None and regressor_r2 + 1e-9 < float(best_regressor)

        if not keep_classifier or load_model() is None:
            save_classifier(clf)
            best_classifier = classifier_accuracy if best_classifier is None else max(float(best_classifier), classifier_accuracy)

        if not keep_regressor or load_metric_model() is None:
            save_regressor(reg)
            best_regressor = regressor_r2 if best_regressor is None else max(float(best_regressor), regressor_r2)

        note_bits = [
            f"Stored history now has {len(all_X_nodes)} node samples and {len(all_X_metrics)} run samples.",
            "Each retrain appends new synthetic WSN data, rebuilds a candidate model, and keeps the strongest-scoring saved version so far.",
        ]
        if keep_classifier or keep_regressor:
            note_bits.append("A previous best model was kept where the new candidate did not improve the score.")
        else:
            note_bits.append("Both saved models were refreshed because the new candidate matched or improved the previous best score.")

        history = list(previous.get("history", []))
        next_version = int(previous.get("model_version", 0)) + 1
        history.append({
            "model_version": next_version,
            "trained_at": None,
            "classifier_accuracy": round(classifier_accuracy, 4),
            "regressor_r2": round(regressor_r2, 4),
            "new_node_samples": int(len(new_X_nodes)),
            "new_run_samples": int(len(new_X_metrics)),
        })
        history = history[-8:]

        status = update_ml_status(
            model_version=next_version,
            last_trained_at=None,
            classifier_accuracy=round(classifier_accuracy, 4),
            regressor_r2=round(regressor_r2, 4),
            best_classifier_accuracy=None if best_classifier is None else round(float(best_classifier), 4),
            best_regressor_r2=None if best_regressor is None else round(float(best_regressor), 4),
            total_node_samples=int(len(all_X_nodes)),
            total_run_samples=int(len(all_X_metrics)),
            node_feature_count=int(all_X_nodes.shape[1]),
            run_feature_count=int(all_X_metrics.shape[1]),
            classifier_model_kept=bool(keep_classifier),
            regressor_model_kept=bool(keep_regressor),
            training_note=" ".join(note_bits),
            history=history,
        )
        for item in status.get("history", []):
            if item.get("trained_at") is None:
                item["trained_at"] = status.get("last_trained_at")
        status = update_ml_status(history=status.get("history", []))
        return {"status": "success", "message": "ML models trained successfully.", "training": status}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"ML Training Error: {exc}") from exc


@app.post("/api/ml/predict")
def api_ml_predict(payload: MLPredictPayload | SimRequest | PredictRequest):
    if isinstance(payload, SimRequest):
        return _predict_simulation(payload)
    if isinstance(payload, PredictRequest):
        return _predict_features(payload.features)
    if payload.sim_request is not None:
        return _predict_simulation(payload.sim_request)
    if payload.features is not None:
        return _predict_features(payload.features)
    raise HTTPException(status_code=422, detail="Provide either features or a simulation request")
