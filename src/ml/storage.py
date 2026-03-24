import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ML_DIR = Path(__file__).parent
NODE_DATASET_PATH = ML_DIR / "node_dataset.npz"
METRIC_DATASET_PATH = ML_DIR / "metric_dataset.npz"
STATUS_PATH = ML_DIR / "training_status.json"


DEFAULT_STATUS = {
    "model_version": 0,
    "last_trained_at": None,
    "classifier_accuracy": None,
    "regressor_r2": None,
    "best_classifier_accuracy": None,
    "best_regressor_r2": None,
    "total_node_samples": 0,
    "total_run_samples": 0,
    "node_feature_count": 0,
    "run_feature_count": 0,
    "training_note": "No model trained yet. Train the model to enable AI scheduling.",
    "model_type": "RandomForestClassifier + RandomForestRegressor",
    "training_basis": "Synthetic WSN simulation data stored locally and reused on every retrain.",
    "classifier_model_kept": False,
    "regressor_model_kept": False,
    "history": [],
}


def _load_npz(path: Path):
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["X"], data["y"]


def append_dataset(path: Path, X: np.ndarray, y: np.ndarray):
    old_X, old_y = _load_npz(path)
    if old_X is None:
        combined_X, combined_y = X, y
    else:
        combined_X = np.vstack([old_X, X])
        combined_y = np.vstack([old_y, y]) if old_y.ndim == 2 else np.concatenate([old_y, y])
    np.savez_compressed(path, X=combined_X, y=combined_y)
    return combined_X, combined_y


def load_status() -> dict:
    if STATUS_PATH.exists():
        return {**DEFAULT_STATUS, **json.loads(STATUS_PATH.read_text())}
    return dict(DEFAULT_STATUS)


def save_status(update: dict) -> dict:
    current = load_status()
    current.update(update)
    if update.get("last_trained_at") is None:
        from datetime import timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        current["last_trained_at"] = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    STATUS_PATH.write_text(json.dumps(current, indent=2))
    return current


NODE_FEATURE_NAMES = ["x_m", "y_m", "dist_to_center_m", "local_neighbors", "nearest_neighbor_m", "effective_radius_m"]
NODE_TARGET_NAMES = ["active_label"]
RUN_FEATURE_NAMES = ["n_nodes", "width_m", "height_m", "sensing_radius_m", "threshold_coeff", "density_nodes_per_m2", "sensor_code", "failure_prob", "aspect_ratio"]
RUN_TARGET_NAMES = ["coverage_scheduled", "energy_saved_pct", "estimated_network_lifetime_rounds"]


def dataset_preview(path: Path, feature_names: list[str], target_names: list[str], max_rows: int = 4) -> dict:
    X, y = _load_npz(path)
    if X is None:
        return {
            "count": 0,
            "feature_names": feature_names,
            "target_names": target_names,
            "preview_rows": [],
        }

    preview_rows = []
    limit = min(max_rows, len(X))
    for i in range(limit):
        row = {name: float(X[i][idx]) for idx, name in enumerate(feature_names)}
        if getattr(y, "ndim", 1) == 1:
            row[target_names[0]] = int(y[i])
        else:
            for idx, name in enumerate(target_names):
                row[name] = float(y[i][idx])
        preview_rows.append(row)

    return {
        "count": int(len(X)),
        "feature_names": feature_names,
        "target_names": target_names,
        "preview_rows": preview_rows,
    }
