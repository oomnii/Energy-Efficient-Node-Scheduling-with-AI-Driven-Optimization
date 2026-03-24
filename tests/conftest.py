from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolate_ml_artifacts(tmp_path, monkeypatch):
    import src.ml.model as model_mod
    import src.ml.storage as storage_mod
    import web.backend.main as main_mod

    node_path = tmp_path / "node_dataset.npz"
    metric_path = tmp_path / "metric_dataset.npz"
    status_path = tmp_path / "training_status.json"
    clf_path = tmp_path / "rf_model.pkl"
    reg_path = tmp_path / "rf_metrics_model.pkl"

    monkeypatch.setattr(storage_mod, "NODE_DATASET_PATH", node_path)
    monkeypatch.setattr(storage_mod, "METRIC_DATASET_PATH", metric_path)
    monkeypatch.setattr(storage_mod, "STATUS_PATH", status_path)
    monkeypatch.setattr(model_mod, "CLASSIFIER_MODEL_PATH", clf_path)
    monkeypatch.setattr(model_mod, "REGRESSOR_MODEL_PATH", reg_path)
    monkeypatch.setattr(main_mod, "NODE_DATASET_PATH", node_path)
    monkeypatch.setattr(main_mod, "METRIC_DATASET_PATH", metric_path)
    yield
