from pathlib import Path

from fastapi.testclient import TestClient

from web.backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simulate_endpoint():
    request_data = {
        "n_nodes": 50,
        "width": 100.0,
        "height": 100.0,
        "sensing_radius": 15.0,
        "threshold_coeff": 0.02,
        "seed": 42,
    }
    response = client.post("/api/run", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert "active_mask" in data
    assert "metrics" in data
    assert len(data["points"]) == 50
    assert "initial_snapshot" in data["metrics"]
    assert "final_snapshot" in data["metrics"]


def test_compare_endpoint():
    request_data = {
        "n_nodes": 40,
        "width": 50.0,
        "height": 50.0,
        "sensing_radius": 15.0,
        "threshold_coeff": 0.02,
        "seed": 100,
    }
    response = client.post("/api/compare", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "scenario" in data
    assert "initial" in data
    assert "final" in data
    assert "voronoi" in data["initial"]
    assert "random_same_off" in data["initial"]
    assert "random_greedy_cov" in data["initial"]
    assert "greedy_cov" in data["initial"]
    assert "ai_based" in data["initial"]
    assert "voronoi" in data["final"]
    assert "final_snapshot" in data["final"]["voronoi"]["metrics"]


def test_experiment_endpoint():
    request_data = {
        "n_nodes": 20,
        "width": 50.0,
        "height": 50.0,
        "sensing_radius": 20.0,
    }
    response = client.post("/api/experiment/density", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "rows" in data
    assert len(data["rows"]) > 0


def test_export_endpoint():
    response = client.post("/api/export/run.csv", json={"n_nodes": 10, "seed": 11})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "node_index" in response.text


def test_ml_train_and_predict_endpoint():
    response = client.post("/api/ml/train")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    pred_response = client.post("/api/ml/predict", json={"features": [1.0, 2.0, 3.0, 1.0, 2.0, 15.0]})
    assert pred_response.status_code == 200
    assert pred_response.json()["prediction"] in {0, 1}

    model_path = Path("src/ml/rf_model.pkl")
    if model_path.exists():
        model_path.unlink()



def test_ml_predict_simulation_payload():
    train_response = client.post("/api/ml/train")
    assert train_response.status_code == 200

    response = client.post("/api/ml/predict", json={
        "n_nodes": 25,
        "width": 60.0,
        "height": 60.0,
        "sensing_radius": 12.0,
        "threshold_coeff": 0.02,
        "seed": 5,
        "enable_ai": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert "active_mask" in data
    assert "metrics" in data
    assert data["metrics"]["algo"] in {"AI-Scheduling", "Voronoi-Threshold"}
    assert "ml_status" in data

    model_path = Path("src/ml/rf_model.pkl")
    if model_path.exists():
        model_path.unlink()


def test_mixed_sensor_run():
    response = client.post("/api/run", json={"n_nodes": 18, "sensor_type": "mixed", "seed": 9})
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["sensor_type"] == "mixed"
    assert len(payload["sensor_labels"]) == 18
    assert set(payload["sensor_labels"]).issubset({"temperature", "humidity", "motion"})


def test_ml_status_and_run_metric_predict():
    status_response = client.get("/api/ml/status")
    assert status_response.status_code == 200
    assert "model_version" in status_response.json()

    client.post("/api/ml/train")
    trained_status = client.get("/api/ml/status").json()
    assert "best_classifier_accuracy" in trained_status
    assert "best_regressor_r2" in trained_status
    pred_response = client.post("/api/ml/predict", json={"features": [60.0, 120.0, 80.0, 15.0, 0.02, 0.00625, 3.0, 0.02, 1.5]})
    assert pred_response.status_code == 200
    payload = pred_response.json()
    assert payload["prediction_type"] == "run_metrics"
    assert set(payload["prediction"].keys()) == {"coverage_scheduled", "energy_saved_pct", "estimated_network_lifetime_rounds"}




def test_ml_memory_endpoint():
    client.post("/api/ml/train")
    response = client.get("/api/ml/memory")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "node_dataset" in payload
    assert "run_dataset" in payload
    assert payload["node_dataset"]["feature_names"]
    assert payload["run_dataset"]["target_names"]
    assert isinstance(payload["status"].get("history", []), list)

def test_map_bounds_and_round_logs_are_returned():
    response = client.post("/api/run", json={
        "n_nodes": 24,
        "map_mode": True,
        "map_bounds": [[21.1600, 72.8200], [21.1700, 72.8400]],
        "sensor_type": "motion",
        "seed": 4,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["map_bounds"] == [[21.16, 72.82], [21.17, 72.84]]
    assert "voronoi_cells" in payload
    assert payload["fault_logs"][0]["scheduled_energy"] >= 0
    assert payload["fault_logs"][0]["failed_count"] >= 0
