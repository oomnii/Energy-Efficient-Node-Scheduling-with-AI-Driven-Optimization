import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from src.ml.storage import load_status, save_status

CLASSIFIER_MODEL_PATH = Path(__file__).with_name("rf_model.pkl")
REGRESSOR_MODEL_PATH = Path(__file__).with_name("rf_metrics_model.pkl")


def _train_classifier(X, y):
    stratify = y if len(set(y.tolist())) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    score = float(accuracy_score(y_test, clf.predict(X_test))) if len(X_test) else 1.0
    return clf, score


def _train_regressor(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=160, max_depth=10, random_state=42)
    reg.fit(X_train, y_train)
    score = float(r2_score(y_test, reg.predict(X_test))) if len(X_test) else 1.0
    return reg, score


def save_classifier(model):
    CLASSIFIER_MODEL_PATH.write_bytes(pickle.dumps(model))


def save_regressor(model):
    REGRESSOR_MODEL_PATH.write_bytes(pickle.dumps(model))


def train_model(X, y):
    clf, _score = _train_classifier(X, y)
    save_classifier(clf)
    return clf


def train_metric_model(X, y):
    reg, _score = _train_regressor(X, y)
    save_regressor(reg)
    return reg


def train_model_with_score(X, y, persist: bool = True):
    clf, score = _train_classifier(X, y)
    if persist:
        save_classifier(clf)
    return clf, score


def train_metric_model_with_score(X, y, persist: bool = True):
    reg, score = _train_regressor(X, y)
    if persist:
        save_regressor(reg)
    return reg, score


def load_model():
    if CLASSIFIER_MODEL_PATH.exists():
        return pickle.loads(CLASSIFIER_MODEL_PATH.read_bytes())
    return None


def load_metric_model():
    if REGRESSOR_MODEL_PATH.exists():
        return pickle.loads(REGRESSOR_MODEL_PATH.read_bytes())
    return None


def predict_nodes(model, features):
    return model.predict(features).astype(bool)


def predict_run_metrics(model, features):
    return model.predict(features)


def current_ml_status() -> dict:
    return load_status()


def update_ml_status(**kwargs) -> dict:
    return save_status(kwargs)
