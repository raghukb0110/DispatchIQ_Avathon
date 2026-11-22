import json
import logging
from pathlib import Path

import joblib

from app.ml.scorer import Scorer

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "scorer.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"

_scorer: Scorer | None = None


def _quick_train(reason: str) -> None:
    from app.ml.train import train

    logging.getLogger(__name__).warning("%s - training a quick model (60 scenarios)", reason)
    train(n_scenarios=60, verbose=False)


def get_scorer(auto_train: bool = True) -> Scorer:
    global _scorer
    if _scorer is None:
        if not MODEL_PATH.exists():
            if not auto_train:
                raise FileNotFoundError("No trained model. Run: python -m app.ml.train")
            _quick_train("No model found")
        try:
            _scorer = joblib.load(MODEL_PATH)
        except Exception:
            # A pickle from different library versions can fail to load; rebuild locally instead.
            if not auto_train:
                raise
            _quick_train("Saved model is incompatible with installed libraries")
            _scorer = joblib.load(MODEL_PATH)
    return _scorer


def model_card() -> dict:
    get_scorer()
    return json.loads(CARD_PATH.read_text())
