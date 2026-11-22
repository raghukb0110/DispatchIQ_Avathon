"""Model wrapper with a uniform score / contribution API across model types."""
import numpy as np


class Scorer:
    def __init__(self, kind: str, model, feature_names: list[str], meta: dict | None = None):
        self.kind = kind  # logreg | lgbm_classifier | lgbm_ranker
        self.model = model
        self.feature_names = feature_names
        self.meta = meta or {}

    def score(self, X: np.ndarray) -> np.ndarray:
        # Single-threaded: candidate sets are tiny, so thread start-up dominates otherwise.
        if self.kind == "lgbm_ranker":
            return self.model.predict(X, num_threads=1)
        if self.kind == "lgbm_classifier":
            return self.model.predict_proba(X, num_threads=1)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def confidence(self, scores: np.ndarray) -> np.ndarray:
        """Share of belief per candidate within one decision."""
        if self.kind == "lgbm_ranker":
            e = np.exp(scores - scores.max())
            return e / e.sum()
        total = scores.sum()
        return scores / total if total > 0 else np.full_like(scores, 1 / len(scores))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the score (TreeSHAP for LightGBM, coef * z-score for logreg)."""
        if self.kind == "logreg":
            scaler, clf = self.model.named_steps["scaler"], self.model.named_steps["clf"]
            return scaler.transform(X) * clf.coef_[0]
        return self.model.predict(X, pred_contrib=True, num_threads=1)[:, :-1]

    def feature_importance(self) -> dict[str, float]:
        if self.kind == "logreg":
            imp = np.abs(self.model.named_steps["clf"].coef_[0])
        else:
            imp = self.model.booster_.feature_importance(importance_type="gain")
        imp = imp / imp.sum() if imp.sum() > 0 else imp
        return {n: float(v) for n, v in sorted(zip(self.feature_names, imp), key=lambda x: -x[1])}
