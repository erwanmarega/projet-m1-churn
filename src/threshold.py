import numpy as np


class ThresholdClassifier:
    def __init__(self, pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X):
        proba = self.pipeline.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self


def find_best_threshold(model, X_val, y_val) -> float:
    from sklearn.metrics import f1_score

    y_proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.6, 0.01):
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_val, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    print(f"  Seuil optimal : {best_thresh:.2f}  (F1={best_f1:.4f})")
    return float(best_thresh)
