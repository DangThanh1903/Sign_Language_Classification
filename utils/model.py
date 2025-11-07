import joblib
import numpy as np
from pathlib import Path

class ASLClassificationModel:
    """
    API thống nhất:
      - load_model(path)
      - predict(feature_1d) -> str (label)
      - predict_proba([[feature]]) -> np.ndarray (n, n_classes)  (n=1 ở app)
      - classes_ (list[str])
    Kỳ vọng file .pkl: {"estimator": pipeline, "classes": list[str]}
    vẫn hỗ trợ fallback kiểu cũ (model, mapping).
    """

    def __init__(self, estimator, classes):
        self.estimator = estimator
        self.classes_ = list(classes)

    @classmethod
    def load_model(cls, path):
        path = Path(path)
        obj = joblib.load(path)
        if isinstance(obj, dict) and "estimator" in obj and "classes" in obj:
            return cls(obj["estimator"], obj["classes"])
        if isinstance(obj, tuple) and len(obj) == 2:
            model, mapping = obj
            classes = [mapping[i] for i in range(len(mapping))]
            return cls(model, classes)
        raise ValueError("Định dạng model không hợp lệ. Hãy lưu theo {'estimator','classes'} hoặc (model, mapping).")

    def predict(self, feature_1d):
        x = np.asarray(feature_1d, dtype=float).reshape(1, -1)
        pred_idx = int(self.estimator.predict(x)[0])
        if 0 <= pred_idx < len(self.classes_):
            return self.classes_[pred_idx]
        return str(pred_idx)

    def predict_proba(self, X):
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Estimator không hỗ trợ predict_proba. Hãy bật probability=True (SVC).")
        X = np.asarray(X, dtype=float)
        return self.estimator.predict_proba(X)
