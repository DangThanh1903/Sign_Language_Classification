import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(data_dir: Path):
    X, y, mapping = [], [], {}
    class_idx = 0

    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_file() or not entry.name.lower().endswith(".npy"):
            continue
        cls_name = Path(entry.name).stem
        arr = np.load(data_dir / entry.name, allow_pickle=False)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        if arr.shape[0] == 0:
            print(f"⚠️  Bỏ qua {entry.name}: không còn mẫu hợp lệ.")
            continue
        X.append(arr)
        y += [class_idx] * arr.shape[0]
        mapping[class_idx] = cls_name
        class_idx += 1

    if len(X) == 0:
        raise RuntimeError("Không có dữ liệu hợp lệ trong thư mục data/. Hãy thu thêm .npy!")

    X = np.vstack(X)
    y = np.array(y, dtype=int)
    classes = [mapping[i] for i in range(len(mapping))]
    return X, y, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training model (Pipeline: Scaler -> PCA -> SVC)")
    parser.add_argument("--model_name", type=str, default="model_pca_svc", help="Tên file model (không kèm đuôi)")
    parser.add_argument("--dir", type=str, default="models", help="Thư mục lưu model")
    parser.add_argument("--data_dir", type=str, default="data", help="Thư mục dữ liệu .npy")
    parser.add_argument("--test_size", type=float, default=0.1, help="Tỷ lệ test split")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔹 Đọc dữ liệu ...")
    X, y, classes = load_data(data_dir)
    print(f"➡️  Tổng mẫu: {X.shape[0]} | Số chiều: {X.shape[1]} | Số lớp: {len(classes)}")
    print("Các lớp:", classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(svd_solver="full")),
        ("svc", SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

    param_grid = {
        "pca__n_components": [0.90, 0.95, 0.99],
        "svc__C": [0.5, 1, 2, 5, 10],
        "svc__gamma": ["scale", 0.1, 0.01, 0.001],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=1)
    print("🔹 Bắt đầu GridSearchCV ...")
    search.fit(X_train, y_train)

    best_est = search.best_estimator_
    print("✅ Best params:", search.best_params_)
    print("✅ CV best score:", search.best_score_)

    y_pred = best_est.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Test accuracy: {test_acc*100:.2f}%")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    artifact = {
        "estimator": best_est,
        "classes": classes
    }
    model_path = out_dir / f"{args.model_name}.pkl"
    joblib.dump(artifact, model_path)
    print(f"💾 Đã lưu model vào: {model_path.resolve()}")
