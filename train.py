import os
import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    save_comparison_table,
)
from src.preprocessing import build_preprocessor, load_data
from src.threshold import ThresholdClassifier, find_best_threshold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RANDOM_STATE = 42

def build_pipelines():
    """Retourne les 4 pipelines (preprocessing + SMOTE + modèle)."""
    lr = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])

    rf = ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    # Gradient Boosting (HistGradientBoosting) : équivalent LightGBM,
    # gère class_weight nativement, pas de dépendance système (libomp)
    gb = Pipeline([
        ("prep", build_preprocessor()),
        ("model", HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            l2_regularization=1.0, class_weight="balanced",
            early_stopping=True, validation_fraction=0.1,
            random_state=RANDOM_STATE)),
    ])

    # MLP : 2 couches cachées, early stopping pour limiter l'overfit
    mlp = ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=1e-3, batch_size=128, learning_rate_init=1e-3,
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=RANDOM_STATE)),
    ])

    return {"Logistic Regression": lr, "Random Forest": rf,
            "Gradient Boosting": gb, "MLP": mlp}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement des données...")
    X, y = load_data(DATA_PATH)
    print(f"Dataset : {X.shape[0]} clients, {X.shape[1]} features")
    print(f"Churn   : {y.mean():.1%} positifs (déséquilibre ~1:9)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train,
        random_state=RANDOM_STATE)
    print(f"Train : {len(X_tr)} | Val : {len(X_val)} | Test : {len(X_test)}")

    pipelines = build_pipelines()
    fitted_raw: dict = {}
    fitted_thresh: dict = {}
    all_metrics = []

    for name, pipeline in pipelines.items():
        print(f"\n[{name.upper()}]")
        pipeline.fit(X_tr, y_tr)
        thresh = find_best_threshold(pipeline, X_val, y_val)
        fitted_raw[name] = pipeline
        fitted_thresh[name] = ThresholdClassifier(pipeline, threshold=thresh)

    for name, model in fitted_thresh.items():
        all_metrics.append(evaluate_model(name, model, X_test, y_test))
        plot_confusion_matrix(name, model, X_test, y_test)

    plot_roc_curves(fitted_raw, X_test, y_test)
    plot_pr_curves(fitted_raw, X_test, y_test)
    save_comparison_table(all_metrics)

    # CV stratifié sur le modèle candidat final (Gradient Boosting)
    print("\n[CROSS-VALIDATION] Stratified K-Fold (k=5) — Gradient Boosting")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gb_cv = Pipeline([
        ("prep", build_preprocessor()),
        ("model", HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    cv_f1 = cross_val_score(gb_cv, X_train, y_train, cv=cv,
                            scoring="f1", n_jobs=-1)
    cv_roc = cross_val_score(gb_cv, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    print(f"  F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

    file_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "MLP": "mlp",
    }
    for name, slug in file_map.items():
        path = os.path.join(MODELS_DIR, f"{slug}.pkl")
        with open(path, "wb") as f:
            pickle.dump(fitted_thresh[name], f)
        print(f"Modèle sauvegardé : {path}")

    # Modèle candidat final : Gradient Boosting
    final_path = os.path.join(MODELS_DIR, "final_model.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(fitted_thresh["Gradient Boosting"], f)
    print(f"Modèle candidat final : {final_path}")

    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
