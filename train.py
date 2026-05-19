import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocessing import load_data, build_preprocessor
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    save_comparison_table,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RANDOM_STATE = 42


def find_best_threshold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.6, 0.01):
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_val, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, t
    print(f"  Seuil optimal : {best_thresh:.2f}  (F1={best_f1:.4f})")
    return best_thresh


class ThresholdClassifier:
    def __init__(self, pipeline, threshold=0.5):
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


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement des données...")
    X, y = load_data(DATA_PATH)
    print(f"Dataset : {X.shape[0]} clients, {X.shape[1]} features")
    print(f"Churn : {y.mean():.1%} positifs (déséquilibre ~1:9)\n")

    # stratify=y pour préserver les proportions de classes dans chaque split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # val séparé du test pour calibrer le seuil sans biaiser l'évaluation finale
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"Train : {len(X_tr)} | Val : {len(X_val)} | Test : {len(X_test)}")

    print("\n[BASELINE] Régression Logistique avec class_weight='balanced'")
    lr_base = Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    lr_base.fit(X_tr, y_tr)
    print("  Recherche du seuil optimal (val set)...")
    lr_pipeline = ThresholdClassifier(lr_base, threshold=find_best_threshold(lr_base, X_val, y_val))

    # SMOTE sans class_weight pour éviter la sur-correction du déséquilibre
    print("\n[RANDOM FOREST] avec SMOTE + ajustement de seuil")
    rf_base = ImbPipeline([
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_base.fit(X_tr, y_tr)
    print("  Recherche du seuil optimal (val set)...")
    rf_pipeline = ThresholdClassifier(rf_base, threshold=find_best_threshold(rf_base, X_val, y_val))

    all_metrics = []
    models = {"Logistic Regression": lr_pipeline, "Random Forest": rf_pipeline}

    for name, model in models.items():
        all_metrics.append(evaluate_model(name, model, X_test, y_test))
        plot_confusion_matrix(name, model, X_test, y_test)

    # pipelines bruts pour predict_proba sur les courbes ROC/PR
    plot_roc_curves({"Logistic Regression": lr_base, "Random Forest": rf_base}, X_test, y_test)
    plot_pr_curves({"Logistic Regression": lr_base, "Random Forest": rf_base}, X_test, y_test)
    save_comparison_table(all_metrics)

    print("\n[CROSS-VALIDATION] Stratified K-Fold (k=5) — Random Forest")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_cv = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    cv_f1 = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    cv_roc = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  ROC-AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

    print("\n[COMPARAISON RÉÉQUILIBRAGE]")
    rf_no_balance = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_no_balance.fit(X_tr, y_tr)

    rf_class_weight = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_class_weight.fit(X_tr, y_tr)

    import pandas as pd
    from sklearn.metrics import recall_score
    imbalance_results = []
    for label, m in [("Sans rééquilibrage", rf_no_balance), ("class_weight", rf_class_weight), ("SMOTE", rf_base)]:
        thresh = find_best_threshold(m, X_val, y_val)
        y_proba = m.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= thresh).astype(int)
        imbalance_results.append({
            "technique": label,
            "seuil": round(thresh, 2),
            "recall_churn": round(recall_score(y_test, y_pred), 4),
            "f1_churn": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
    df_imbalance = pd.DataFrame(imbalance_results).set_index("technique")
    print(df_imbalance.to_string())
    df_imbalance.to_csv(os.path.join(RESULTS_DIR, "imbalance_comparison.csv"))

    for name, model in {"logistic_regression": lr_pipeline, "random_forest": rf_pipeline}.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Modèle sauvegardé : {path}")

    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()
