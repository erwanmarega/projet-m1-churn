import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    return {
        "model": name,
        "accuracy": report["accuracy"],
        "precision_churn": report["1"]["precision"],
        "recall_churn": report["1"]["recall"],
        "f1_churn": report["1"]["f1-score"],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def plot_confusion_matrix(name: str, model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matrice de confusion — {name}")
    plt.tight_layout()
    path = f"{RESULTS_DIR}/confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curves(models_dict: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models_dict.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("Courbes ROC — Comparaison des modèles")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png", dpi=150)
    plt.close()


def plot_pr_curves(models_dict: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models_dict.items():
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title("Courbes Precision-Recall — Comparaison des modèles")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pr_curves.png", dpi=150)
    plt.close()


def save_comparison_table(all_metrics: list):
    df = pd.DataFrame(all_metrics).set_index("model").round(4)
    print("\n" + "="*60)
    print("TABLEAU COMPARATIF DES MODÈLES")
    print("="*60)
    print(df.to_string())
    df.to_csv(f"{RESULTS_DIR}/model_comparison.csv")
    return df
