import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RANDOM_STATE = 42


def _feature_names_from_preprocessor(preprocessor) -> list:
    """Reconstruit la liste des noms de features post-ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform(
            preprocessor._df_columns).shape[1])]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Chargement du modèle Gradient Boosting...")
    with open(os.path.join(MODELS_DIR, "gradient_boosting.pkl"), "rb") as f:
        wrapped = pickle.load(f)

    pipeline = wrapped.pipeline
    preprocessor = pipeline.named_steps["prep"]
    gb_model = pipeline.named_steps["model"]

    print("Préparation des données...")
    X, y = load_data(DATA_PATH)
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = _feature_names_from_preprocessor(preprocessor)

    print("Calcul des valeurs SHAP (Permutation / Interventional)...")
    background = shap.utils.sample(X_train_t, 100, random_state=RANDOM_STATE)
    explainer = shap.Explainer(gb_model.predict_proba, background,
                                feature_names=feature_names)
    sample = shap.utils.sample(X_test_t, 200, random_state=RANDOM_STATE)
    shap_values = explainer(sample)
    # shap_values shape : (n, n_features, 2) → on garde la classe 1
    shap_values = shap_values[..., 1]

    print("Génération des figures...")
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    plt.figure(figsize=(9, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "shap_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out}")

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    print("\nTop 10 features (|SHAP| moyen) :")
    for i in order:
        print(f"  {feature_names[i]:<40s} {mean_abs[i]:.4f}")


if __name__ == "__main__":
    main()
