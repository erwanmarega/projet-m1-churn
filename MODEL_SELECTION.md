# Choix du modèle final — Justification

## Tableau comparatif (test set, 2000 clients, churn = 10.2 %)

| Modèle              | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC | PR-AUC |
|---------------------|---------:|------------------:|---------------:|-----------:|--------:|-------:|
| Logistic Regression |   0.7485 |            0.2152 |         0.5539 |     0.3100 |  0.7439 | 0.2695 |
| Random Forest       |   0.7325 |            0.2450 |         0.7794 |     0.3728 |  0.7884 | 0.2757 |
| **Gradient Boosting** | **0.6925** |        **0.2307** |     **0.8627** | **0.3640** | **0.8027** | **0.2825** |
| MLP                 |   0.7845 |            0.1890 |         0.3382 |     0.2425 |  0.6463 | 0.1821 |

Cross-validation (Stratified 5-Fold) sur le candidat final :
- **F1**     : 0.3442 ± 0.0252
- **ROC-AUC** : 0.7686 ± 0.0225 → modèle stable, pas de dépendance à un split favorable

## Modèle retenu : Gradient Boosting (HistGradientBoosting)

### Pourquoi

1. **ROC-AUC le plus élevé** (0.8027) → meilleur séparateur global toutes décisions
   confondues. C'est la métrique la moins sensible au seuil et au déséquilibre.
2. **Recall churn le plus élevé** (0.86) → minimise les **faux négatifs**, qui sont
   le coût business dominant : un client à risque non détecté part sans action de
   rétention, on perd `total_revenue × proba_churn`. À l'inverse, un faux positif
   coûte au pire un email ou un appel.
3. **PR-AUC le plus élevé** (0.2825) → métrique recommandée pour les classes
   déséquilibrées (1:9 ici). Confirme que le gain est réel, pas une illusion
   d'arrondi du F1.
4. **Stable en CV** (σ ROC-AUC = 0.022) → pas de surapprentissage exploitable
   par split.

### Pourquoi pas les autres

- **Logistic Regression** : interprétable et rapide, mais ROC-AUC 0.74 et recall 0.55
  laissent passer ~45 % des churners → coût d'opportunité élevé. Conservée comme
  baseline méthodologique.
- **Random Forest** : très proche en F1 (0.37 vs 0.36) mais en retrait sur ROC-AUC
  (-1.4 pt) et PR-AUC (-0.7 pt). Modèle valable, second choix raisonnable.
- **MLP** : ROC-AUC 0.65, recall 0.34 — sous-performant sur ce jeu. Confirme la
  remarque pédagogique du sujet : **le Deep Learning n'est pas automatiquement
  supérieur**. Sur un dataset tabulaire de 10 k lignes avec features bien
  conçues, les modèles à arbres dominent. Sensible aux hyperparamètres, plus
  coûteux à entraîner, peu interprétable nativement.

### Compromis performance / interprétabilité

Le Gradient Boosting est moins interprétable qu'une régression logistique mais
le binding **SHAP** (`src/shap_analysis.py`, figures `results/shap_summary.png`
et `results/shap_importance.png`) restitue l'interprétabilité locale et globale.

### Top 10 features (|SHAP| moyen) — cohérence métier confirmée

| # | Feature                 | |SHAP|  | Lecture business |
|--:|-------------------------|--------:|------------------|
| 1 | csat_score              |  0.1031 | Satisfaction basse ⇒ risque ⤴ |
| 2 | tenure_months           |  0.0583 | Nouveaux clients = volatils |
| 3 | payment_failures        |  0.0523 | Échecs de paiement = signal critique |
| 4 | monthly_logins          |  0.0359 | Désengagement |
| 5 | last_login_days_ago     |  0.0152 | Inactivité récente |
| 6 | inactivity_risk         |  0.0148 | Feature engineerée — confirme l'utilité |
| 7 | avg_resolution_time     |  0.0123 | Support lent = irritant |
| 8 | features_used           |  0.0107 | Adoption produit |
| 9 | nps_score               |  0.0079 | Précurseur classique |
| 10 | marketing_click_rate    |  0.0063 | Engagement marketing |

Recommandations actionnables côté CRM :
- **Alerter** sur csat_score < 3 + tenure < 6 mois → cohort prioritaire
- **Re-contacter** automatiquement après ≥ 2 payment_failures consécutifs
- **Campagne réactivation** quand last_login_days_ago franchit le P90

## API d'inférence

Le modèle est exposé via FastAPI (`api/`) :
- `GET /health` → état du service
- `GET /model-info` → métadonnées (nom, version, seuil, features attendues)
- `POST /predict` → JSON client → `{churn, probability, threshold, risk_level}`

Lancement : `uv run uvicorn api.main:app`
Tests : `uv run pytest` (48 tests, 100 % branch coverage sur `api/`)
