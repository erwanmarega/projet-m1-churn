import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.features import engineer_features, drop_unused_columns, TARGET


# ordre logique : engagement croissant
CONTRACT_ORDER = [["Monthly", "Yearly", "Two-Year"]]

CATEGORICAL_NOMINAL = [
    "gender",
    "customer_segment",
    "signup_channel",
    "payment_method",
    "discount_applied",
    "price_increase_last_3m",
    "survey_response",
]

# NaN = absence de plainte, pas une valeur inconnue
COMPLAINT_TYPE = ["complaint_type"]

CATEGORICAL_ORDINAL = ["contract_type"]

NUMERICAL = [
    "age",
    "tenure_months",
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
    "monthly_fee",
    "total_revenue",
    "payment_failures",
    "avg_resolution_time",
    "csat_score",
    "escalations",
    "email_open_rate",
    "marketing_click_rate",
    "nps_score",
    "referral_count",
    "tickets_per_month",
    "revenue_per_month",
    "engagement_score",
    "support_burden",
    "inactivity_risk",
]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = drop_unused_columns(df)
    df = engineer_features(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    complaint_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="No Complaint")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=CONTRACT_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERICAL),
        ("nom", nominal_pipe, CATEGORICAL_NOMINAL),
        ("complaint", complaint_pipe, COMPLAINT_TYPE),
        ("ord", ordinal_pipe, CATEGORICAL_ORDINAL),
    ], remainder="drop")
