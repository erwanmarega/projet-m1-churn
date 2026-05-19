import pandas as pd


COLS_TO_DROP = ["customer_id", "city", "country"]
TARGET = "churn"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tickets_per_month"] = df["support_tickets"] / (df["tenure_months"] + 1)
    df["revenue_per_month"] = df["total_revenue"] / (df["tenure_months"] + 1)
    df["engagement_score"] = (df["email_open_rate"] + df["marketing_click_rate"]) / 2
    df["support_burden"] = df["support_tickets"] * df["avg_resolution_time"]
    df["inactivity_risk"] = df["last_login_days_ago"] / (df["monthly_logins"] + 1)
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
