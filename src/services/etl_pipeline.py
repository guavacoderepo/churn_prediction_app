from fastapi import HTTPException
import pandas as pd
import joblib


class ETLPipeline:
    """ETL pipeline to extract, transform, and scale customer churn data."""

    def __init__(self, source: str = "../../dataset/Customer-churn-datase.csv") -> None:
        self.scaler = joblib.load("data/scaler.save")
        self.source = source

    def extract_data(self) -> pd.DataFrame:
        """Read historical data from CSV."""
        try:
            return pd.read_csv(self.source)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Data extraction error: {e}")

    def transform_data(self, data: pd.DataFrame | None = None):
        """
        Transform and scale input data.
        If no data provided, extract from source.
        """
        try:
            df = data if isinstance(data, pd.DataFrame) else self.extract_data()

            # Drop duplicates, missing values, and unnecessary column
            df = df.drop_duplicates(keep="first").drop(columns="customerID", axis=1).dropna(axis=0)

            # Ordinal yes/no columns
            yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
            df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})

            # Contract mapping
            df["Contract"] = df["Contract"].replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

            # Group mapping
            group_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            df[group_cols] = df[group_cols].replace({'No': 0, 'Yes': 1, 'No internet service': 3})

            # Gender mapping
            df["gender"] = df["gender"].replace({"Male": 1, "Female": 0})

            # Internet service mapping
            df["InternetService"] = df["InternetService"].replace({'DSL': 0, 'Fiber optic': 1, 'No': 3})

            # Multiple lines mapping
            df["MultipleLines"] = df["MultipleLines"].replace({'No': 0, 'Yes': 1, 'No phone service': 3})

            # Payment method mapping
            df["PaymentMethod"] = df["PaymentMethod"].replace({
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 3,
                'Credit card (automatic)': 4
            })

            # Scale data
            return self.scaler.transform(df)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Data transformation error: {e}")
