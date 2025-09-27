from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
from pandas import DataFrame
import pandas as pd
import joblib


pd.set_option('future.no_silent_downcasting', True)

class ETLPipeline:
    """ETL pipeline to extract, transform, and scale customer churn data."""

    def __init__(self, source: Union[str, DataFrame] = "dataset/Customer-churn-datase.csv") -> None:
        self.scaler: Optional[StandardScaler] = None
        self.source = source
    
    def load_scaler(self):
        if self.scaler is None:
            try:
                self.scaler = joblib.load("src/data/scaler.save")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Scaler load error: {e}")

    def extract_data(self) -> pd.DataFrame:
        """
        Read historical data from CSV.
        Connect to a database base to retrieve recent data eg. redis, SQL, NOSQL or Kafka for streaming
        """
        try:
            if isinstance(self.source, DataFrame):
                return self.source
            return pd.read_csv(self.source)
        except Exception as e:
            raise Exception(f"Data extraction error: {e}")

    def transform_data(self, data: pd.DataFrame | None = None):
        """
        Transform and scale input data.
        If no data provided, extract from source.
        """
        try:
            if self.scaler is None:
                self.load_scaler()

            df = data if isinstance(data, pd.DataFrame) else self.extract_data()

            df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

            # Drop duplicates, missing values, and unnecessary column
            df = df.drop_duplicates(keep="first").drop(columns="customerID", axis=1).dropna(axis=0)

            # Ordinal yes/no columns
            yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
            df[yes_no_cols] = df[yes_no_cols].replace({"Yes": 1, "No": 0})

            # Contract mapping
            df["Contract"] = df["Contract"].replace({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

            # Group mapping
            group_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            df[group_cols] = df[group_cols].replace({'No': 0, 'Yes': 1, 'No internet service': 3}).infer_objects(copy=False)

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

             # Separate target
            if "Churn" in df.columns:
                df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
                y = df[["Churn"]]
                X = df.drop(columns="Churn")
            else:
                y = None
                X = df
            
            # Scale data
            X_scaled = self.scaler.transform(X) # type: ignore
            return X_scaled, y
        except Exception as e:
            raise Exception(f"Data transformation error: {e}")

    def tranform_split_data(self):
        X, y = self.transform_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


