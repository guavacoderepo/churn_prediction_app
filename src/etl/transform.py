from fastapi import HTTPException
import pandas as pd
import joblib

def transform_data(data):
    try:
        # Convert incoming dict to DataFrame
        df = pd.DataFrame(data)

        # Feature engineering
        df = df.drop_duplicates(keep="first")
        df = df.drop(columns="customerID", axis=1)
        df = df.dropna(axis=0)

        # Columns with yes/no
        ordinal_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in ordinal_cols:
            df[col] = df[col].map({"Yes": 1, "No": 0})

        # Contract mapping
        contract_ord = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df.Contract = df.Contract.map(contract_ord)

        # Group mapping
        group_ord = {'No': 0, 'Yes': 1, 'No internet service': 3}
        group_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for col in group_cols:
            df[col] = df[col].map(group_ord)

        # Gender mapping
        df.gender = df.gender.map({"Male": 1, "Female": 0})

        # Internet service mapping
        internet_ord = {'DSL': 0, 'Fiber optic': 1, 'No': 3}
        df.InternetService = df.InternetService.map(internet_ord)

        # Multiple lines mapping
        multiplelines_ord = {'No': 0, 'Yes': 1, 'No phone service': 3}
        df.MultipleLines = df.MultipleLines.map(multiplelines_ord)

        # Payment method mapping
        payment_ord = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 3, "Credit card (automatic)": 4}
        df.PaymentMethod = df.PaymentMethod.map(payment_ord)

        # Load scaler and transform
        scaler = joblib.load("data/scaler.save")
        df_scaled = scaler.transform(df)

        return df_scaled
    
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(status_code=500, detail=f"Data transformation error: {str(e)}")
