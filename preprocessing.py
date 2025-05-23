import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Carga datos desde un archivo CSV."""
    return pd.read_csv(path)

def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Elimina outliers usando el método IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

def normalize_data(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Normaliza variables numéricas usando Z-Score."""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df