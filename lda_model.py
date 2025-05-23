import pandas as pd  # <-- Añadir esta línea
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def train_lda(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    """Entrena un modelo LDA y devuelve métricas."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42  # Semilla para reproducibilidad
    )
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda, X_test, y_test