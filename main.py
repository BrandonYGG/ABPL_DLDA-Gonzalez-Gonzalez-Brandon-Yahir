from src.preprocessing import load_data, remove_outliers, normalize_data
from src.lda_model import train_lda
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if __name__ == "__main__":
    # Cargar datos
    df = load_data("data/raw/credit_data.csv")
    
    # Preprocesamiento
    df = remove_outliers(df, "Ingresos")
    df = normalize_data(df, ["Ingresos", "Deuda", "Edad"])
    
    # Entrenar modelo
    X = df[["Ingresos", "Deuda", "Edad", "HistorialCrediticio"]]
    y = df["Clase"]
    lda, X_test, y_test = train_lda(X, y)
    
    # Generar matriz de confusión
    y_pred = lda.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("results/plots/confusion_matrix.png")  # Guardar gráfico