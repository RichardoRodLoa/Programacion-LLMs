import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def calcular_dependencia_parcial(df, target_col, feature_objetivo, grid_size=10):
    """
    Calcula manualmente la dependencia parcial de una variable en un modelo RandomForestClassifier.

    Args:
        df (pd.DataFrame): DataFrame con variables numéricas y columna objetivo binaria.
        target_col (str): Nombre de la columna objetivo.
        feature_objetivo (str): Nombre de la variable a analizar.
        grid_size (int): Número de puntos en la grilla (default=10).

    Returns:
        dict: Contiene 'valores_feature', 'predicciones_promedio', 'nombre_feature'.
    """
    # Separar X y y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Crear grilla de valores para la feature objetivo
    feature_values = X[feature_objetivo]
    grid = np.linspace(feature_values.min(), feature_values.max(), grid_size)

    # Calcular predicciones promedio para cada valor de la grilla
    preds = []
    for val in grid:
        X_temp = X.copy()
        X_temp[feature_objetivo] = val
        prob = model.predict_proba(X_temp)[:, 1].mean()
        preds.append(float(prob))

    return {
        "valores_feature": grid,
        "predicciones_promedio": np.array(preds),
        "nombre_feature": feature_objetivo
    }
