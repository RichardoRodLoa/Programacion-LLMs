import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def calcular_estabilidad_importancia(df, target_col, n_iteraciones):
    """
    Entrena varias instancias de un RandomForestRegressor con diferentes random_state
    y calcula la desviación estándar de las importancias de las variables.

    Args:
        df (pd.DataFrame): DataFrame con variables predictoras y la columna objetivo.
        target_col (str): Nombre de la columna objetivo.
        n_iteraciones (int): Número de iteraciones (semillas distintas).

    Returns:
        np.ndarray: Desviación estándar de las importancias por variable.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    importancias = []

    for semilla in range(n_iteraciones):
        modelo = RandomForestRegressor(
            n_estimators=120,
            random_state=semilla,
            n_jobs=-1
        )
        modelo.fit(df[feature_cols], df[target_col])
        importancias.append(modelo.feature_importances_)

    return np.std(np.array(importancias), axis=0)
