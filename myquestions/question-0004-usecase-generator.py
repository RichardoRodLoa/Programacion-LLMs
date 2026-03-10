import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def evaluar_regresion_lineal(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return r2


def generar_caso_de_uso_evaluar_regresion_lineal():
    df = pd.DataFrame({
        "feat1": np.random.randn(15),
        "feat2": np.random.randn(15),
        "target": np.random.randn(15) * 10 + 5
    })
    input_dict = {"df": df, "target_col": "target"}
    output = evaluar_regresion_lineal(df, "target")
    return input_dict, output
