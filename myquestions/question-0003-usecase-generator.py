import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def entrenar_logistica(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict(X)
    return preds


def generar_caso_de_uso_entrenar_logistica():
    df = pd.DataFrame({
        "feat1": np.random.randn(20),
        "feat2": np.random.randn(20),
        "target": np.random.choice([0, 1], size=20)
    })
    input_dict = {"df": df, "target_col": "target"}
    output = entrenar_logistica(df, "target")
    return input_dict, output
