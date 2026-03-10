import numpy as np
import pandas as pd


def agrupar_ventas(df, producto_col, cantidad_col):
    grouped = df.groupby(producto_col)[cantidad_col].sum().reset_index()
    grouped.rename(columns={cantidad_col: "total_vendido"}, inplace=True)
    return grouped


def generar_caso_de_uso_agrupar_ventas():
    productos = ["A", "B", "C"]
    df = pd.DataFrame({
        "producto": np.random.choice(productos, size=10),
        "cantidad": np.random.randint(1, 10, size=10),
        "fecha": pd.date_range("2021-01-01", periods=10)
    })
    input_dict = {"df": df, "producto_col": "producto", "cantidad_col": "cantidad"}
    output = agrupar_ventas(df, "producto", "cantidad")
    return input_dict, output
