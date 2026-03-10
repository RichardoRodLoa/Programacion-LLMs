import numpy as np
import pandas as pd


def limpiar_y_ordenar_estudiantes(df, nombre_col, nota_col):
    df_clean = df.drop_duplicates(subset=[nombre_col])
    df_sorted = df_clean.sort_values(by=nota_col, ascending=False)
    return df_sorted


def generar_caso_de_uso_limpiar_y_ordenar_estudiantes():
    nombres = ["Ana", "Luis", "Marta", "Pedro"]
    df = pd.DataFrame({
        "nombre": np.random.choice(nombres, size=8),
        "edad": np.random.randint(18, 25, size=8),
        "nota": np.random.uniform(0, 10, size=8)
    })
    input_dict = {"df": df, "nombre_col": "nombre", "nota_col": "nota"}
    output = limpiar_y_ordenar_estudiantes(df, "nombre", "nota")
    return input_dict, output
