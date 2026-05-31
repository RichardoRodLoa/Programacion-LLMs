import pandas as pd

def balancear_fallas_criticas(df, target_col):
    """
    Balancea un dataset duplicando filas de la clase minoritaria
    hasta igualar el tamaño de la clase mayoritaria.

    Args:
        df (pd.DataFrame): DataFrame con columna objetivo.
        target_col (str): Nombre de la columna objetivo.

    Returns:
        pd.DataFrame: DataFrame balanceado.
    """
    # Contar ocurrencias de cada clase
    counts = df[target_col].value_counts()
    major, minor = counts.idxmax(), counts.idxmin()

    # Separar clases
    df_major = df[df[target_col] == major]
    df_minor = df[df[target_col] == minor]

    # Sobremuestreo de la clase minoritaria
    df_minor_up = df_minor.sample(len(df_major), replace=True, random_state=42)

    # Concatenar y mezclar
    df_balanced = pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced
