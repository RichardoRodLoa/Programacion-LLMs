import pandas as pd

def detectar_solapamientos(df, paciente_col, fecha_col, hora_inicio_col, duracion_min_col):
    """
    Detecta solapamientos de citas por paciente.

    Args:
        df (pd.DataFrame): DataFrame con citas.
        paciente_col (str): Nombre de la columna de paciente.
        fecha_col (str): Nombre de la columna de fecha.
        hora_inicio_col (str): Nombre de la columna de hora de inicio.
        duracion_min_col (str): Nombre de la columna de duración en minutos.

    Returns:
        pd.DataFrame: DataFrame con columnas originales + inicio_dt, fin_dt, solapada.
    """
    # Convertir fecha a datetime
    fecha_dt = pd.to_datetime(df[fecha_col], errors="coerce")

    # Construir inicio_dt combinando fecha + hora
    inicio_dt = pd.to_datetime(
        fecha_dt.dt.strftime("%Y-%m-%d") + " " + df[hora_inicio_col].astype(str),
        errors="coerce"
    )
    df = df.copy()
    df["inicio_dt"] = inicio_dt

    # Calcular fin_dt
    df["fin_dt"] = df["inicio_dt"] + pd.to_timedelta(df[duracion_min_col].astype(int), unit="m")

    # Ordenar por paciente y por inicio
    df = df.sort_values(by=[paciente_col, "inicio_dt"]).reset_index(drop=True)

    # Calcular fin anterior por paciente
    fin_anterior = df.groupby(paciente_col)["fin_dt"].shift(1)

    # Detectar solapamiento
    df["solapada"] = (df["inicio_dt"] < fin_anterior).fillna(False).astype(bool)

    return df
