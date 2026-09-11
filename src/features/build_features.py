import os
import pandas as pd
import numpy as np


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula promedios móviles por jugador ordenados cronológicamente por jornada (GW).
    Usa min_periods=1 para soportar inicios de temporada con pocos partidos.
    """
    df = df.sort_values(by=["element", "round"]).reset_index(drop=True)

    # Columnas para promedios móviles
    rolling_cols = ["total_points", "ict_index", "influence", "creativity", "threat", "minutes"]

    for col in rolling_cols:
        if col in df.columns:
            # Asegurar tipo numérico
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Rolling average de 3 partidos
            df[f"{col}_roll_3"] = (
                df.groupby("element")[col]
                .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
                .fillna(0)
            )
            
            # Rolling average de 5 partidos
            df[f"{col}_roll_5"] = (
                df.groupby("element")[col]
                .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
                .fillna(0)
            )

    return df


def process_features(raw_merged_filepath: str, output_filepath: str) -> pd.DataFrame:
    """
    Transforma el dataset procesado en un conjunto final de características para ML.
    """
    if not os.path.exists(raw_merged_filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {raw_merged_filepath}")

    df = pd.read_csv(raw_merged_filepath)

    df = build_rolling_features(df)

    if "chance_of_playing_next_round" in df.columns:
        df["chance_of_playing_next_round"] = (
            df["chance_of_playing_next_round"].fillna(100).astype(float)
        )
    else:
        df["chance_of_playing_next_round"] = 100.0

    # Ver si era local o visitante (1/0) para cada partido
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(int)
    else:
        df["was_home"] = 0

    # Proxy por Precio y Posición
    if "now_cost" in df.columns:
        df["now_cost"] = df["now_cost"] / 10.0  # Convertir a valor real (ej. 100 -> 10.0m)

    # 5. Selección de columnas de características finales (X) y Target (y)
    feature_columns = [
        "element",
        "round",
        "element_type",        # Posición
        "now_cost",            # Precio actual
        "was_home",            # Local/Visitante
        "opponent_team",       # ID rival para Dificultad del Rival (FDR)
        "chance_of_playing_next_round",
        "total_points_roll_3",
        "total_points_roll_5",
        "ict_index_roll_3",
        "ict_index_roll_5",
        "influence_roll_3",
        "creativity_roll_3",
        "threat_roll_3",
        "minutes_roll_3",
    ]

    target_column = "total_points"

    # Filtrar solo columnas existentes para prevenir errores
    available_features = [col for col in feature_columns if col in df.columns]
    
    output_cols = available_features + ([target_column] if target_column in df.columns else [])
    df_features = df[output_cols].copy()

    # Guardar dataset procesado con características
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_features.to_csv(output_filepath, index=False)
    print(f"Features generadas exitosamente y guardadas en: {output_filepath}")

    return df_features


if __name__ == "__main__":
    INPUT_PATH = "data/processed/merged_gw_data.csv"
    OUTPUT_PATH = "data/processed/dataset_features.csv"
    process_features(INPUT_PATH, OUTPUT_PATH)