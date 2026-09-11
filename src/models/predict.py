import os
import pandas as pd
import xgboost as xgb


def generate_predictions(dataset_path: str, model_path: str, output_path: str) -> pd.DataFrame:
    """Carga el modelo XGBoost guardado y genera predicciones de puntos (xP)
    para la siguiente jornada ajustadas por disponibilidad.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"No se encontró el dataset de características: {dataset_path}"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo: {model_path}"
        )

    df = pd.read_csv(dataset_path)

    # Cargar el modelo
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # Filtrar para obtener solo los datos de la última jornada disponible por jugador = a la condición con la que encarará el próximo partido
    df_latest = df.sort_values("round").groupby("element").last().reset_index()

    # Seleccionar exactamente las mismas columnas usadas en el entrenamiento
    ignore_cols = ["element", "total_points"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    X_predict = df_latest[feature_cols]

    # Generar predicciones de puntos base (Unadjusted xP)
    df_latest["predicted_points_raw"] = model.predict(X_predict)

    # Ajustar por probabilidad de juego ( chance_of_playing_next_round )
    # Si viene vacio, asumimos 100% de disponibilidad (1.0)
    if "chance_of_playing_next_round" in df_latest.columns:
        df_latest["chance_of_playing_next_round"] = (
            df_latest["chance_of_playing_next_round"].fillna(100) / 100.0
        )
    else:
        df_latest["chance_of_playing_next_round"] = 1.0

    # Puntos esperados finales = Predicción * Probabilidad de jugar
    df_latest["expected_points"] = (
        df_latest["predicted_points_raw"]
        * df_latest["chance_of_playing_next_round"]
    )

    df_latest["expected_points"] = df_latest["expected_points"].clip(lower=0.0)

    # Cargar metadata de los jugadores para enriquecer la salida (Nombre, Precio, Posición)
    players_clean_path = "data/processed/players_clean.csv"
    if os.path.exists(players_clean_path):
        df_players = pd.read_csv(players_clean_path)
        output_df = df_latest[
            ["element", "predicted_points_raw", "expected_points"]].merge(df_players[["id", "web_name", "position", "team_name", "now_cost"]],
            left_on="element",
            right_on="id",
            how="inner",
        )
    else:
        output_df = df_latest

    # Ordenar por puntos esperados proyectados
    output_df = output_df.sort_values(by="expected_points", ascending=False).reset_index(drop=True)

    # Exportar resultados
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Predicciones completadas para {len(output_df)} jugadores.")
    print(f"Guardado en: {output_path}")

    # Mostrar Top 10 jugadores con mayor proyección
    print("\n Top 10 Jugadores Proyectados para la Siguiente Jornada:")
    print(output_df[["web_name", "position", "team_name", "now_cost", "expected_points"]].head(50))

    return output_df


if __name__ == "__main__":
    DATASET_PATH = "data/processed/dataset_features.csv"
    MODEL_PATH = "models/xgboost_fpl.json"
    OUTPUT_PATH = "data/processed/predictions.csv"

    generate_predictions(DATASET_PATH, MODEL_PATH, OUTPUT_PATH)