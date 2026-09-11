import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_fpl_model(features_filepath: str, model_output_path: str):
    """
    Entrena un modelo XGBoost Regressor utilizando un split basado en tiempo
    para predecir los puntos esperados de cada jugador en FPL.
    """
    if not os.path.exists(features_filepath):
        raise FileNotFoundError(f"No se encontró el archivo de características: {features_filepath}")

    df = pd.read_csv(features_filepath)

    # Definir características (X) y variable objetivo (y)
    ignore_cols = ["element", "total_points"]
    feature_cols = [col for col in df.columns if col not in ignore_cols]
    
    target_col = "total_points"

    # Usamos las últimas 3 jornadas del dataset como set de validación
    max_round = df["round"].max()
    train_mask = df["round"] <= (max_round - 3)
    val_mask = df["round"] > (max_round - 3)

    # Caso especial: si no hay suficientes datos para validación, usar 80/20 split
    unique_rounds = sorted(df["round"].unique())

    if len(unique_rounds) > 3:
        # Si hay suficientes jornadas, usamos las últimas 2 para validación
        val_rounds = unique_rounds[-2:]
        train_mask = ~df["round"].isin(val_rounds)
        val_mask = df["round"].isin(val_rounds)
    else:
        # Si tenemos muy pocas jornadas (datos de prueba), usamos un split porcentual
        split_idx = int(len(df) * 0.8)
        train_mask = df.index < split_idx
        val_mask = df.index >= split_idx

    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, target_col]
    X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, target_col]

    print(f"Dataset cargado: {len(df)} registros totales.")
    print(f"Entrenamiento: {len(X_train)} filas (hasta GW{max_round - 3}).")
    print(f"Validación: {len(X_val)} filas (GW{max_round - 2} a GW{max_round}).")

    # Asignación de Pesos (Sample Weights): darle más valor a los partidos recientes
    # Partidos de la jornada actual tienen mayor peso que los de inicios de temporada
    sample_weights = np.exp(df.loc[train_mask, "round"] / max_round)

    # Configurar e Instanciar XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )

    # Entrenar con Early Stopping para evitar Overfitting
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluación del modelo
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    mae = mean_absolute_error(y_val, preds_val)

    print("\n Resultados de Validación:")
    print(f"   • MAE  (Error Absoluto Medio): {mae:.3f} puntos")
    print(f"   • RMSE (Raíz Error Cuadrático): {rmse:.3f} puntos")

    # Guardar el modelo entrenado
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    print(f"\n Modelo XGBoost guardado exitosamente en: {model_output_path}")


if __name__ == "__main__":
    FEATURES_PATH = "data/processed/dataset_features.csv"
    MODEL_PATH = "models/xgboost_fpl.json"
    train_fpl_model(FEATURES_PATH, MODEL_PATH)