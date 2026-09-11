import os
import time
import requests
import pandas as pd
# Configuración de la API y rutas de almacenamiento https://www.postman.com/fplassist/fpl-assist/request/jwu0n11/boostrap-static

BASE_URL = "https://fantasy.premierleague.com/api"
RAW_DATA_PATH = "data/raw"

def ensure_raw_directory():
    """Asegura que la carpeta data/raw exista."""
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

def fetch_bootstrap_static():
    """
    Obtiene los datos generales de la Fantasy Premier League:
    - Jugadores (elements)
    - Equipos (teams)
    - Jornadas (events)
    - Posiciones (element_types)
    """
    url = f"{BASE_URL}/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    ensure_raw_directory()
    
    # Guardar DataFrames principales
    df_players = pd.DataFrame(data["elements"])
    df_teams = pd.DataFrame(data["teams"])
    df_events = pd.DataFrame(data["events"])
    
    df_players.to_csv(f"{RAW_DATA_PATH}/players_raw.csv", index=False)
    df_teams.to_csv(f"{RAW_DATA_PATH}/teams_raw.csv", index=False)
    df_events.to_csv(f"{RAW_DATA_PATH}/events_raw.csv", index=False)
    
    print(f"[✓] Bootstrap static descargado: {len(df_players)} jugadores y {len(df_teams)} equipos.")
    return df_players

def fetch_fixtures():
    """Obtiene el calendario completo de partidos y su nivel de dificultad (FDR)."""
    url = f"{BASE_URL}/fixtures/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    ensure_raw_directory()
    df_fixtures = pd.DataFrame(data)
    df_fixtures.to_csv(f"{RAW_DATA_PATH}/fixtures_raw.csv", index=False)
    
    print(f"[✓] Calendario descargado: {len(df_fixtures)} partidos.")
    return df_fixtures

def fetch_player_histories(df_players, limit=None):
    """
    Obtiene el historial partido a partido de cada jugador.
    Nota: Usa 'limit' para hacer pruebas rápidas (ej. limit=10).
    """
    ensure_raw_directory()
    all_histories = []
    
    player_ids = df_players["id"].tolist()
    if limit:
        player_ids = player_ids[:limit]
        
    print(f"Descargando historial para {len(player_ids)} jugadores...")
    
    for idx, p_id in enumerate(player_ids):
        url = f"{BASE_URL}/element-summary/{p_id}/"
        res = requests.get(url)
        if res.status_code == 200:
            player_data = res.json()
            history = player_data.get("history", [])
            for game in history:
                game["player_id"] = p_id
            all_histories.extend(history)
        
        # Pausa ligera para evitar saturar la API
        time.sleep(0.05)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(player_ids):
            print(f"  Procesados {idx + 1}/{len(player_ids)} jugadores...")
            
    df_history = pd.DataFrame(all_histories)
    df_history.to_csv(f"{RAW_DATA_PATH}/players_history_raw.csv", index=False)
    print(f"[✓] Histórico de partidos guardado: {len(df_history)} registros de rendimiento.")
    return df_history

if __name__ == "__main__":
    print(">>> Iniciando el script fetcher.py...")
    try:
        players_df = fetch_bootstrap_static()
        fetch_fixtures()
        # Prueba primero descargando solo 10 jugadores para confirmar que funciona
        print(">>> Descargando muestra de historiales...")
        fetch_player_histories(players_df) 
        print(">>> ¡Proceso completado con éxito!")
    except Exception as e:
        print(f"[!] Ocurrió un error durante la ejecución: {e}")