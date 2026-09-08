import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

def process_raw_data():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    df_players = pd.read_csv(f"{RAW_PATH}/players_raw.csv")
    df_teams = pd.read_csv(f"{RAW_PATH}/teams_raw.csv")
    
    # Mapeo de equipos y posiciones (1: GKP, 2: DEF, 3: MID, 4: FWD)
    team_map = dict(zip(df_teams["id"], df_teams["name"]))
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    
    df_players["team_name"] = df_players["team"].map(team_map)
    df_players["position"] = df_players["element_type"].map(pos_map)
    df_players["now_cost"] = df_players["now_cost"] / 10.0  # El precio viene multiplicado por 10
    
    df_players.to_csv(f"{PROCESSED_PATH}/players_clean.csv", index=False)
    print(f"[✓] Jugadores procesados: {len(df_players)} guardados en {PROCESSED_PATH}/players_clean.csv")

if __name__ == "__main__":
    process_raw_data()