from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from src.models.predict_match import MatchPredictor
from src.simulation.worldcup_simulator import WorldCup2026Simulator

MODEL_PATH = ROOT / 'models' / 'xgb_match_model.joblib'
STATES_PATH = ROOT / 'data' / 'processed' / 'latest_team_states.csv'
GROUPS_PATH = ROOT / 'data' / 'processed' / 'worldcup_2026_groups.csv'
TITLE_ODDS = ROOT / 'reports' / 'worldcup_2026_title_odds.csv'

if __name__ == '__main__':
    predictor = MatchPredictor(MODEL_PATH, STATES_PATH)
    simulator = WorldCup2026Simulator(predictor, GROUPS_PATH, seed=42)
    title_odds, groups = simulator.simulate_many(n_sims=300)
    TITLE_ODDS.parent.mkdir(parents=True, exist_ok=True)
    title_odds.to_csv(TITLE_ODDS, index=False)
    print('Top 15 title probabilities (provisional simulator, 300 simulations):')
    print(title_odds.head(15).to_string(index=False))
    print(f'Saved title odds to: {TITLE_ODDS}')
