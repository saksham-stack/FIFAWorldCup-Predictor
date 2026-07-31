from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.predict_match import MatchPredictor

MODEL_PATH = ROOT / 'models' / 'xgb_match_model.joblib'
STATES_PATH = ROOT / 'data' / 'processed' / 'latest_team_states.csv'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python scripts/predict_match.py "Argentina" "France" [neutral:true|false] [tournament]')
    team_a = sys.argv[1]
    team_b = sys.argv[2]
    neutral = sys.argv[3].lower() != 'false' if len(sys.argv) >= 4 else True
    tournament = sys.argv[4] if len(sys.argv) >= 5 else 'FIFA World Cup'
    predictor = MatchPredictor(MODEL_PATH, STATES_PATH)
    probs = predictor.predict_proba(team_a, team_b, neutral=neutral, tournament=tournament)
    print(probs)
