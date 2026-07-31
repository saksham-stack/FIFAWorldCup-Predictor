from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from src.features.build_match_features import MODEL_FEATURE_COLUMNS, build_prediction_row


class MatchPredictor:
    def __init__(self, bundle_path: str | Path, latest_states_path: str | Path):
        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle['model']
        self.feature_cols = self.bundle.get('feature_cols', MODEL_FEATURE_COLUMNS)
        self.label_map = {0: 'team_a_loss', 1: 'draw', 2: 'team_a_win'}
        self.latest_states = pd.read_csv(latest_states_path)

    def predict_proba(self, team_a: str, team_b: str, tournament: str = 'FIFA World Cup', neutral: bool = True, match_date: str | None = None) -> dict:
        X = build_prediction_row(team_a, team_b, tournament, neutral, self.latest_states, match_date)[self.feature_cols]
        probs = self.model.predict_proba(X)[0]
        return {
            'team_a': team_a,
            'team_b': team_b,
            'team_a_loss': float(probs[0]),
            'draw': float(probs[1]),
            'team_a_win': float(probs[2]),
        }
