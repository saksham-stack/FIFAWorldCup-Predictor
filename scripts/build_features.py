from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
from src.features.build_match_features import build_training_features

RAW_RESULTS = ROOT / 'data' / 'raw' / 'results.csv'
OUT_FEATURES = ROOT / 'data' / 'processed' / 'match_features.csv'
OUT_STATES = ROOT / 'data' / 'processed' / 'latest_team_states.csv'

if __name__ == '__main__':
    if not RAW_RESULTS.exists():
        raise FileNotFoundError(f'Missing raw results file: {RAW_RESULTS}')
    results = pd.read_csv(RAW_RESULTS)
    features, states = build_training_features(results)
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FEATURES, index=False)
    states.to_csv(OUT_STATES, index=False)
    print(f'Saved features: {OUT_FEATURES} ({len(features):,} rows)')
    print(f'Saved states: {OUT_STATES} ({len(states):,} teams)')
