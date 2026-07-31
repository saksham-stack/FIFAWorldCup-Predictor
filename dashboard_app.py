from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from src.models.predict_match import MatchPredictor

if __name__ == '__main__' and get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit('Run this app with `streamlit run dashboard_app.py` from the project root.')

st.set_page_config(page_title='World Cup 2026 Predictor', page_icon='⚽', layout='wide')
st.title('⚽ FIFA World Cup 2026 Match & Title Predictor')

model_path = ROOT / 'models' / 'xgb_match_model.joblib'
states_path = ROOT / 'data' / 'processed' / 'latest_team_states.csv'
title_odds_path = ROOT / 'reports' / 'worldcup_2026_title_odds.csv'

if not model_path.exists() or not states_path.exists():
    st.error('Model artifacts not found. Run scripts/run_pipeline.py first.')
    st.stop()

predictor = MatchPredictor(model_path, states_path)
teams = sorted(predictor.latest_states['team'].unique().tolist())

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox('Team A', teams, index=teams.index('Argentina') if 'Argentina' in teams else 0)
with col2:
    team_b = st.selectbox('Team B', teams, index=teams.index('France') if 'France' in teams else 1)
neutral = st.checkbox('Neutral venue', value=True)
tournament = st.selectbox('Tournament type', ['FIFA World Cup', 'World Cup qualification', 'Friendly'])

if st.button('Predict match'):
    probs = predictor.predict_proba(team_a, team_b, tournament=tournament, neutral=neutral)
    out = pd.DataFrame([
        {'outcome': f'{team_a} win', 'probability': probs['team_a_win']},
        {'outcome': 'Draw', 'probability': probs['draw']},
        {'outcome': f'{team_b} win', 'probability': probs['team_a_loss']},
    ])
    out['probability'] = (100 * out['probability']).round(2)
    st.dataframe(out, use_container_width=True)
    st.bar_chart(out.set_index('outcome'))

st.markdown('## Provisional World Cup 2026 title odds')
if title_odds_path.exists():
    title_odds = pd.read_csv(title_odds_path)
    st.dataframe(title_odds.head(20), use_container_width=True)
else:
    st.info('Run scripts/predict_worldcup_2026.py to generate title odds.')
