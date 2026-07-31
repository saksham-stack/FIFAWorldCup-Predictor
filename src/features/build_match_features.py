from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque

import numpy as np
import pandas as pd

BASE_ELO = 1500.0
ROLLING_N = 5
HOME_ADV_ELO = 70.0

TOURNAMENT_IMPORTANCE = {
    'fifa world cup': 1.00,
    'world cup qualification': 0.75,
    'uefa euro': 0.90,
    'uefa euro qualification': 0.70,
    'copa américa': 0.88,
    'african cup of nations': 0.85,
    'africa cup of nations': 0.85,
    'afc asian cup': 0.85,
    'concacaf championship': 0.82,
    'gold cup': 0.82,
    'confederations cup': 0.84,
    'nations league': 0.68,
    'friendly': 0.35,
}


@dataclass
class TeamState:
    elo: float = BASE_ELO
    matches: int = 0
    recent_points: Deque[float] | None = None
    recent_gf: Deque[float] | None = None
    recent_ga: Deque[float] | None = None
    recent_gd: Deque[float] | None = None
    last_date: datetime | None = None

    def __post_init__(self):
        self.recent_points = self.recent_points or deque(maxlen=ROLLING_N)
        self.recent_gf = self.recent_gf or deque(maxlen=ROLLING_N)
        self.recent_ga = self.recent_ga or deque(maxlen=ROLLING_N)
        self.recent_gd = self.recent_gd or deque(maxlen=ROLLING_N)


def tournament_weight(name: str) -> float:
    text = (name or '').strip().lower()
    for key, value in TOURNAMENT_IMPORTANCE.items():
        if key in text:
            return value
    return 0.55


def expected_score(elo_a: float, elo_b: float, home_advantage: float = 0.0) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_a + home_advantage) - elo_b) / 400.0))


def k_factor(weight: float) -> float:
    return 18.0 + 42.0 * weight


def _avg(values: Deque[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _rest_days(state: TeamState, current_date: datetime) -> int:
    if state.last_date is None:
        return 14
    return max((current_date - state.last_date).days, 0)


def _points_for_result(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _build_row(team_a: str, team_b: str, team_a_state: TeamState, team_b_state: TeamState,
               current_date: datetime, tournament: str, neutral: bool, team_a_home_adv_flag: int,
               target: int) -> dict:
    importance = tournament_weight(tournament)
    team_a_rest = _rest_days(team_a_state, current_date)
    team_b_rest = _rest_days(team_b_state, current_date)
    exp_a = expected_score(team_a_state.elo, team_b_state.elo, HOME_ADV_ELO * team_a_home_adv_flag)
    return {
        'date': current_date.date().isoformat(),
        'team_a': team_a,
        'team_b': team_b,
        'tournament': tournament,
        'tournament_importance': importance,
        'neutral': int(neutral),
        'team_a_home_advantage': int(team_a_home_adv_flag),
        'team_a_elo': round(team_a_state.elo, 3),
        'team_b_elo': round(team_b_state.elo, 3),
        'elo_diff': round(team_a_state.elo - team_b_state.elo, 3),
        'team_a_matches': team_a_state.matches,
        'team_b_matches': team_b_state.matches,
        'matches_diff': team_a_state.matches - team_b_state.matches,
        'team_a_points_avg_5': round(_avg(team_a_state.recent_points), 4),
        'team_b_points_avg_5': round(_avg(team_b_state.recent_points), 4),
        'points_form_diff_5': round(_avg(team_a_state.recent_points) - _avg(team_b_state.recent_points), 4),
        'team_a_gf_avg_5': round(_avg(team_a_state.recent_gf), 4),
        'team_b_gf_avg_5': round(_avg(team_b_state.recent_gf), 4),
        'gf_diff_5': round(_avg(team_a_state.recent_gf) - _avg(team_b_state.recent_gf), 4),
        'team_a_ga_avg_5': round(_avg(team_a_state.recent_ga), 4),
        'team_b_ga_avg_5': round(_avg(team_b_state.recent_ga), 4),
        'ga_diff_5': round(_avg(team_a_state.recent_ga) - _avg(team_b_state.recent_ga), 4),
        'team_a_gd_avg_5': round(_avg(team_a_state.recent_gd), 4),
        'team_b_gd_avg_5': round(_avg(team_b_state.recent_gd), 4),
        'gd_diff_5': round(_avg(team_a_state.recent_gd) - _avg(team_b_state.recent_gd), 4),
        'team_a_rest_days': team_a_rest,
        'team_b_rest_days': team_b_rest,
        'rest_days_diff': team_a_rest - team_b_rest,
        'expected_score_elo': round(exp_a, 6),
        'target': target,  # 0=loss, 1=draw, 2=win from team_a perspective
    }


def _ensure_state(states: dict, team: str) -> TeamState:
    if team not in states:
        states[team] = TeamState()
    return states[team]


def build_training_features(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = results.copy()
    results['date'] = pd.to_datetime(results['date'])
    results = results.dropna(subset=['home_score', 'away_score']).copy()
    results = results.sort_values(['date', 'home_team', 'away_team']).reset_index(drop=True)

    states: dict[str, TeamState] = {}
    rows = []

    for _, match in results.iterrows():
        date = match['date']
        home = match['home_team']
        away = match['away_team']
        home_score = int(match['home_score'])
        away_score = int(match['away_score'])
        tournament = match['tournament']
        neutral = bool(match['neutral'])

        home_state = _ensure_state(states, home)
        away_state = _ensure_state(states, away)

        if home_score > away_score:
            home_target, away_target = 2, 0
            result_home, result_away = 1.0, 0.0
        elif home_score < away_score:
            home_target, away_target = 0, 2
            result_home, result_away = 0.0, 1.0
        else:
            home_target, away_target = 1, 1
            result_home = result_away = 0.5

        rows.append(_build_row(home, away, home_state, away_state, date, tournament, neutral, 0 if neutral else 1, home_target))
        rows.append(_build_row(away, home, away_state, home_state, date, tournament, neutral, 0 if neutral else -1, away_target))

        imp = tournament_weight(tournament)
        k = k_factor(imp)
        exp_home = expected_score(home_state.elo, away_state.elo, 0.0 if neutral else HOME_ADV_ELO)
        exp_away = 1.0 - exp_home
        home_state.elo += k * (result_home - exp_home)
        away_state.elo += k * (result_away - exp_away)

        home_state.matches += 1
        away_state.matches += 1
        home_state.recent_points.append(_points_for_result(home_score, away_score))
        away_state.recent_points.append(_points_for_result(away_score, home_score))
        home_state.recent_gf.append(home_score)
        home_state.recent_ga.append(away_score)
        home_state.recent_gd.append(home_score - away_score)
        away_state.recent_gf.append(away_score)
        away_state.recent_ga.append(home_score)
        away_state.recent_gd.append(away_score - home_score)
        home_state.last_date = date
        away_state.last_date = date

    features = pd.DataFrame(rows)
    latest_states = []
    for team, state in states.items():
        latest_states.append({
            'team': team,
            'elo': round(state.elo, 3),
            'matches': state.matches,
            'points_avg_5': round(_avg(state.recent_points), 4),
            'gf_avg_5': round(_avg(state.recent_gf), 4),
            'ga_avg_5': round(_avg(state.recent_ga), 4),
            'gd_avg_5': round(_avg(state.recent_gd), 4),
            'last_date': state.last_date.date().isoformat() if state.last_date is not None else None,
        })
    states_df = pd.DataFrame(latest_states).sort_values('elo', ascending=False).reset_index(drop=True)
    return features, states_df


def build_prediction_row(team_a: str, team_b: str, tournament: str, neutral: bool,
                         latest_states: pd.DataFrame, match_date: str | None = None) -> pd.DataFrame:
    match_dt = pd.to_datetime(match_date or pd.Timestamp.today().date())
    states = latest_states.set_index('team').to_dict(orient='index')

    def team_vector(team_name: str) -> dict:
        candidates = [part.strip() for part in team_name.split('/') if part.strip()]
        rows = []
        for candidate in candidates:
            if candidate in states:
                rows.append(states[candidate])
        if rows:
            frame = pd.DataFrame(rows)
            out = frame.mean(numeric_only=True).to_dict()
            out['team'] = team_name
            return out
        return {
            'team': team_name,
            'elo': BASE_ELO,
            'matches': 0,
            'points_avg_5': 1.0,
            'gf_avg_5': 1.0,
            'ga_avg_5': 1.0,
            'gd_avg_5': 0.0,
            'last_date': None,
        }

    a = team_vector(team_a)
    b = team_vector(team_b)
    a_last = pd.to_datetime(a['last_date']) if a.get('last_date') else None
    b_last = pd.to_datetime(b['last_date']) if b.get('last_date') else None
    a_rest = max((match_dt - a_last).days, 0) if a_last is not None else 14
    b_rest = max((match_dt - b_last).days, 0) if b_last is not None else 14
    home_flag = 0 if neutral else 1
    imp = tournament_weight(tournament)
    exp = expected_score(float(a['elo']), float(b['elo']), 0.0 if neutral else HOME_ADV_ELO)

    return pd.DataFrame([{
        'tournament_importance': imp,
        'neutral': int(neutral),
        'team_a_home_advantage': home_flag,
        'team_a_elo': float(a['elo']),
        'team_b_elo': float(b['elo']),
        'elo_diff': float(a['elo']) - float(b['elo']),
        'team_a_matches': int(a['matches']),
        'team_b_matches': int(b['matches']),
        'matches_diff': int(a['matches']) - int(b['matches']),
        'team_a_points_avg_5': float(a['points_avg_5']),
        'team_b_points_avg_5': float(b['points_avg_5']),
        'points_form_diff_5': float(a['points_avg_5']) - float(b['points_avg_5']),
        'team_a_gf_avg_5': float(a['gf_avg_5']),
        'team_b_gf_avg_5': float(b['gf_avg_5']),
        'gf_diff_5': float(a['gf_avg_5']) - float(b['gf_avg_5']),
        'team_a_ga_avg_5': float(a['ga_avg_5']),
        'team_b_ga_avg_5': float(b['ga_avg_5']),
        'ga_diff_5': float(a['ga_avg_5']) - float(b['ga_avg_5']),
        'team_a_gd_avg_5': float(a['gd_avg_5']),
        'team_b_gd_avg_5': float(b['gd_avg_5']),
        'gd_diff_5': float(a['gd_avg_5']) - float(b['gd_avg_5']),
        'team_a_rest_days': int(a_rest),
        'team_b_rest_days': int(b_rest),
        'rest_days_diff': int(a_rest) - int(b_rest),
        'expected_score_elo': float(exp),
    }])


MODEL_FEATURE_COLUMNS = [
    'tournament_importance', 'neutral', 'team_a_home_advantage', 'team_a_elo', 'team_b_elo', 'elo_diff',
    'team_a_matches', 'team_b_matches', 'matches_diff', 'team_a_points_avg_5', 'team_b_points_avg_5',
    'points_form_diff_5', 'team_a_gf_avg_5', 'team_b_gf_avg_5', 'gf_diff_5', 'team_a_ga_avg_5',
    'team_b_ga_avg_5', 'ga_diff_5', 'team_a_gd_avg_5', 'team_b_gd_avg_5', 'gd_diff_5',
    'team_a_rest_days', 'team_b_rest_days', 'rest_days_diff', 'expected_score_elo'
]
