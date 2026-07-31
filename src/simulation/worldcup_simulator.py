from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.models.predict_match import MatchPredictor


@dataclass
class TeamStanding:
    team: str
    group: str
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


COMMON_WIN_SCORES = [(1, 0), (2, 0), (2, 1), (3, 1), (3, 0)]
COMMON_DRAW_SCORES = [(0, 0), (1, 1), (2, 2)]
WIN_WEIGHTS = [0.30, 0.22, 0.28, 0.12, 0.08]
DRAW_WEIGHTS = [0.28, 0.58, 0.14]


class WorldCup2026Simulator:
    def __init__(self, predictor: MatchPredictor, groups_path: str | Path, seed: int = 42):
        self.predictor = predictor
        self.groups = pd.read_csv(groups_path)
        self.rng = random.Random(seed)
        self.elo_lookup = predictor.latest_states.set_index('team')['elo'].to_dict()

    def _sample_score(self, probs: dict, team_a: str, team_b: str) -> tuple[int, int, str]:
        outcome = self.rng.choices(['team_a_loss', 'draw', 'team_a_win'], weights=[probs['team_a_loss'], probs['draw'], probs['team_a_win']], k=1)[0]
        if outcome == 'team_a_win':
            score = self.rng.choices(COMMON_WIN_SCORES, weights=WIN_WEIGHTS, k=1)[0]
            return score[0], score[1], team_a
        if outcome == 'team_a_loss':
            score = self.rng.choices(COMMON_WIN_SCORES, weights=WIN_WEIGHTS, k=1)[0]
            return score[1], score[0], team_b
        score = self.rng.choices(COMMON_DRAW_SCORES, weights=DRAW_WEIGHTS, k=1)[0]
        return score[0], score[1], 'draw'

    def _simulate_group_stage(self):
        standings = {row.team: TeamStanding(team=row.team, group=row.group) for row in self.groups.itertuples(index=False)}
        match_rows = []
        for group, sub in self.groups.groupby('group'):
            teams = sub['team'].tolist()
            for team_a, team_b in combinations(teams, 2):
                probs = self.predictor.predict_proba(team_a, team_b, tournament='FIFA World Cup', neutral=True)
                goals_a, goals_b, winner = self._sample_score(probs, team_a, team_b)
                sa = standings[team_a]
                sb = standings[team_b]
                sa.gf += goals_a; sa.ga += goals_b
                sb.gf += goals_b; sb.ga += goals_a
                if goals_a > goals_b:
                    sa.points += 3
                elif goals_b > goals_a:
                    sb.points += 3
                else:
                    sa.points += 1; sb.points += 1
                match_rows.append({'stage': 'group', 'group': group, 'team_a': team_a, 'team_b': team_b, 'goals_a': goals_a, 'goals_b': goals_b, 'winner': winner})

        table = pd.DataFrame([{
            'team': s.team, 'group': s.group, 'points': s.points, 'gf': s.gf, 'ga': s.ga, 'gd': s.gd,
            'elo': float(self.elo_lookup.get(s.team, 1500.0))
        } for s in standings.values()])
        table = table.sort_values(['group', 'points', 'gd', 'gf', 'elo'], ascending=[True, False, False, False, False]).reset_index(drop=True)
        table['group_rank'] = table.groupby('group').cumcount() + 1
        auto = table[table['group_rank'] <= 2].copy()
        third = table[table['group_rank'] == 3].copy().sort_values(['points', 'gd', 'gf', 'elo'], ascending=False).head(8)
        qualifiers = pd.concat([auto, third], ignore_index=True)
        return table, qualifiers, pd.DataFrame(match_rows)

    def _play_knockout_match(self, team_a: str, team_b: str) -> str:
        probs = self.predictor.predict_proba(team_a, team_b, tournament='FIFA World Cup', neutral=True)
        outcome = self.rng.choices(['team_a_loss', 'draw', 'team_a_win'], weights=[probs['team_a_loss'], probs['draw'], probs['team_a_win']], k=1)[0]
        if outcome == 'team_a_win':
            return team_a
        if outcome == 'team_a_loss':
            return team_b
        elo_a = float(self.elo_lookup.get(team_a, 1500.0))
        elo_b = float(self.elo_lookup.get(team_b, 1500.0))
        p_pen_a = 1.0 / (1.0 + 10 ** (-(elo_a - elo_b) / 400.0))
        return team_a if self.rng.random() < p_pen_a else team_b

    def _simulate_knockout(self, qualifiers: pd.DataFrame) -> str:
        seeds = qualifiers.copy()
        seeds['seed_bucket'] = seeds['group_rank'].map({1: 0, 2: 1, 3: 2})
        seeds = seeds.sort_values(['seed_bucket', 'points', 'gd', 'gf', 'elo'], ascending=[True, False, False, False, False]).reset_index(drop=True)
        teams = seeds['team'].tolist()
        round32_pairs = list(zip(teams[:16], list(reversed(teams[16:]))))
        winners = [self._play_knockout_match(a, b) for a, b in round32_pairs]
        while len(winners) > 1:
            winners = [self._play_knockout_match(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
        return winners[0]

    def simulate_many(self, n_sims: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
        champions = []
        semifinal_counts = {}
        for _ in range(n_sims):
            group_table, qualifiers, _ = self._simulate_group_stage()
            champion = self._simulate_knockout(qualifiers)
            champions.append(champion)
        champ_df = pd.Series(champions).value_counts(normalize=True).mul(100).reset_index()
        champ_df.columns = ['team', 'title_probability_pct']
        champ_df = champ_df.sort_values('title_probability_pct', ascending=False).reset_index(drop=True)
        return champ_df, self.groups.copy()
