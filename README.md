# FIFA World Cup 2026 Predictor

A complete starter project for predicting **international football match outcomes** and generating **provisional 2026 World Cup title probabilities**.

## What this project does
- downloads the historical international football results dataset
- builds leak-free pre-match features
- computes rolling team form and dynamic Elo ratings
- trains an XGBoost multiclass match model
- predicts win / draw / loss probabilities for any match
- simulates the 2026 World Cup many times to estimate title odds
- includes a small Streamlit app for quick use

## Data source
This project uses the `martj42/international_results` dataset, mirrored from GitHub and aligned with the Kaggle dataset you shared.

## Architecture
- `scripts/download_data.py` downloads raw CSV files
- `scripts/build_features.py` creates training rows from pre-match information only
- `scripts/train_match_model.py` trains the XGBoost model
- `scripts/predict_worldcup_2026.py` runs a provisional tournament simulator
- `src/features/build_match_features.py` handles Elo + rolling features
- `src/models/predict_match.py` predicts a single match
- `src/simulation/worldcup_simulator.py` simulates the World Cup
- `dashboard_app.py` is a lightweight Streamlit interface

## Important note on the 2026 title odds
The tournament simulator in this starter version is **provisional**:
- it uses the official 48-team group format
- it includes unresolved playoff placeholders by averaging candidate team strength
- it uses a generic seeded knockout pairing, not the exact official bracket mapping

So the **match model is real**, but the **title odds are a strong prototype rather than a final official prediction engine**.

## One-command pipeline
```bash
python3 scripts/run_pipeline.py
```

## Individual commands
```bash
python3 scripts/download_data.py
python3 scripts/build_features.py
python3 scripts/train_match_model.py
python3 scripts/predict_worldcup_2026.py
streamlit run dashboard_app.py
```

## Model target
The model predicts results from **team A perspective**:
- `0` = team A loss
- `1` = draw
- `2` = team A win

## Core features
- dynamic Elo for both teams
- Elo difference
- rolling points over last 5 matches
- rolling goals for / against / goal difference
- matches played
- neutral venue
- home-advantage flag
- tournament importance
- estimated expected score from Elo
- rest days

## Recommended next improvements
- replace provisional knockout seeding with the exact FIFA 2026 bracket mapping
- add official FIFA rankings or external Elo snapshots as extra features
- add hyperparameter tuning and probability calibration
- model scorelines directly with Poisson / bivariate Poisson for better tie-break handling
- add a production database and public deployment
