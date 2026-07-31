# ⚽ FIFA World Cup 2026 Predictor

A data-driven football prediction project for the upcoming FIFA World Cup 2026, combining historical match data, team form signals, and an XGBoost-based model to estimate match outcomes and tournament title chances.

## 🌍 Project Overview
This project is designed to help fans, analysts, and football enthusiasts explore how a modern machine learning pipeline can be used to:

- predict match outcomes for international football fixtures
- estimate win / draw / loss probabilities
- simulate a World Cup 2026 tournament and generate provisional title odds
- visualize predictions through a lightweight Streamlit dashboard

## 🏆 What the project includes
- historical football results ingestion
- leak-free feature engineering for pre-match prediction
- dynamic Elo ratings and rolling form indicators
- training of a multiclass XGBoost model
- match probability predictions for any pair of teams
- a tournament simulation for provisional 2026 title odds
- an interactive dashboard for quick exploration

## 📊 Data Source
The project uses historical international football results from the public dataset `martj42/international_results`, mirrored and prepared for model training.

## 🧠 Model Approach
The model predicts outcomes from the perspective of Team A:

- `0` = Team A loss
- `1` = draw
- `2` = Team A win

Core features include:
- dynamic Elo ratings for both teams
- Elo difference
- rolling points over the last 5 matches
- rolling goals scored / conceded
- goal difference
- matches played
- neutral venue flag
- home advantage signal
- tournament importance
- expected score estimates from Elo
- rest days

## 🏗️ Project Structure
- `scripts/download_data.py` downloads raw match data
- `scripts/build_features.py` creates training-ready pre-match features
- `scripts/train_match_model.py` trains the predictive model
- `scripts/predict_worldcup_2026.py` runs the tournament simulation
- `src/features/build_match_features.py` handles feature engineering logic
- `src/models/predict_match.py` provides match prediction utilities
- `src/simulation/worldcup_simulator.py` simulates the World Cup bracket
- `dashboard_app.py` launches the Streamlit web app

## 🚀 Getting Started
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python scripts/run_pipeline.py
```

### 3. Launch the dashboard
```bash
streamlit run dashboard_app.py
```

## ⚙️ Individual Scripts
```bash
python scripts/download_data.py
python scripts/build_features.py
python scripts/train_match_model.py
python scripts/predict_worldcup_2026.py
```

## 🔎 Note on 2026 Title Odds
The tournament simulation is currently a strong prototype rather than an official FIFA forecast. It uses:

- the official 48-team group structure
- placeholder handling for unresolved playoff matchups
- a general seeded knockout structure rather than a fully exact tournament bracket

The match prediction model itself is the core reliable component, while the title probabilities provide an exciting preview of how the tournament could unfold.

## 🔮 Future Improvements
Potential next steps include:
- replacing the provisional bracket logic with the exact FIFA World Cup 2026 draw structure
- adding official FIFA rankings or external Elo snapshots as features
- tuning model hyperparameters and improving probability calibration
- exploring scoreline prediction models such as Poisson or bivariate Poisson
- deploying the app for public access
