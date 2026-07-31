from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier

from src.features.build_match_features import MODEL_FEATURE_COLUMNS

FEATURES_PATH = ROOT / 'data' / 'processed' / 'match_features.csv'
MODEL_PATH = ROOT / 'models' / 'xgb_match_model.joblib'
METRICS_CSV = ROOT / 'reports' / 'metrics.csv'
METRICS_MD = ROOT / 'reports' / 'model_summary.md'

if __name__ == '__main__':
    df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
    df = df[df['date'] >= '1990-01-01'].copy()
    df = df.sort_values('date').reset_index(drop=True)

    split_date = df['date'].quantile(0.85)
    train = df[df['date'] < split_date].copy()
    test = df[df['date'] >= split_date].copy()

    X_train = train[MODEL_FEATURE_COLUMNS]
    y_train = train['target']
    X_test = test[MODEL_FEATURE_COLUMNS]
    y_test = test['target']

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=450,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    metrics = {
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'split_date': str(split_date.date()),
        'accuracy': float(accuracy_score(y_test, preds)),
        'log_loss': float(log_loss(y_test, probs)),
    }
    metrics_df = pd.DataFrame([metrics])
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV, index=False)

    report = classification_report(y_test, preds, target_names=['loss', 'draw', 'win'])
    summary = f'''# Match Outcome Model Summary

## Dataset
- Training rows: **{len(train):,}**
- Test rows: **{len(test):,}**
- Chronological split date: **{split_date.date()}**
- Features: **{len(MODEL_FEATURE_COLUMNS)}**

## Metrics
- Accuracy: **{metrics['accuracy']:.4f}**
- Log loss: **{metrics['log_loss']:.4f}**

## Classification report
```
{report}
```

## Notes
- Target encoding is from **team A perspective**: 0=loss, 1=draw, 2=win.
- Training uses only pre-match information to avoid leakage.
- Model is intended for match-probability generation, which then feeds tournament simulation.
'''
    METRICS_MD.write_text(summary, encoding='utf-8')

    bundle = {
        'model': model,
        'feature_cols': MODEL_FEATURE_COLUMNS,
        'metrics': metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(summary)
    print(f'Saved model to: {MODEL_PATH}')
