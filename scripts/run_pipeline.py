from __future__ import annotations

import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
steps = [
    [sys.executable, 'scripts/download_data.py'],
    [sys.executable, 'scripts/build_features.py'],
    [sys.executable, 'scripts/train_match_model.py'],
    [sys.executable, 'scripts/predict_worldcup_2026.py'],
]

if __name__ == '__main__':
    for step in steps:
        print(f'\n>>> Running: {" ".join(step)}')
        subprocess.run(step, cwd=ROOT, check=True)
    print('\nPipeline completed successfully.')
