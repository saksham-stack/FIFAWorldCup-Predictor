from pathlib import Path
import pandas as pd

RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    'results.csv': 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
    'shootouts.csv': 'https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv',
    'former_names.csv': 'https://raw.githubusercontent.com/martj42/international_results/master/former_names.csv',
}

if __name__ == '__main__':
    for filename, url in URLS.items():
        path = RAW_DIR / filename
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
        print(f'Saved {filename}: {len(df):,} rows')
