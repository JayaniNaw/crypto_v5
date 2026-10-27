"""Evaluate fraud classifier on independent synthetic simulation runs."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.modeling import evaluate_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', default='outputs')
    p.add_argument('--runs', nargs='+', default=['run_101', 'run_202', 'run_303'])
    a = p.parse_args()
    folder = Path(a.folder)
    data = {}
    for name in a.runs:
        path = folder / name / 'transactions.csv'
        if not path.is_file():
            raise SystemExit(f'Missing {path}; run python generate.py first')
        data[name] = pd.read_csv(path)
    _, metrics, importance = evaluate_runs(data)
    print(json.dumps(metrics, indent=2))
    print('\nTop 10 allowed features:\n', importance.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
