"""Generate independent simulation runs from a SAVED learned PPO policy."""
import argparse
import json
from pathlib import Path
from src.ppo_engine import Config, load_policy
from src.datasets import generate_v5, save_run, describe_quality


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/ppo_policy.pt')
    p.add_argument('--output', default='outputs')
    p.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303])
    p.add_argument('--transactions', type=int, default=3000)
    p.add_argument('--duration', type=int, default=30)
    p.add_argument('--normal-users', type=int, default=120)
    p.add_argument('--merchants', type=int, default=24)
    p.add_argument('--exchanges', type=int, default=5)
    p.add_argument('--scammers', type=int, default=15)
    p.add_argument('--money-launderers', type=int, default=12)
    a = p.parse_args()
    model_path = Path(a.model)
    if not model_path.is_file():
        raise SystemExit(f'No trained PPO policy at {model_path}. Run python train.py first.')
    model = load_policy(model_path)
    cfg = Config(target_transactions=a.transactions, duration_days=a.duration,
                 normal_users=a.normal_users, merchants=a.merchants,
                 exchanges=a.exchanges, scammers=a.scammers,
                 money_launderers=a.money_launderers)
    for seed in a.seeds:
        run = f'run_{seed}'
        data = generate_v5(model, cfg, seed=seed, run_id=run)
        path = save_run(a.output, data, run_id=run)
        print(f'{path}: {json.dumps(describe_quality(data[0], data[1]), indent=2)}')

if __name__ == '__main__':
    main()
