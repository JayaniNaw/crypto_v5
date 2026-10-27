"""Train genuine shared role-conditioned PPO locally, not in Streamlit.
Run: python train.py --steps 25000
"""
import argparse
import json
from pathlib import Path
from src.ppo_engine import Config, save_policy, train_ppo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=25000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout', type=int, default=512)
    parser.add_argument('--normal-users', type=int, default=120)
    parser.add_argument('--merchants', type=int, default=24)
    parser.add_argument('--exchanges', type=int, default=5)
    parser.add_argument('--scammers', type=int, default=15)
    parser.add_argument('--money-launderers', type=int, default=12)
    parser.add_argument('--output', default='models/ppo_policy.pt')
    args = parser.parse_args()
    cfg = Config(normal_users=args.normal_users, merchants=args.merchants,
                 exchanges=args.exchanges, scammers=args.scammers,
                 money_launderers=args.money_launderers, seed=args.seed)
    model, log = train_ppo(cfg, total_steps=args.steps, rollout_steps=args.rollout,
                           seed=args.seed, verbose=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_policy(model, destination, config=cfg)
    (destination.parent / 'policy_info.json').write_text(json.dumps({
        'training_steps': args.steps, 'seed': args.seed,
        'demonstration_checkpoint': args.steps < 25000,
        'note': 'Assess PPO training convergence and synthetic data realism before research claims.'
    }, indent=2), encoding='utf-8')
    log.to_csv(destination.parent / 'ppo_training_log.csv', index=False)
    print(f'Learned PPO checkpoint: {destination.resolve()}')
    print(f'Training history: {(destination.parent / "ppo_training_log.csv").resolve()}')

if __name__ == '__main__':
    main()
