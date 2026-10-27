"""Free, learning-driven multi-agent cryptocurrency transaction simulator.

PPO is implemented in PyTorch (no paid API, no Stable-Baselines dependency).
Each acting wallet uses one shared role-conditioned policy. All four transaction
choices come from the trained network, not fixed behaviour-selection weights.
Environment mechanics, scenario opportunities, and rewards are explicitly designed;
this does NOT claim that the resulting synthetic data has been validated as realistic.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import copy
import json
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

# Keep Colab CPU workloads reproducible and lightweight.
ROLES = ("Normal User", "Merchant", "Exchange", "Scammer", "Money Launderer")
ROLE_PREFIX = ("U", "M", "E", "S", "L")
NUM_ACTIONS = (7, 7, 5, 2)  # recipient group, amount fraction, delay, intent
AMOUNT_FRACTIONS = (0.003, 0.012, 0.035, 0.08, 0.16, 0.32, 0.62)
DELAY_FACTORS = (0.4, 0.75, 1.0, 1.65, 2.5)
FEE_SATS = (200, 500, 1000, 1800)
SAT = 100_000_000
START_DATE = "2026-01-01T00:00:00"
OBS_SIZE = 5 + 11


@dataclass
class Config:
    normal_users: int = 90
    merchants: int = 18
    exchanges: int = 5
    scammers: int = 12
    money_launderers: int = 10
    episode_steps: int = 256
    target_transactions: int = 3000
    duration_days: int = 30
    seed: int = 42
    start_date: str = START_DATE
    # A simulator *scenario opportunity* is distinct from the learned action.
    # These probabilities constrain incident exposure, not behavioural decisions.
    opportunity_malicious: float = 0.40
    opportunity_normal: float = 0.022
    opportunity_merchant: float = 0.015
    opportunity_exchange: float = 0.005
    scam_offer_probability: float = 0.10  # simulated unsolicited payment request to a normal wallet

    def validate(self):
        counts = self.role_counts()
        if min(counts) < 1 or sum(counts) < 10:
            raise ValueError("At least one of each role and 10 total wallets are required")
        if self.episode_steps < 10 or self.target_transactions < 10:
            raise ValueError("episode_steps and target_transactions must be >= 10")
        if self.duration_days <= 0:
            raise ValueError("duration_days must be positive")
        for p in (self.opportunity_malicious, self.opportunity_normal,
                  self.opportunity_merchant, self.opportunity_exchange, self.scam_offer_probability):
            if not 0 <= p <= 1:
                raise ValueError("opportunity probabilities must be in [0,1]")
        datetime.fromisoformat(self.start_date)
        return self

    def role_counts(self):
        return [self.normal_users, self.merchants, self.exchanges,
                self.scammers, self.money_launderers]


class Wallet:
    def __init__(self, wid, role_id, initial_sats, age_days, org_id):
        self.id = wid
        self.role_id = role_id
        self.initial_sats = int(initial_sats)
        self.balance = int(initial_sats)
        self.age_days = int(age_days)
        self.org_id = org_id
        self.tx_count = 0
        self.sent = 0
        self.received = 0
        self.unique_peers = set()
        self.out_peers = set()
        self.in_peers = set()
        self.last_time = None
        self.sent_btc = 0.0
        self.received_btc = 0.0
        self.recent_times = []
        self.active_scenario = None
        self.scenario_budget = 0
        self.compromised_remaining = 0
        self.pending_scam_wallet = None  # simulated unsolicited request, unknown to the user


class CryptoMultiAgentEnv:
    """Sequentially schedules wallet actors; the actor uses a *shared learned* policy.

    This is a role-conditioned shared-policy multi-agent training architecture,
    not full centralized-critic MAPPO. Wallet scheduling and random recipient
    selection *within an action-selected category* are environment mechanics.
    """
    def __init__(self, cfg: Config, *, generation=False):
        self.cfg = copy.deepcopy(cfg).validate()
        self.generation = generation
        self.reset(cfg.seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else int(seed))
        self.wallets = []
        self.wallet_by_id = {}
        self.txs = []
        self.scenarios = {}
        self.fees_collected = 0
        self.clock = 0.0
        self.steps = 0
        self.scenario_counter = 0
        self.actor_index = 0
        self.fraud_opp = False
        self.active_offer_wallet = None
        for role, (prefix, number) in enumerate(zip(ROLE_PREFIX, self.cfg.role_counts())):
            for i in range(number):
                # Random initial balances are explicit initial endowments (not hidden top-ups).
                # All balances thereafter move through valid ledger transactions.
                raw = self.rng.lognormal(mean=0.15, sigma=0.9)
                scale = (1.7, 6.0, 20.0, 2.0, 3.0)[role]
                btc = float(np.clip(raw * scale, 0.12, 75.0))
                sats = int(round(btc * SAT))
                org = int(self.rng.integers(0, 3)) if role in (3, 4) else -1
                wallet = Wallet(f"{prefix}_{i+1:04d}", role, sats,
                                int(self.rng.integers(14, 1700)), org)
                self.wallets.append(wallet)
                self.wallet_by_id[wallet.id] = wallet
        self.initial_supply = sum(w.balance for w in self.wallets)
        self._select_actor()
        return self.observe()

    def _select_actor(self):
        # Wallet scheduling belongs to the environment; learned PPO still picks
        # recipient group, amount, delay and intent for each selected wallet.
        weights = np.array([1.0 if w.role_id < 3 else 1.3 for w in self.wallets], dtype=float)
        weights /= weights.sum()
        self.actor_index = int(self.rng.choice(len(self.wallets), p=weights))
        actor = self.wallets[self.actor_index]
        self.active_offer_wallet = None
        if actor.role_id == 0:
            # A persistent compromise is an incident state; the PPO policy
            # chooses subsequent transfers rather than following a script.
            if actor.compromised_remaining == 0 and self.rng.random() < self.cfg.opportunity_normal:
                actor.compromised_remaining = int(self.rng.integers(2, 5))
            # Scam wallet sends an unsolicited invoice; user has no role/identity
            # access. The PPO must select a compatible recipient category to pay it.
            if actor.compromised_remaining == 0 and self.rng.random() < self.cfg.scam_offer_probability:
                scammers = [w for w in self.wallets if w.role_id == 3]
                if scammers:
                    self.active_offer_wallet = scammers[int(self.rng.integers(len(scammers)))]
            self.fraud_opp = bool(actor.compromised_remaining > 0)
        elif actor.active_scenario and actor.scenario_budget > 0:
            self.fraud_opp = True
        else:
            p = (self.cfg.opportunity_normal, self.cfg.opportunity_merchant,
                 self.cfg.opportunity_exchange, self.cfg.opportunity_malicious,
                 self.cfg.opportunity_malicious)[actor.role_id]
            self.fraud_opp = bool(self.rng.random() < p)

    @property
    def actor(self):
        return self.wallets[self.actor_index]

    def observe(self):
        w = self.actor
        role = np.zeros(len(ROLES), dtype=np.float32)
        role[w.role_id] = 1.0
        # State is observed *before* the current transaction.
        obs = np.array([
            np.log1p(w.balance / SAT) / 5,
            np.log1p(w.tx_count) / 10,
            np.log1p(w.sent) / 9,
            np.log1p(w.received) / 9,
            len(w.unique_peers) / max(1, len(self.wallets) - 1),
            np.log1p(w.sent_btc) / 12,
            np.log1p(w.received_btc) / 12,
            min(1.0, self.clock / max(1.0, self.cfg.episode_steps * 1000)),
            1.0 if self.fraud_opp else 0.0,
            1.0 if w.active_scenario and w.scenario_budget > 0 else 0.0,
            min(1, w.age_days / 1500),
        ], dtype=np.float32)
        result = np.concatenate([role, obs])
        assert len(result) == OBS_SIZE
        return result

    def _recipient(self, group, sender):
        others = [w for w in self.wallets if w.id != sender.id]
        if group == 0:  # previously contacted wallet
            candidates = [w for w in others if w.id in sender.unique_peers]
        elif group == 1:  # unfamiliar wallet
            candidates = [w for w in others if w.id not in sender.unique_peers]
        elif group == 2:
            candidates = [w for w in others if w.role_id == 1]
        elif group == 3:
            candidates = [w for w in others if w.role_id == 2]
        elif group == 4:  # agent's organization, if present
            candidates = [w for w in others if w.org_id >= 0 and w.org_id == sender.org_id]
        elif group == 5:  # highly connected (using past information only)
            top = sorted(others, key=lambda w: len(w.unique_peers), reverse=True)
            candidates = top[:max(1, len(top) // 10)]
        else:
            candidates = others
        if not candidates:
            candidates = others
        # A prompted user can choose the unfamiliar-wallet or general-wallet
        # recipient category. If that category includes their pending invoice,
        # the environment resolves it to the named payee. The policy determines
        # WHETHER to use this category; the environment resolves wallet IDs.
        if (self.active_offer_wallet is not None and sender.role_id == 0
                and group in (1, 6) and self.active_offer_wallet in candidates):
            return self.active_offer_wallet
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def step(self, action):
        if len(action) != 4:
            raise ValueError("Each PPO action needs four discrete components")
        recipient_group, amount_choice, delay_choice, intent_choice = map(int, action)
        if not (0 <= recipient_group < 7 and 0 <= amount_choice < 7
                and 0 <= delay_choice < 5 and 0 <= intent_choice < 2):
            raise ValueError("PPO action component out of range")
        actor = self.actor
        receiver = self._recipient(recipient_group, actor)
        # Time is determined by learned delay choice, plus stochastic network jitter.
        gap = max(1.0, 350.0 * DELAY_FACTORS[delay_choice] * float(self.rng.lognormal(0, 0.3)))
        self.clock += gap
        self.steps += 1
        base_fee = int(FEE_SATS[int(self.rng.integers(0, len(FEE_SATS)))])
        amount = int(actor.balance * AMOUNT_FRACTIONS[amount_choice])
        # Every proposed transaction must be finance-valid; insufficient funds
        # cause a failed attempted action (no ledger row), not an invisible top-up.
        successful = amount >= 1000 and actor.balance >= amount + base_fee
        requested_illicit = intent_choice == 1
        compromise_active = actor.role_id == 0 and actor.compromised_remaining > 0
        # A scam victim may act with ordinary intent. The fraudulent *event*
        # is determined from an actual pending deceptive payment request.
        victim_payment = bool(successful and actor.role_id == 0
                              and self.active_offer_wallet is not None
                              and receiver.id == self.active_offer_wallet.id
                              and not requested_illicit)
        illicit = bool(successful and ((requested_illicit and self.fraud_opp) or victim_payment))
        reward = -0.2
        if requested_illicit and not self.fraud_opp:
            reward -= 0.35  # trying to invoke an unavailable scenario is costly
        if successful:
            sender_balance_before = actor.balance
            receiver_balance_before = receiver.balance
            tx_count_sender_before = actor.tx_count
            tx_count_receiver_before = receiver.tx_count
            sender_conn_before = len(actor.unique_peers)
            receiver_conn_before = len(receiver.unique_peers)
            sender_out_before = len(actor.out_peers)
            sender_in_before = len(actor.in_peers)
            receiver_out_before = len(receiver.out_peers)
            receiver_in_before = len(receiver.in_peers)
            sender_sent_before = actor.sent_btc
            receiver_received_before = receiver.received_btc
            sender_repeat = int(receiver.id in actor.unique_peers)
            # Five named mixed-behaviour scenarios plus explicit "other" cases.
            # scenario_type is research ground truth, NEVER a model predictor.
            if victim_payment:
                scenario_type = "scam_victim_payment"
            elif illicit and compromise_active:
                scenario_type = "compromised_wallet_transfer"
            elif not illicit and actor.role_id in (3, 4):
                scenario_type = "malicious_wallet_ordinary"
            elif not illicit and actor.role_id < 3 and receiver.role_id == 3:
                scenario_type = "ordinary_to_scammer_wallet"
            elif not illicit and actor.role_id < 3 and receiver.role_id < 3:
                scenario_type = "legitimate_to_legitimate"
            elif illicit:
                scenario_type = "other_fraud_activity"  # scam/laundering actions
            else:
                scenario_type = "other_ordinary"
            scenario_id = ""
            if illicit:
                # Persistent compromise and multi-wallet laundering may extend
                # an existing incident; victim payments start a new incident.
                reuse = (not victim_payment and actor.active_scenario
                         and actor.scenario_budget > 0
                         and self.scenarios.get(actor.active_scenario, {}).get("scenario_type") == scenario_type)
                if not reuse:
                    self.scenario_counter += 1
                    actor.active_scenario = f"SCN{self.scenario_counter:07d}"
                    actor.scenario_budget = int(self.rng.integers(2, 7))
                    self.scenarios[actor.active_scenario] = {
                        "scenario_id": actor.active_scenario,
                        "scenario_type": scenario_type,
                        "originator_wallet": actor.id,
                        "affected_wallet": actor.id if compromise_active or victim_payment else "",
                        "recipient_wallet": receiver.id if victim_payment else "",
                        "fraud_label": 1, "start_event": len(self.txs) + 1,
                        "transaction_count": 0,
                    }
                scenario_id = actor.active_scenario
                actor.scenario_budget -= 1
                self.scenarios[scenario_id]["transaction_count"] += 1
                if (not victim_payment and receiver.org_id >= 0
                        and receiver.org_id == actor.org_id and receiver.role_id in (3, 4)):
                    receiver.active_scenario = scenario_id
                    receiver.scenario_budget = max(receiver.scenario_budget, 2)
                if actor.scenario_budget <= 0 or victim_payment:
                    actor.active_scenario = None
            # A compromised wallet can still make a genuine transaction and
            # may later recover. Incident state survives several actor turns.
            if compromise_active:
                actor.compromised_remaining = max(0, actor.compromised_remaining - 1)
            actor.balance -= amount + base_fee
            receiver.balance += amount
            self.fees_collected += base_fee
            actor.sent += 1
            receiver.received += 1
            actor.tx_count += 1
            receiver.tx_count += 1
            actor.unique_peers.add(receiver.id)
            receiver.unique_peers.add(actor.id)
            actor.out_peers.add(receiver.id)
            receiver.in_peers.add(actor.id)
            actor.sent_btc += amount / SAT
            receiver.received_btc += amount / SAT
            actor.last_time = self.clock
            receiver.last_time = self.clock
            self.txs.append({
                "transaction_id": f"TX{len(self.txs)+1:08d}",
                "event_seconds": self.clock,
                "sender_wallet": actor.id,
                "receiver_wallet": receiver.id,
                "amount_btc": amount / SAT,
                "transaction_fee_btc": base_fee / SAT,
                "sender_balance_before": sender_balance_before / SAT,
                "sender_balance_after": actor.balance / SAT,
                "receiver_balance_before": receiver_balance_before / SAT,
                "receiver_balance_after": receiver.balance / SAT,
                "sender_wallet_age_days": actor.age_days,
                "receiver_wallet_age_days": receiver.age_days,
                "sender_tx_count_before": tx_count_sender_before,
                "receiver_tx_count_before": tx_count_receiver_before,
                "sender_unique_counterparties_before": sender_conn_before,
                "receiver_unique_counterparties_before": receiver_conn_before,
                "sender_out_degree_before": sender_out_before,
                "sender_in_degree_before": sender_in_before,
                "receiver_out_degree_before": receiver_out_before,
                "receiver_in_degree_before": receiver_in_before,
                "sender_total_sent_before_btc": sender_sent_before,
                "receiver_total_received_before_btc": receiver_received_before,
                "sender_repeat_counterparty": sender_repeat,
                # Internal simulator metadata: NEVER use as fraud-model features.
                "initiating_agent_type": ROLES[actor.role_id],
                "sender_agent_type": ROLES[actor.role_id],
                "receiver_agent_type": ROLES[receiver.role_id],
                "policy_recipient_group": recipient_group,
                "policy_amount_choice": amount_choice,
                "policy_delay_choice": delay_choice,
                "policy_intent_choice": intent_choice,
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "fraud_label": int(illicit),
                "label": "Fraud" if illicit else "Genuine",
            })
            # Incentivizes successful transactions and task fulfilment; does not
            # script what a wallet must transfer, nor make amounts label-specific.
            reward = (1.05 if illicit else 0.65)
            if recipient_group == 0 and tx_count_sender_before > 1:
                reward += 0.05
            if recipient_group in (2, 3) and actor.role_id < 3:
                reward += 0.08
            if illicit and recipient_group in (1, 4, 5):
                reward += 0.10
            if illicit and receiver.org_id >= 0 and receiver.org_id == actor.org_id:
                reward += 0.15  # encourage learned multi-wallet scenario continuation
        if not successful:
            reward -= 0.20
        done = self.steps >= self.cfg.episode_steps if not self.generation else False
        # Training is one shared policy controlling the acting wallet at each
        # sequential environment turn (not independent MAPPO agents).
        if not done:
            self._select_actor()
        return (np.zeros(OBS_SIZE, dtype=np.float32) if done else self.observe(),
                float(reward), done, {"successful": successful,
                                      "fraud": illicit})

    def export(self, *, run_id="run_0", normalize_duration=True):
        if not self.txs:
            raise RuntimeError("No successful transactions were generated")
        df = pd.DataFrame(self.txs).sort_values("event_seconds", kind="stable").reset_index(drop=True)
        start = datetime.fromisoformat(self.cfg.start_date)
        raw = df["event_seconds"].to_numpy(dtype=float)
        if normalize_duration and len(df) > 1:
            target_seconds = float(self.cfg.duration_days * 86400)
            normalized = (raw - raw[0]) / (raw[-1] - raw[0]) * target_seconds
        else:
            normalized = raw - raw[0]
        dt = pd.to_datetime(start) + pd.to_timedelta(normalized, unit="s")
        df["sender_wallet_age_days"] = df["sender_wallet_age_days"] + normalized / 86400
        df["receiver_wallet_age_days"] = df["receiver_wallet_age_days"] + normalized / 86400
        df.insert(1, "timestamp", dt)
        last_seen = {}
        elapsed = []
        for wallet, second in zip(df["sender_wallet"], normalized):
            elapsed.append(float(second - last_seen[wallet]) if wallet in last_seen else -1.0)
            last_seen[wallet] = second
        df["time_since_last_tx_seconds"] = elapsed
        df.insert(0, "run_id", run_id)
        df = df.drop(columns=["event_seconds"])
        wallets = pd.DataFrame([{
            "run_id": run_id, "wallet_id": w.id, "agent_id": w.id,
            "agent_type": ROLES[w.role_id], "wallet_age_days": w.age_days + float(normalized[-1] / 86400),
            "initial_wallet_age_days": w.age_days,
            "initial_balance_btc": w.initial_sats / SAT,
            "current_balance_btc": w.balance / SAT,
            "transaction_count": w.tx_count,
            "unique_counterparties": len(w.unique_peers),
            "out_degree": len(w.out_peers), "in_degree": len(w.in_peers),
        } for w in self.wallets])
        scenario_df = pd.DataFrame(list(self.scenarios.values()), columns=[
            "scenario_id", "scenario_type", "originator_wallet",
            "affected_wallet", "recipient_wallet", "fraud_label",
            "start_event", "transaction_count"])
        scenario_df.insert(0, "run_id", run_id)
        scenario_df["start_time"] = pd.NaT
        if len(scenario_df):
            starts = df.loc[df["scenario_id"].ne(""), ["scenario_id", "timestamp"]]
            times = starts.groupby("scenario_id")["timestamp"].min()
            scenario_df["start_time"] = scenario_df["scenario_id"].map(times)
        metadata = {
            "run_id": run_id, "seed": self.cfg.seed,
            "requested_transactions": self.cfg.target_transactions,
            "actual_transactions": len(df), "actual_fraud": int(df.fraud_label.sum()),
            "fraud_rate": float(df.fraud_label.mean()),
            "actual_duration_days": float(normalized[-1] / 86400) if len(df) > 1 else 0,
            "model_architecture": "role_conditioned_shared_PPO_sequential_agents",
            "initial_supply_btc": self.initial_supply / SAT,
            "final_supply_btc": sum(w.balance for w in self.wallets) / SAT,
            "total_fees_btc": self.fees_collected / SAT,
            "scenario_count": len(scenario_df),
            "limitations": ["Not a Bitcoin UTXO simulation", "not calibrated to real chain data",
                            "fraud opportunities, reward functions, and scam offers are designed environment mechanics",
                            "a scam offer is simulated, not evidence of real deception",
                            "simulation duration normalized for export"],
        }
        return df, wallets, scenario_df, metadata


class ActorCritic(nn.Module):
    def __init__(self, obs_size=OBS_SIZE, hidden=96):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, n) for n in NUM_ACTIONS])
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return [head(h) for head in self.actor_heads], self.value_head(h).squeeze(-1)

    def act(self, obs, *, deterministic=False):
        logits, value = self(obs)
        dists = [Categorical(logits=x) for x in logits]
        actions = [d.probs.argmax(dim=-1) if deterministic else d.sample() for d in dists]
        action_tensor = torch.stack(actions, dim=-1)
        logprob = torch.stack([d.log_prob(a) for d, a in zip(dists, actions)], dim=-1).sum(dim=-1)
        return action_tensor, logprob, value

    def evaluate_actions(self, obs, actions):
        logits, values = self(obs)
        dists = [Categorical(logits=x) for x in logits]
        logprob = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return logprob, entropy, values


def train_ppo(cfg: Config, *, total_steps=20000, rollout_steps=512,
              epochs=4, minibatch_size=128, lr=3e-4, seed=42,
              device="cpu", verbose=True, initial_policy=None,
              initial_optimizer_state=None, initial_steps=0,
              checkpoint_path=None, checkpoint_every_rollouts=5,
              return_optimizer=False):
    """Actual clipped PPO with GAE(lambda), entropy and value loss.

    One shared policy observes the next scheduled actor after each environment
    transition. Thus, this optimizes a collective multi-agent system objective.
    """
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    cfg.validate()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    net = (copy.deepcopy(initial_policy) if initial_policy is not None
           else ActorCritic()).to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if initial_optimizer_state is not None:
        optimizer.load_state_dict(initial_optimizer_state)
    env = CryptoMultiAgentEnv(cfg)
    obs = env.observe()
    steps = 0
    stats = []
    while steps < total_steps:
        n = min(rollout_steps, total_steps - steps)
        buffer = {key: [] for key in ("obs", "actions", "logprob", "rewards", "done", "values")}
        fraud_events = 0
        valid_events = 0
        rewards = 0.0
        for i in range(n):
            observation = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                actions, logp, value = net.act(observation)
            a = actions[0].cpu().numpy().tolist()
            next_obs, reward, done, info = env.step(a)
            buffer["obs"].append(obs.copy())
            buffer["actions"].append(a)
            buffer["logprob"].append(logp.item())
            buffer["values"].append(value.item())
            buffer["rewards"].append(reward)
            buffer["done"].append(float(done))
            fraud_events += int(info["fraud"])
            valid_events += int(info["successful"])
            rewards += reward
            obs = env.reset(int(env.rng.integers(0, 2**31))) if done else next_obs
        with torch.no_grad():
            _, last_value = net(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        last_value = float(last_value.item())
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        gamma = 0.99
        lam = 0.95
        for i in reversed(range(n)):
            next_val = last_value if i == n - 1 else buffer["values"][i + 1]
            continuing = 1.0 - buffer["done"][i]
            delta = buffer["rewards"][i] + gamma * next_val * continuing - buffer["values"][i]
            gae = delta + gamma * lam * continuing * gae
            advantages[i] = gae
        returns = advantages + np.asarray(buffer["values"], dtype=np.float32)
        obs_t = torch.as_tensor(np.asarray(buffer["obs"]), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(np.asarray(buffer["actions"]), dtype=torch.long, device=device)
        old_logp_t = torch.as_tensor(buffer["logprob"], dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        if n > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        for _ in range(epochs):
            for indices in torch.randperm(n, device=device).split(minibatch_size):
                new_logp, entropy, vpred = net.evaluate_actions(obs_t[indices], actions_t[indices])
                ratio = torch.exp(new_logp - old_logp_t[indices])
                pg_loss = -torch.minimum(ratio * adv_t[indices],
                                          torch.clamp(ratio, 0.8, 1.2) * adv_t[indices]).mean()
                v_loss = (vpred - returns_t[indices]).square().mean()
                loss = pg_loss + 0.5 * v_loss - 0.025 * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
        steps += n
        stat = {"steps": initial_steps + steps, "mean_reward": rewards / n,
                "valid_rate": valid_events / n, "fraud_action_rate": fraud_events / n}
        stats.append(stat)
        if checkpoint_path and (len(stats) % checkpoint_every_rollouts == 0 or steps == total_steps):
            save_training_checkpoint(net, optimizer.state_dict(), checkpoint_path,
                                     config=cfg, completed_steps=initial_steps + steps)
        if verbose and (len(stats) == 1 or steps % max(rollout_steps, 5000) < rollout_steps or steps == total_steps):
            print("PPO", stat)
    result = (net.cpu().eval(), pd.DataFrame(stats))
    return (*result, optimizer.state_dict()) if return_optimizer else result


def save_training_checkpoint(model: ActorCritic, optimizer_state, path, *,
                             config: Config, completed_steps: int):
    """Training checkpoint including optimizer momentum for Colab interruption."""
    torch.save({"state_dict": model.state_dict(), "optimizer_state": optimizer_state,
                "completed_steps": int(completed_steps), "config": vars(config),
                "obs_size": OBS_SIZE, "num_actions": NUM_ACTIONS}, path)


def load_training_checkpoint(path):
    """Only load checkpoints you created / trust."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["obs_size"] != OBS_SIZE or tuple(ckpt["num_actions"]) != NUM_ACTIONS:
        raise ValueError("Invalid checkpoint architecture")
    policy = ActorCritic()
    policy.load_state_dict(ckpt["state_dict"])
    return policy.eval(), ckpt["optimizer_state"], int(ckpt["completed_steps"])


def save_policy(model: ActorCritic, path, *, config=None):
    """Save trained policy; load only checkpoints you personally trust."""
    checkpoint = {"state_dict": model.state_dict(), "obs_size": OBS_SIZE,
                  "num_actions": NUM_ACTIONS,
                  "config": vars(config) if config is not None else {}}
    torch.save(checkpoint, path)


def load_policy(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint["obs_size"] != OBS_SIZE or tuple(checkpoint["num_actions"]) != NUM_ACTIONS:
        raise ValueError("Checkpoint architecture does not match this V3 simulator")
    model = ActorCritic()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def generate_dataset(model: ActorCritic, cfg: Config, *, seed=None, run_id="run_0",
                     max_attempts_factor=8, stochastic=True):
    config = copy.deepcopy(cfg)
    if seed is not None:
        config.seed = int(seed)
    torch.set_num_threads(min(2, torch.get_num_threads()))  # avoid thread oversubscription on tiny inference batches
    torch.manual_seed(config.seed + 991)
    env = CryptoMultiAgentEnv(config, generation=True)
    max_attempts = config.target_transactions * max_attempts_factor
    for _ in range(max_attempts):
        with torch.no_grad():
            obs_t = torch.as_tensor(env.observe(), dtype=torch.float32).unsqueeze(0)
            action, _, _ = model.act(obs_t, deterministic=not stochastic)
        env.step(action[0].tolist())
        if len(env.txs) >= config.target_transactions:
            break
    if len(env.txs) < config.target_transactions:
        raise RuntimeError(f"Only {len(env.txs)} valid transactions after {max_attempts} attempts; "
                           "adjust initial funding, training, or limits")
    return env.export(run_id=run_id)


def audit_dataset(txs: pd.DataFrame, wallets: pd.DataFrame,
                  scenarios: pd.DataFrame, meta: dict):
    """Assert finance, temporal, identity and scenario invariants."""
    problems = []
    if not txs["timestamp"].is_monotonic_increasing:
        problems.append("timestamps not sorted")
    if txs["transaction_id"].duplicated().any():
        problems.append("duplicate transaction ID")
    if (txs.sender_wallet == txs.receiver_wallet).any():
        problems.append("self-transfer")
    if (txs.amount_btc <= 0).any() or (txs.transaction_fee_btc <= 0).any():
        problems.append("nonpositive amount or fee")
    for col in ("sender_balance_before", "sender_balance_after", "receiver_balance_before",
                "receiver_balance_after"):
        if (txs[col] < -1e-9).any():
            problems.append("negative " + col)
    if not np.allclose(txs.sender_balance_before - txs.amount_btc - txs.transaction_fee_btc,
                       txs.sender_balance_after, atol=1e-8):
        problems.append("sender accounting inconsistency")
    if not np.allclose(txs.receiver_balance_before + txs.amount_btc,
                       txs.receiver_balance_after, atol=1e-8):
        problems.append("receiver accounting inconsistency")
    if abs(wallets.initial_balance_btc.sum() - wallets.current_balance_btc.sum()
           - txs.transaction_fee_btc.sum()) > 2e-6:
        problems.append("global monetary conservation failure")
    if not txs.loc[txs.fraud_label.eq(1), "scenario_id"].str.len().gt(0).all():
        problems.append("fraud transaction lacks scenario ID")
    if txs.loc[txs.fraud_label.eq(0), "scenario_id"].str.len().gt(0).any():
        problems.append("genuine transaction includes fraud scenario ID")
    if not scenarios.empty:
        counts = txs.loc[txs.fraud_label.eq(1)].groupby("scenario_id").size()
        expected = scenarios.set_index("scenario_id")["transaction_count"]
        if counts.to_dict() != expected.to_dict():
            problems.append("scenario counts mismatch")
    if len(txs) != meta["actual_transactions"] or int(txs.fraud_label.sum()) != meta["actual_fraud"]:
        problems.append("metadata counts mismatch")
    if problems:
        raise AssertionError(" | ".join(problems))
    return {"passed": True, "transactions": len(txs),
            "fraud": int(txs.fraud_label.sum()), "scenarios": len(scenarios),
            "supply_conserved": True, "chronological": True}


# Explicit allowlist: no simulator role, agent identity, policy action, scenario,
# post-event balance, fraud label, or future-derived whole-graph feature enters ML.
MODEL_FEATURES = [
    "amount_btc", "transaction_fee_btc", "sender_balance_before",
    "receiver_balance_before", "sender_wallet_age_days", "receiver_wallet_age_days",
    "sender_tx_count_before", "receiver_tx_count_before",
    "sender_unique_counterparties_before", "receiver_unique_counterparties_before",
    "sender_out_degree_before", "sender_in_degree_before",
    "receiver_out_degree_before", "receiver_in_degree_before",
    "sender_total_sent_before_btc", "receiver_total_received_before_btc",
    "sender_repeat_counterparty", "time_since_last_tx_seconds",
]


def evaluate_heldout_runs(datasets: list[pd.DataFrame]):
    """Train on earlier independent runs, test on last unseen run.

    This is a demonstration of generalisation *across synthetic runs*, not a
    claim of transfer to genuine blockchain data.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                                 roc_auc_score, confusion_matrix)
    if len(datasets) < 3:
        raise ValueError("Provide at least three independent runs")
    training = pd.concat(datasets[:-1], ignore_index=True)
    test = datasets[-1]
    if min(training.fraud_label.nunique(), test.fraud_label.nunique()) < 2:
        raise ValueError("Both genuine and fraud classes required in train and test runs")
    x_train = training[MODEL_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(-1)
    x_test = test[MODEL_FEATURES].replace([np.inf,-np.inf], np.nan).fillna(-1)
    model = RandomForestClassifier(n_estimators=130, min_samples_leaf=5,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    model.fit(x_train, training.fraud_label)
    pred_prob = model.predict_proba(x_test)[:, 1]
    pred = (pred_prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test.fraud_label, pred,
                                                                average="binary", zero_division=0)
    metrics = {"train_runs": len(datasets)-1, "test_run": str(test.run_id.iloc[0]),
               "train_count": len(training), "test_count": len(test),
               "test_prevalence": float(test.fraud_label.mean()),
               "precision": float(precision), "recall": float(recall), "f1": float(f1),
               "pr_auc": float(average_precision_score(test.fraud_label, pred_prob)),
               "roc_auc": float(roc_auc_score(test.fraud_label, pred_prob)),
               "confusion_matrix": confusion_matrix(test.fraud_label, pred).tolist(),
               "note": "Synthetic held-out-run evaluation; does not establish real-world fraud performance."}
    return model, metrics


def write_run(outdir, run_id, outputs):
    outdir = Path(outdir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    txs, wallets, scenarios, meta = outputs
    audit_dataset(txs, wallets, scenarios, meta)
    txs.to_csv(outdir / "transactions.csv", index=False)
    wallets.to_csv(outdir / "wallets.csv", index=False)
    scenarios.to_csv(outdir / "scenarios.csv", index=False)
    with (outdir / "experiment.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return outdir
