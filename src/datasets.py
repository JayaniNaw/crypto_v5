"""26-column human-readable transaction dataset; full PPO provenance stays separate.

NEVER construct supervised ML X with this entire dataframe. Use the explicit
allowlist in src/modeling.py. All features reflect pre-transaction state.
"""
from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.ppo_engine import audit_dataset, generate_dataset

TX_COLUMNS = [
    "run_id", "transaction_id", "timestamp", "sender_wallet", "receiver_wallet",
    "amount_btc", "transaction_fee_btc", "sender_balance_before_btc",
    "receiver_balance_before_btc", "sender_wallet_age_days", "receiver_wallet_age_days",
    "sender_tx_count_before", "receiver_tx_count_before",
    "sender_unique_counterparties_before", "receiver_unique_counterparties_before",
    "sender_out_degree_before", "receiver_in_degree_before",
    "sender_total_sent_before_btc", "receiver_total_received_before_btc",
    "sender_repeat_counterparty", "seconds_since_sender_last_tx",
    "sender_tx_last_hour", "sender_tx_last_24h", "scenario_id", "scenario_type", "fraud_label",
]
assert len(TX_COLUMNS) == 26


def slim_transactions(original: pd.DataFrame) -> pd.DataFrame:
    """Convert audited V3 engine events into V5 public schema.

    For each row, recent activity features use PRIOR sender transactions ONLY.
    Therefore the current transaction and all future events are excluded.
    """
    source = original.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = source.rename(columns={
        "sender_balance_before": "sender_balance_before_btc",
        "receiver_balance_before": "receiver_balance_before_btc",
        "time_since_last_tx_seconds": "seconds_since_sender_last_tx",
    }).copy()
    if df["timestamp"].isna().any():
        raise ValueError("Timestamps cannot be missing")
    recent = defaultdict(deque)
    one_hour, one_day = [], []
    for row in df.itertuples(index=False):
        wallet, time = row.sender_wallet, pd.Timestamp(row.timestamp)
        times = recent[wallet]
        while times and (time - times[0]).total_seconds() > 86400:
            times.popleft()
        one_day.append(len(times))
        one_hour.append(sum((time - previous).total_seconds() <= 3600 for previous in times))
        times.append(time)  # Only after capturing previous activity
    df["sender_tx_last_hour"] = one_hour
    df["sender_tx_last_24h"] = one_day
    result = df[TX_COLUMNS].copy()
    if len(result.columns) != 26 or not result["fraud_label"].isin([0, 1]).all():
        raise AssertionError("Invalid V5 transaction schema or labels")
    return result


def generate_v5(model, cfg, *, seed: int, run_id: str):
    """Use learned PPO to generate, then audit FULL internal ledger before slimming."""
    full, wallets, scenarios, metadata = generate_dataset(model, cfg, seed=seed, run_id=run_id)
    audit = audit_dataset(full, wallets, scenarios, metadata)
    tx = slim_transactions(full)
    metadata = {
        **metadata,
        "dataset_version": "5",
        "decision_policy": "trained shared role-conditioned PPO",
        "transactions_columns": TX_COLUMNS,
        "ledger_audit": audit,
        "caveat": "Synthetic scenario exposure, rewards, recipient resolution and scheduling are designed environment mechanics; not verified against real blockchain fraud rates.",
    }
    # Policy/agent roles and post-transfer wallet balances are absent from the
    # public transaction table, but wallet role records remain for simulator audit.
    return tx, wallets, scenarios, metadata


def save_run(base_path: str | Path, outputs, *, run_id: str):
    dest = Path(base_path) / run_id
    dest.mkdir(parents=True, exist_ok=True)
    tx, wallets, scenarios, metadata = outputs
    tx.to_csv(dest / "transactions.csv", index=False)
    wallets.to_csv(dest / "wallets.csv", index=False)
    scenarios.to_csv(dest / "scenarios.csv", index=False)
    (dest / "experiment.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return dest


def describe_quality(tx: pd.DataFrame, wallets: pd.DataFrame):
    """Honest diagnostics, not claims of external realism."""
    roles = wallets.set_index("wallet_id")["agent_type"]
    sender_role = tx["sender_wallet"].map(roles)
    receiver_role = tx["receiver_wallet"].map(roles)
    genuine_to_scam = (
        tx["fraud_label"].eq(0)
        & sender_role.isin(["Normal User", "Merchant", "Exchange"])
        & receiver_role.eq("Scammer")
    )
    mixed_wallets = int(tx.groupby("sender_wallet")["fraud_label"].nunique().eq(2).sum())
    fraud = tx[tx["fraud_label"].eq(1)]["amount_btc"]
    genuine = tx[tx["fraud_label"].eq(0)]["amount_btc"]
    overlap = bool(len(fraud) and len(genuine) and fraud.min() <= genuine.max() and genuine.min() <= fraud.max())
    return {
        "wallets": int(wallets.wallet_id.nunique()),
        "total_transactions": int(len(tx)),
        "fraud_rate": float(tx.fraud_label.mean()),
        "mixed_sender_wallets": mixed_wallets,
        "genuine_payments_to_scammer_wallet": int(genuine_to_scam.sum()),
        "class_amount_ranges_overlap": overlap,
        "source_note": "Simulator-generated indicators; not real-world validation",
        "scenario_counts": tx.scenario_type.value_counts().to_dict(),
    }
