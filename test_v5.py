"""Run `python -m pytest -q test_v5.py`. Checks code invariants, not real-world realism."""
import numpy as np
import pandas as pd
import pytest
import torch
from src.ppo_engine import ActorCritic, CryptoMultiAgentEnv, Config, load_policy, save_policy, train_ppo
from src.datasets import TX_COLUMNS, generate_v5, describe_quality
from src.modeling import NUMERIC_FEATURES, MODEL_FEATURES, make_model_frame, evaluate_runs


@pytest.fixture(scope='module')
def fixture_data():
    cfg = Config(normal_users=35, merchants=8, exchanges=3, scammers=7,
                 money_launderers=6, target_transactions=520, episode_steps=96,
                 duration_days=14, seed=11)
    torch.manual_seed(11)
    initial = ActorCritic()
    model, log = train_ppo(cfg, total_steps=1000, rollout_steps=200, epochs=2,
                           verbose=False, seed=11)
    runs = {f'run_{s}': generate_v5(model, cfg, seed=s, run_id=f'run_{s}')
            for s in (101,202,303)}
    return cfg, initial, model, log, runs


def test_ppo_is_actually_trained(fixture_data):
    cfg, initial, model, log, runs = fixture_data
    assert len(log) > 0 and log.steps.iloc[-1] == 1000
    assert any(not torch.allclose(a,b) for a,b in zip(initial.parameters(), model.parameters()))
    assert model(torch.tensor(CryptoMultiAgentEnv(cfg).observe()).unsqueeze(0))[1].shape == (1,)


def test_26_columns_and_field_types(fixture_data):
    _, _, _, _, runs = fixture_data
    for tx, w, s, m in runs.values():
        assert list(tx.columns) == TX_COLUMNS and len(tx.columns) == 26
        assert len(tx) == 520 and tx.fraud_label.isin([0,1]).all()
        assert tx.timestamp.is_monotonic_increasing
        assert (tx.amount_btc > 0).all()
        assert 'policy_intent_choice' not in tx.columns
        assert 'initiating_agent_type' not in tx.columns


def test_ground_truth_audit_and_no_hidden_funding(fixture_data):
    _, _, _, _, runs = fixture_data
    for tx, wallets, scenarios, meta in runs.values():
        assert meta['ledger_audit']['passed']
        assert len(scenarios) > 0
        counts = tx[tx.fraud_label==1].groupby('scenario_id').size().to_dict()
        assert counts == scenarios.set_index('scenario_id').transaction_count.to_dict()
        assert abs(wallets.initial_balance_btc.sum() - wallets.current_balance_btc.sum()
                   - tx.transaction_fee_btc.sum()) < 2e-6


def test_no_predictor_leakage(fixture_data):
    _, _, _, _, runs = fixture_data
    tx = next(iter(runs.values()))[0]
    X, y = make_model_frame(tx)
    assert list(X.columns) == MODEL_FEATURES
    assert 'fraud_label' not in MODEL_FEATURES
    assert 'scenario_type' not in MODEL_FEATURES
    assert not any(name in MODEL_FEATURES for name in ['scenario_id','transaction_id',
             'sender_wallet','receiver_wallet','run_id','timestamp','initiating_agent_type'])
    assert y.equals(tx.fraud_label.astype(int))


def test_past_only_rolling_counts(fixture_data):
    _, _, _, _, runs = fixture_data
    tx = next(iter(runs.values()))[0]
    for _, group in tx.groupby('sender_wallet'):
        ordered = group.sort_values('timestamp')
        for i, (_, event) in enumerate(ordered.iterrows()):
            past = ordered.iloc[:i]
            t = event.timestamp
            expected_1h = sum((t - past.timestamp).dt.total_seconds() <= 3600)
            expected_24h = sum((t - past.timestamp).dt.total_seconds() <= 86400)
            assert event.sender_tx_last_hour == expected_1h
            assert event.sender_tx_last_24h == expected_24h


def test_multiple_runs_unique_and_seed_reproducible(fixture_data):
    cfg, _, model, _, runs = fixture_data
    first = runs['run_101'][0]
    repeat = generate_v5(model, cfg, seed=101, run_id='run_101')[0]
    pd.testing.assert_frame_equal(first, repeat)
    assert not first[['sender_wallet','receiver_wallet','amount_btc']].equals(
        runs['run_202'][0][['sender_wallet','receiver_wallet','amount_btc']])


def test_mixed_behaviour_and_amount_overlap(fixture_data):
    _, _, _, _, runs = fixture_data
    tx, wallets, _, _ = runs['run_101']
    # V5 never computes fraud labels from a transaction-size cutoff.
    assert tx.fraud_label.nunique() == 2
    assert (tx.groupby('sender_wallet').fraud_label.nunique() == 2).any()
    assert describe_quality(tx, wallets)['class_amount_ranges_overlap']


def test_validation_test_separated(fixture_data):
    _, _, _, _, runs = fixture_data
    training = {k:v[0] for k,v in runs.items()}
    _, result, imp = evaluate_runs(training)
    assert result['training_run'] == 'run_101'
    assert result['validation_run'] == 'run_202'
    assert result['testing_run'] == 'run_303'
    assert 0 <= result['selected_threshold_on_validation'] <= 1
    assert len(imp) == len(MODEL_FEATURES)


def test_save_and_load_policy(fixture_data, tmp_path):
    cfg, _, model, _, _ = fixture_data
    path = tmp_path/'model.pt'
    save_policy(model,path,config=cfg)
    loaded = load_policy(path)
    x = torch.tensor(CryptoMultiAgentEnv(cfg).observe()).unsqueeze(0)
    assert torch.allclose(model(x)[1], loaded(x)[1])


def test_all_five_named_behaviours_and_ground_truth(fixture_data):
    _, _, model, _, _ = fixture_data
    # Larger run makes rare scenario observations more likely; not guaranteed
    # under EVERY seed, and tests should not claim guaranteed prevalence.
    cfg = Config(normal_users=50, merchants=10, exchanges=3, scammers=10,
                 money_launderers=8, target_transactions=1600)
    tx, wallets, scenarios, meta = generate_v5(model, cfg, seed=101, run_id='run_101')
    expected = {'legitimate_to_legitimate', 'ordinary_to_scammer_wallet',
                'scam_victim_payment', 'compromised_wallet_transfer',
                'malicious_wallet_ordinary'}
    assert expected.issubset(set(tx.scenario_type))
    assert len(tx.columns) == 26
    assert (tx.loc[tx.scenario_type.isin(['scam_victim_payment','compromised_wallet_transfer']), 'fraud_label'] == 1).all()
    assert (tx.loc[tx.scenario_type.isin(['legitimate_to_legitimate', 'ordinary_to_scammer_wallet', 'malicious_wallet_ordinary']), 'fraud_label'] == 0).all()
    assert (tx.loc[tx.fraud_label.eq(0), 'scenario_id'].fillna('') == '').all()
    role = wallets.set_index('wallet_id').agent_type
    payment = tx[tx.scenario_type.eq('scam_victim_payment')]
    assert payment.sender_wallet.map(role).eq('Normal User').all()
    assert payment.receiver_wallet.map(role).eq('Scammer').all()
    assert scenarios.loc[scenarios.scenario_type.eq('scam_victim_payment'), 'affected_wallet'].ne('').all()
    assert meta['ledger_audit']['passed']
