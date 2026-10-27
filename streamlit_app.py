"""Compact horizontal-tab Streamlit workflow powered by learned PPO actions.
Training belongs in `python train.py`; the web app only loads and runs that policy.
"""
from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ppo_engine import Config, load_policy
from src.datasets import TX_COLUMNS, generate_v5, describe_quality, save_run
from src.graphs import build_graph, network_figure, network_summary
from src.modeling import evaluate_runs, evaluate_temporal

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / 'models' / 'ppo_policy.pt'
OUTPUTS = ROOT / 'outputs'
st.set_page_config(page_title='Crypto Transaction Simulator', page_icon='🪙', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container{padding-top:1.25rem;max-width:1300px}
div[data-testid="stMetric"]{border:1px solid rgba(120,140,160,.16);border-radius:10px;padding:12px}
div[data-testid="stSidebar"]{display:none}
</style>''', unsafe_allow_html=True)

if 'runs' not in st.session_state:
    st.session_state.runs = {}
if 'chosen_run' not in st.session_state:
    st.session_state.chosen_run = None


@st.cache_resource(show_spinner='Loading trained PPO network…')
def cached_policy(path: str, mtime_ns: int):
    # Only load locally generated trusted model files, never user-uploaded checkpoints.
    return load_policy(path)


def available_policy():
    if not MODEL_FILE.exists():
        return None
    return cached_policy(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)


def chosen_outputs():
    selected = st.session_state.chosen_run
    return st.session_state.runs.get(selected) if selected else None


def zip_runs(runs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for run_name, (transactions, wallets, scenarios, meta) in runs.items():
            archive.writestr(f'{run_name}/transactions.csv', transactions.to_csv(index=False))
            archive.writestr(f'{run_name}/wallets.csv', wallets.to_csv(index=False))
            archive.writestr(f'{run_name}/scenarios.csv', scenarios.to_csv(index=False))
            archive.writestr(f'{run_name}/experiment.json', json.dumps(meta, indent=2, default=str))
        archive.writestr('README.txt',
            'PPO V5 synthetic data. All fraud labels are simulator ground truth. '
            'Do not train fraud classifiers on scenario_id, wallet IDs, agent_type, '
            'PPO action choices, or any post-transaction state. '
            'Use a separate held-out simulation run for final testing.\n')
    return buf.getvalue()


def show_selection():
    return chosen_outputs()


st.title('Crypto Transaction Simulator')
st.caption('Generate and explore synthetic cryptocurrency transactions with learned multi-agent PPO policies.')
if st.session_state.runs:
    names = list(st.session_state.runs)
    if st.session_state.chosen_run not in names:
        st.session_state.chosen_run = names[0]
    current = st.selectbox('View run', names, index=names.index(st.session_state.chosen_run),
                           key='active_simulation_run')
    st.session_state.chosen_run = current

tab_sim, tab_tx, tab_graph, tab_eda, tab_model = st.tabs(
    ['Simulation', 'Transactions', 'Network Graph', 'EDA', 'Modelling'])

with tab_sim:
    st.subheader('Simulation setup')
    if not MODEL_FILE.exists():
        st.warning('No trained PPO model was found. Run `python train.py --steps 25000` first.')
    else:
        info_path = ROOT / 'models' / 'policy_info.json'
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding='utf-8'))
            steps = int(info.get('training_steps', 0))
            status = 'short demonstration policy' if info.get('demonstration_checkpoint', True) else 'saved policy'
            st.caption(f'PPO {status} available · {steps:,} training steps. Retrain after modifying simulator behaviour.')
        else:
            st.caption('PPO checkpoint available. Verify training quality before research experiments.')

    with st.form('generate_transactions_form'):
        c1, c2, c3 = st.columns([1, 1, 1.5], gap='medium')
        with c1:
            tx_count = st.number_input('Transactions per run', min_value=100,
                                       max_value=100000, value=3000, step=500)
        with c2:
            days = st.number_input('Simulation duration (days)', min_value=1,
                                   max_value=365, value=30)
        with c3:
            seeds_text = st.text_input('Random seeds', '101, 202, 303',
                help='One to five distinct integer seeds, comma-separated. Every seed creates a separate dataset.')
        with st.expander('Wallet population (advanced)', expanded=False):
            wcols = st.columns(5)
            with wcols[0]: users = st.number_input('Normal users', 5, 2000, 120)
            with wcols[1]: merchants = st.number_input('Merchants', 1, 500, 24)
            with wcols[2]: exchanges = st.number_input('Exchanges', 1, 100, 5)
            with wcols[3]: scammers = st.number_input('Scammers', 1, 500, 15)
            with wcols[4]: launderers = st.number_input('Money launderers', 1, 500, 12)
        submit = st.form_submit_button('Generate transactions', type='primary',
                                       disabled=not MODEL_FILE.exists(), use_container_width=True)
    if submit:
        try:
            seeds = [int(part.strip()) for part in seeds_text.split(',') if part.strip()]
            if not seeds or len(seeds) > 5 or len(seeds) != len(set(seeds)):
                raise ValueError('Please provide between 1 and 5 distinct integer seeds.')
            cfg = Config(normal_users=users, merchants=merchants, exchanges=exchanges,
                         scammers=scammers, money_launderers=launderers,
                         target_transactions=tx_count, duration_days=days)
            cfg.validate()
            policy = available_policy()
            new_runs = {}
            with st.spinner('Generating datasets using the trained PPO policy…'):
                for seed in seeds:
                    run_id = f'run_{seed}'
                    result = generate_v5(policy, cfg, seed=seed, run_id=run_id)
                    save_run(OUTPUTS, result, run_id=run_id)
                    new_runs[run_id] = result
            st.session_state.runs = new_runs
            st.session_state.chosen_run = next(iter(new_runs))
            st.success(f'Generated {len(new_runs)} runs · {tx_count:,} transactions per run · {len(TX_COLUMNS)} columns.')
        except (ValueError, AssertionError, RuntimeError) as exc:
            st.error(str(exc))

    with st.expander('Load datasets saved previously in outputs/', expanded=False):
        if st.button('Load saved runs'):
            found = 0
            for path in sorted(OUTPUTS.glob('run_*/transactions.csv')):
                folder = path.parent
                try:
                    tx = pd.read_csv(path, parse_dates=['timestamp'])
                    wallets = pd.read_csv(folder / 'wallets.csv')
                    scenarios = pd.read_csv(folder / 'scenarios.csv')
                    meta = json.loads((folder / 'experiment.json').read_text())
                    if list(tx.columns) != TX_COLUMNS:
                        continue
                    st.session_state.runs[folder.name] = (tx, wallets, scenarios, meta)
                    st.session_state.chosen_run = folder.name
                    found += 1
                except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
                    continue
            st.success(f'Loaded {found} datasets.') if found else st.info('No saved V5 runs found.')

    if st.session_state.runs:
        st.divider()
        result = show_selection()
        tx, wallets, scenarios, meta = result
        diagnostics = describe_quality(tx, wallets)
        cols = st.columns(5)
        for col, label, value in zip(cols,
            ['Transactions', 'Fraud', 'Genuine', 'Wallets', 'Mixed-behaviour wallets'],
            [len(tx), int(tx.fraud_label.sum()), int((tx.fraud_label == 0).sum()),
             diagnostics['wallets'], diagnostics['mixed_sender_wallets']]):
            col.metric(label, f'{value:,}')
        st.download_button('Download selected transactions.csv',
                           data=tx.to_csv(index=False).encode('utf-8'),
                           file_name=f'transactions_{st.session_state.chosen_run}.csv',
                           mime='text/csv', use_container_width=True, type='primary')
        with st.expander('Download individual runs or all research files'):
            for name, (run_tx, *_rest) in st.session_state.runs.items():
                st.download_button(f'Download {name} transactions.csv',
                                   data=run_tx.to_csv(index=False).encode('utf-8'),
                                   file_name=f'transactions_{name}.csv', mime='text/csv',
                                   key=f'csv_{name}')
            st.download_button('Download all runs as ZIP (transactions, wallets, scenarios, metadata)',
                               data=zip_runs(st.session_state.runs),
                               file_name='crypto_ppo_simulation_results.zip',
                               mime='application/zip', key='download_all')
        st.caption('Scenario names and fraud labels are generated ground truth. They must not be used as fraud-model input features.')

with tab_tx:
    st.subheader('Transactions · 26 documented fields')
    if chosen_outputs() is None:
        st.info('Generate or load a run from Simulation first.')
    else:
        result = show_selection()
        tx = result[0]
        a, b, c = st.columns([1,1,1])
        with a:
            selected_label = st.selectbox('Fraud label', ['All', 'Genuine', 'Fraud'])
        with b:
            wallet_filter = st.text_input('Filter wallet ID (sender or receiver)')
        with c:
            scenario_filter = st.selectbox('Scenario type', ['All'] + sorted(tx.scenario_type.dropna().unique().tolist()))
        filtered = tx.copy()
        if selected_label != 'All':
            filtered = filtered[filtered.fraud_label.eq(1 if selected_label == 'Fraud' else 0)]
        if scenario_filter != 'All':
            filtered = filtered[filtered.scenario_type.eq(scenario_filter)]
        if wallet_filter:
            filtered = filtered[
                filtered.sender_wallet.str.contains(wallet_filter, case=False, regex=False)
                | filtered.receiver_wallet.str.contains(wallet_filter, case=False, regex=False)]
        st.dataframe(filtered, use_container_width=True, height=520)
        st.caption('scenario_type and scenario_id are simulator ground truth for explanation only. They are excluded from ML inputs; fraud_label is used separately as the supervised target.')

with tab_graph:
    st.subheader('Transaction Network Graph')
    if chosen_outputs() is None:
        st.info('Generate or load a run first.')
    else:
        tx, wallets, _, _ = show_selection()
        max_records = st.slider('Transactions to plot (for responsiveness)', 100, min(4000,len(tx)), min(1200,len(tx))) if len(tx) >= 100 else len(tx)
        graph = build_graph(tx.head(max_records), wallets)
        summary = network_summary(graph)
        x = st.columns(4)
        for c, label, val in zip(x,['Wallet nodes','Edges','Graph density','Avg degree'],
                                 [summary['nodes'], summary['edges'], f"{summary['density']:.4f}",f"{summary['avg_degree']:.2f}"]):
            c.metric(label, val)
        st.caption('Wallet role colors are simulation ground truth for network explanation, not features supplied to the fraud classifier. Red edges indicate known simulated fraud scenarios.')
        st.plotly_chart(network_figure(graph), use_container_width=True)

with tab_eda:
    st.subheader('Exploratory Data Analysis')
    if chosen_outputs() is None:
        st.info('Generate or load a run first.')
    else:
        tx, wallets, scenarios, _ = show_selection()
        metrics = describe_quality(tx, wallets)
        a,b,c = st.columns(3)
        a.metric('Fraud share',f"{metrics['fraud_rate']:.1%}")
        b.metric('Amount ranges overlap', 'Yes' if metrics['class_amount_ranges_overlap'] else 'No')
        c.metric('Genuine → scammer wallet',metrics['genuine_payments_to_scammer_wallet'])
        q = tx.copy()
        q['Class'] = q.fraud_label.map({0:'Genuine',1:'Fraud'})
        fig1 = px.histogram(q, x='amount_btc', color='Class', nbins=70,
                            marginal='box', barmode='overlay', title='Transaction amount overlap')
        fig1.update_xaxes(type='log')
        st.plotly_chart(fig1, use_container_width=True)
        daily = q.assign(day=pd.to_datetime(q.timestamp).dt.date).groupby(['day','Class']).size().reset_index(name='Transactions')
        st.plotly_chart(px.line(daily, x='day', y='Transactions', color='Class', title='Transactions over time'), use_container_width=True)
        f1,f2 = st.columns(2)
        with f1:
            st.plotly_chart(px.histogram(q, x='sender_tx_last_hour', color='Class',
                                        barmode='overlay', title='Sender activity in last hour'),use_container_width=True)
        with f2:
            st.plotly_chart(px.histogram(q, x='sender_unique_counterparties_before',color='Class',
                                        barmode='overlay',title='Previous connected wallets'),use_container_width=True)
        scenario_counts = q.scenario_type.value_counts().reset_index()
        scenario_counts.columns = ['scenario_type', 'transactions']
        st.plotly_chart(px.bar(scenario_counts, x='scenario_type', y='transactions',
                               title='Why a transaction was categorised (simulation ground truth)'),
                        use_container_width=True)
        st.caption('These are synthetic patterns. Statistical overlap alone does not establish similarity to real cryptocurrency data.')

with tab_model:
    st.subheader('Fraud Detection Modelling · separate from PPO')
    st.write('PPO generates transactions. Random Forest is trained separately to predict **fraud_label** from transaction, historical wallet, temporal and network features. The ground-truth label is always excluded from the prediction input X.')
    runs = st.session_state.runs
    if not runs:
        st.info('Generate or load datasets first.')
    else:
        if len(runs) < 3:
            st.warning('Only one/two runs available. You can test a chronological split, but independent train/validation/test runs are stronger.')
        mode = st.radio('Evaluation approach',
                        ['Independent runs (recommended)', 'Chronological single run'],horizontal=True)
        if mode.startswith('Independent') and len(runs)<3:
            st.info('Generate at least three runs first, e.g. 101, 202, 303.')
        elif st.button('Evaluate Random Forest', type='primary'):
            try:
                if mode.startswith('Independent'):
                    _, stats, importance = evaluate_runs({k:v[0] for k,v in runs.items()})
                else:
                    _, stats, importance = evaluate_temporal(chosen_outputs()[0])
                st.markdown('#### Evaluation')
                st.json(stats, expanded=True)
                st.plotly_chart(px.bar(importance.head(14), x='importance', y='feature',
                                       orientation='h',title='Allowed model features'),use_container_width=True)
                st.caption('A high synthetic test score is NOT evidence of real-world fraud detection performance.')
            except ValueError as e:
                st.error(str(e))

