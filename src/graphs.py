
from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


ROLE_SYMBOL = {
    "Normal User": "circle",
    "Merchant": "square",
    "Exchange": "triangle-up",
    "Scammer": "hexagon",
    "Money Launderer": "diamond",
    "External Service": "star",
}

FRAUD_COLOR = {
    "Legitimate": "#2563eb",
    "Fraudulent": "#dc2626",
    "External": "#6b7280",
}


def build_graph(transactions: pd.DataFrame, wallets: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    wallet_meta = wallets.set_index("wallet_id").to_dict("index") if not wallets.empty else {}

    for wid, meta in wallet_meta.items():
        role = meta.get("agent_type", "Unknown")
        fraud_status = (
            "Fraudulent" if role in {"Scammer", "Money Launderer"}
            else "External" if role == "External Service"
            else "Legitimate"
        )
        G.add_node(wid, agent_type=role, fraud_status=fraud_status)

    for _, row in transactions.iterrows():
        s, r = row["sender_wallet"], row["receiver_wallet"]
        if not G.has_node(s):
            G.add_node(s, agent_type="Unknown", fraud_status="External")
        if not G.has_node(r):
            G.add_node(r, agent_type="Unknown", fraud_status="External")
        if G.has_edge(s, r):
            G[s][r]["count"] += 1
            G[s][r]["amount"] += float(row["amount_btc"])
            G[s][r]["fraud_count"] += int(row["fraud_label"])
        else:
            G.add_edge(
                s, r,
                count=1,
                amount=float(row["amount_btc"]),
                fraud_count=int(row["fraud_label"]),
            )
    return G


def network_summary(G: nx.DiGraph) -> dict:
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "avg_degree": 0.0}
    degrees = [d for _, d in G.degree()]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(degrees) / max(1, len(degrees)),
    }


def network_figure(G: nx.DiGraph, seed: int = 42, max_nodes: int = 80) -> go.Figure:
    if G.number_of_nodes() == 0:
        return go.Figure()

    # Keep graph readable in the browser by selecting highest-degree nodes.
    if G.number_of_nodes() > max_nodes:
        ranked = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = [n for n, _ in ranked]
        H = G.subgraph(keep).copy()
    else:
        H = G

    pos = nx.spring_layout(H, seed=seed, k=0.7)

    edge_traces = []
    for fraud_edge, color, name in [
        (False, "#94a3b8", "Genuine/other flow"),
        (True, "#ef4444", "Fraud-labelled flow"),
    ]:
        x, y = [], []
        for u, v, data in H.edges(data=True):
            edge_is_fraud = data.get("fraud_count", 0) > 0
            if edge_is_fraud != fraud_edge:
                continue
            x.extend([pos[u][0], pos[v][0], None])
            y.extend([pos[u][1], pos[v][1], None])
        edge_traces.append(
            go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(width=1.2 if fraud_edge else 0.9, color=color),
                hoverinfo="none", name=name,
            )
        )

    node_traces = []
    for role in ROLE_SYMBOL:
        xs, ys, text, colors, sizes = [], [], [], [], []
        for node, attrs in H.nodes(data=True):
            if attrs.get("agent_type") != role:
                continue
            xs.append(pos[node][0])
            ys.append(pos[node][1])
            status = attrs.get("fraud_status", "External")
            colors.append(FRAUD_COLOR[status])
            degree = H.degree(node)
            sizes.append(11 + min(22, degree * 1.7))
            text.append(
                f"<b>{node}</b><br>Role: {role}<br>Status: {status}<br>Degree: {degree}"
            )
        if xs:
            node_traces.append(
                go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(
                        symbol=ROLE_SYMBOL[role],
                        size=sizes,
                        color=colors,
                        line=dict(width=1, color="white"),
                    ),
                    text=text, hoverinfo="text", name=role,
                )
            )

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h"),
        hovermode="closest",
    )
    return fig
