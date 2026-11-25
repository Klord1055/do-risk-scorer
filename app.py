"""D&O Program Risk Scorer: score a public company's D&O exposure, simulate losses,
build an insurance tower, and benchmark carrier quotes."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk.model import (
    DEFAULT_WEIGHTS, SECTOR_BASE_FREQ, Company, benchmark_quotes, build_tower,
    claim_frequency, company_from_row, layer_metrics, recommend_limit,
    recommend_retention, risk_score, risk_tier, score_portfolio, simulate_losses,
)

st.set_page_config(page_title="D&O Program Risk Scorer", page_icon="🛡️", layout="wide")
DATA = Path(__file__).parent / "data" / "sample_companies.csv"
NAVY = "#1F2A44"
TIER_COLORS = {"Low": "#2E8B57", "Moderate": "#C9A227", "Elevated": "#E07B39", "High": "#C0392B"}


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(DATA)


sample = load_sample()

st.title("D&O Program Risk Scorer")
st.caption(
    "Scores a public company's directors and officers liability exposure, simulates annual losses, "
    "designs an insurance tower, and benchmarks carrier quotes against a modeled price. "
    "Sample companies are fictional. Model parameters are illustrative assumptions you can change."
)

# Sidebar: company inputs
with st.sidebar:
    st.header("Company")
    pick = st.selectbox("Load a sample company", ["Custom"] + sample["name"].tolist(), index=1)
    base = company_from_row(sample[sample["name"] == pick].iloc[0]) if pick != "Custom" else Company(
        "Custom Co", "Technology", 2000, 500, 0.45, 0.25, 0.5, 5)
    name = st.text_input("Name", base.name)
    sector = st.selectbox("Sector", list(SECTOR_BASE_FREQ), index=list(SECTOR_BASE_FREQ).index(base.sector))
    mcap = st.number_input("Market cap ($M)", 50.0, 500000.0, float(base.market_cap_m), step=100.0)
    rev = st.number_input("Revenue ($M)", 0.0, 500000.0, float(base.revenue_m), step=50.0)
    vol = st.slider("Annualized stock volatility", 0.10, 1.20, float(base.volatility), 0.01)
    dd = st.slider("52 week drawdown from peak", 0.0, 0.90, float(base.drawdown), 0.01)
    de = st.slider("Debt to equity", 0.0, 5.0, float(base.debt_to_equity), 0.05)
    yrs = st.number_input("Years public", 0.0, 100.0, float(base.years_public), step=0.5)
    spac = st.checkbox("Went public via SPAC", base.spac)
    ma = st.checkbox("Major M&A in last 2 years", base.major_ma)
    restate = st.checkbox("Restatement or SEC inquiry", base.restatement)
    prior = st.number_input("Prior securities claims (5 yrs)", 0, 10, int(base.prior_claims))

    with st.expander("Factor weights"):
        weights = {k: st.slider(k, 0.0, 0.5, v, 0.01, key=f"w_{k}") for k, v in DEFAULT_WEIGHTS.items()}

company = Company(name, sector, mcap, rev, vol, dd, de, yrs, spac, ma, restate, prior)
score, breakdown = risk_score(company, weights)
tier = risk_tier(score)
freq = claim_frequency(company, score)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Risk score", "Loss model and tower", "Quote benchmark", "Portfolio screen", "Methodology"])

# Tab 1: score
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{score:.0f} / 100")
    c2.metric("Tier", tier)
    c3.metric("Annual claim probability", f"{freq:.1%}")
    c4.metric("Recommended ABC limit", f"${recommend_limit(company, score):,.0f}M")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
               "steps": [{"range": [0, 20], "color": "#D5EFDD"}, {"range": [20, 40], "color": "#F6EBC0"},
                         {"range": [40, 65], "color": "#F8D9C4"}, {"range": [65, 100], "color": "#F3C6C1"}]}))
    gauge.update_layout(height=260, margin=dict(t=20, b=10, l=30, r=30))

    bar = px.bar(breakdown.sort_values("Contribution"), x="Contribution", y="Factor", orientation="h",
                 hover_data={"Score": ":.0f", "Weight": ":.0%"}, color_discrete_sequence=[NAVY])
    bar.update_layout(height=320, margin=dict(t=30, b=10), title="What drives the score")

    a, b = st.columns([1, 1.4])
    a.plotly_chart(gauge, use_container_width=True)
    b.plotly_chart(bar, use_container_width=True)

    top = breakdown.iloc[0]
    st.info(f"**Biggest driver:** {top['Factor']} (factor score {top['Score']:.0f}). "
            f"{name} sits in the **{tier}** tier, about {freq / SECTOR_BASE_FREQ[sector]:.1f}x the "
            f"{sector.lower()} sector's baseline claim rate.")

# Tab 2: loss model and tower
with tab2:
    st.subheader("Simulated annual D&O loss")
    sims = st.select_slider("Simulations", [10_000, 25_000, 50_000, 100_000], 50_000)
    losses = simulate_losses(company, score, n=sims)
    hit = losses[losses > 0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected annual loss", f"${losses.mean():,.2f}M")
    m2.metric("Loss if a claim hits (median)", f"${np.median(hit):,.1f}M" if len(hit) else "n/a")
    m3.metric("1 in 100 year loss", f"${np.percentile(losses, 99):,.1f}M")
    m4.metric("1 in 250 year loss", f"${np.percentile(losses, 99.6):,.1f}M")

    if len(hit):
        hist = px.histogram(x=hit, nbins=60, log_y=True, color_discrete_sequence=[NAVY],
                            labels={"x": "Loss when a claim occurs ($M)"})
        hist.update_layout(height=280, margin=dict(t=10, b=10), yaxis_title="Simulated years (log)")
        st.plotly_chart(hist, use_container_width=True)

    st.subheader("Tower design")
    t1, t2, t3, t4, t5 = st.columns(5)
    layer_size = t1.selectbox("Layer size ($M)", [5.0, 10.0, 15.0, 25.0], index=1)
    side_a = t2.selectbox("Side A DIC limit ($M)", [0.0, 5.0, 10.0, 15.0, 25.0], index=2)
    primary_rol = t3.slider("Primary rate on line", 0.02, 0.15, 0.06, 0.005, format="%.3f")
    decay = t4.slider("Excess pricing step down", 0.55, 0.95, 0.75, 0.01, help="Each excess layer is priced at this share of the layer below it")
    side_a_rol = t5.slider("Side A rate on line", 0.01, 0.06, 0.025, 0.005, format="%.3f")

    tower = build_tower(company, score, layer_size, side_a)
    metrics = layer_metrics(tower, losses, primary_rol, decay, side_a_rol)

    fig = go.Figure()
    for _, r in metrics.iloc[::-1].iterrows():
        color = "#8E7CC3" if r["Layer"].startswith("Side A") else NAVY
        fig.add_trace(go.Bar(x=[r["Limit ($M)"]], y=[r["Layer"]], base=[r["Attaches at ($M)"]], orientation="h",
                             marker_color=color, showlegend=False,
                             text=f"${r['Limit ($M)']:.0f}M xs ${r['Attaches at ($M)']:.1f}M · hit {r['Prob. layer is hit']:.1%}",
                             textposition="inside", insidetextanchor="middle"))
    fig.add_vline(x=tower.retention, line_dash="dot", annotation_text=f"Retention ${tower.retention:g}M")
    fig.update_layout(height=60 + 38 * len(metrics), margin=dict(t=30, b=10), xaxis_title="$ millions",
                      title=f"${tower.total_limit:,.0f}M ABC tower over a ${tower.retention:g}M retention")
    st.plotly_chart(fig, use_container_width=True)

    show = metrics.copy()
    st.dataframe(show.style.format({
        "Attaches at ($M)": "{:,.1f}", "Limit ($M)": "{:,.0f}", "Prob. layer is hit": "{:.2%}",
        "Expected loss ($M)": "{:,.3f}", "Rate on line": "{:.2%}", "Indicated premium ($M)": "{:,.3f}",
        "Loss ratio": "{:.0%}"}), use_container_width=True, hide_index=True)
    tot_prem = metrics["Indicated premium ($M)"].sum()
    st.caption(f"Total indicated program premium: **${tot_prem:,.2f}M**. Loss ratio is modeled expected loss "
               "divided by premium; layers far below the program average are where you have room to negotiate.")
    st.session_state["metrics"] = metrics

# Tab 3: carrier quote benchmark
with tab3:
    st.subheader("Benchmark carrier quotes")
    st.write("Enter or edit quotes by layer. The table compares each quote to the model's indicated price "
             "for the same layer and flags where to push back.")
    metrics = st.session_state.get("metrics")
    if metrics is not None:
        rng = np.random.default_rng(3)
        carriers = ["Carrier A", "Carrier B", "Carrier C", "Carrier D", "Carrier E", "Carrier F", "Carrier G"]
        default = pd.DataFrame({
            "Layer": metrics["Layer"],
            "Carrier": [carriers[i % len(carriers)] for i in range(len(metrics))],
            "Quoted premium ($M)": (metrics["Indicated premium ($M)"] * rng.uniform(0.85, 1.25, len(metrics))).round(3),
        })
        quotes = st.data_editor(default, num_rows="dynamic", use_container_width=True, hide_index=True)
        bench = benchmark_quotes(quotes, metrics)
        st.dataframe(bench.style.format({
            "Quoted premium ($M)": "{:,.3f}", "Indicated premium ($M)": "{:,.3f}", "Expected loss ($M)": "{:,.3f}",
            "Quote vs indicated": "{:+.0%}", "Quote loss ratio": "{:.0%}"}),
            use_container_width=True, hide_index=True)
        diff = (bench["Quoted premium ($M)"] - bench["Indicated premium ($M)"]).sum()
        st.metric("Program cost vs indication", f"${diff:+,.3f}M",
                  help="Positive means the quoted program costs more than the model indicates")
        st.caption("Default quotes are randomly generated around the indication for demonstration. Replace them with real quotes.")

# Tab 4: portfolio
with tab4:
    st.subheader("Screen a book of companies")
    st.write("Upload a CSV with the same columns as the sample file, or use the 12 fictional sample companies.")
    up = st.file_uploader("CSV", type="csv")
    book = pd.read_csv(up) if up else sample
    port = score_portfolio(book, weights)
    sc = px.scatter(port, x="Market cap ($M)", y="Risk score", color="Tier", size="Recommended limit ($M)",
                    hover_name="Company", log_x=True, color_discrete_map=TIER_COLORS)
    sc.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(sc, use_container_width=True)
    st.dataframe(port.style.format({"Market cap ($M)": "{:,.0f}", "Claim probability": "{:.1%}",
                                    "Recommended limit ($M)": "{:,.0f}", "Retention ($M)": "{:,.2f}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("Download results (CSV)", port.to_csv(index=False), "do_portfolio_scores.csv")
    with st.expander("CSV template columns"):
        st.code(",".join(sample.columns))

# Tab 5: methodology
with tab5:
    st.markdown((Path(__file__).parent / "HOW_IT_WORKS.md").read_text())
