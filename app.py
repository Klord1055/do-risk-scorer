"""D&O Program Risk Scorer: score a public company's D&O exposure, simulate losses, test whether the
tower is big enough, benchmark carrier quotes, and show the fastest way to lower the risk."""
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from risk.model import (DEFAULT_WEIGHTS, SECTOR_BASE_FREQ, Company, benchmark_quotes, best_quotes, build_tower,
                        claim_frequency, company_from_row, exhaustion_probability, improvement_paths,
                        layer_metrics, limit_needed, recommend_limit, recommend_retention,
                        recommended_tower_limit, risk_score, risk_tier, score_portfolio, simulate_losses)
from ui import intro, note, GOLD, GREEN, NAVY, RED, setup, takeaway

st.set_page_config(page_title="D&O Program Risk Scorer", page_icon="🛡️", layout="wide")
setup()
W = "stretch"
ROOT = Path(__file__).parent
TIER_COLORS = {"Low": GREEN, "Moderate": GOLD, "Elevated": "#C25E00", "High": RED}
REQUIRED = ["name", "sector", "market_cap_m", "revenue_m", "volatility", "drawdown", "debt_to_equity",
            "years_public", "spac", "major_ma", "restatement", "prior_claims"]


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "sample_companies.csv")


@st.cache_data
def losses_for(c: Company, score: float, n: int) -> np.ndarray:
    return simulate_losses(c, score, n=n)


def reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


sample = load_sample()

with st.sidebar:
    st.button("Reset to defaults", on_click=reset, width=W)
    st.header("1. Company")
    pick = st.selectbox("Start from a sample company", ["Custom"] + sample["name"].tolist(), index=1, key="pick",
                        help="Twelve fictional companies. Pick one, then change anything below.")
    base = company_from_row(sample[sample["name"] == pick].iloc[0]) if pick != "Custom" else Company(
        "Custom Co", "Technology", 2000, 500, 0.45, 0.25, 0.5, 5)
    k = f"_{pick}"  # new widget keys per company so its values load fresh
    name = st.text_input("Name", base.name, key="name" + k)
    sector = st.selectbox("Sector", list(SECTOR_BASE_FREQ), index=list(SECTOR_BASE_FREQ).index(base.sector), key="sector" + k,
                          help="Some sectors get sued more often. Biotech and tech have the highest base rates.")
    mcap = st.number_input("Market cap ($M)", 50.0, 500000.0, float(base.market_cap_m), step=100.0, key="mcap" + k,
                           help="Bigger companies face bigger claims when the stock drops.")
    rev = st.number_input("Revenue ($M)", 0.0, 500000.0, float(base.revenue_m), step=50.0, key="rev" + k)
    st.header("2. Stock and balance sheet")
    vol = st.slider("Stock volatility (annual)", 10, 120, int(round(base.volatility * 100)), 1, format="%d%%", key="vol" + k,
                    help="How much the stock swings in a year. Big swings lead to more shareholder suits.") / 100
    dd = st.slider("Drop from 52 week high", 0, 90, int(round(base.drawdown * 100)), 1, format="%d%%", key="dd" + k,
                   help="Sharp drops are the most common trigger for securities class actions.") / 100
    de = st.slider("Debt to equity", 0.0, 5.0, float(base.debt_to_equity), 0.05, format="%.2fx", key="de" + k)
    yrs = st.number_input("Years since going public", 0.0, 100.0, float(base.years_public), step=0.5, key="yrs" + k,
                          help="The first three years after an IPO carry the highest claim risk.")
    st.header("3. Red flags")
    spac = st.checkbox("Went public through a SPAC", base.spac, key="spac" + k)
    ma = st.checkbox("Major merger or acquisition in last 2 years", base.major_ma, key="ma" + k)
    restate = st.checkbox("Restatement or SEC inquiry", base.restatement, key="restate" + k)
    prior = st.number_input("Securities claims in last 5 years", 0, 10, int(base.prior_claims), key="prior" + k)
    with st.expander("Advanced: factor weights"):
        weights = {f: st.slider(f, 0.0, 0.5, v, 0.01, key=f"w_{f}") for f, v in DEFAULT_WEIGHTS.items()}
        if sum(weights.values()) == 0:
            st.warning("All weights are zero, so default weights are used.")
            weights = DEFAULT_WEIGHTS

company = Company(name, sector, mcap, rev, vol, dd, de, yrs, spac, ma, restate, prior)
score, breakdown = risk_score(company, weights)
score = round(score, 1)
tier = risk_tier(score)
freq = claim_frequency(company, score)

# Tower settings come from the tower tab but feed the headline
ts = st.session_state.get("tower", dict(layer=10.0, side_a=10.0, tlr=60, minrol=1.0, sims=50_000, total=0))
losses = losses_for(company, score, ts["sims"])
rec_total = recommended_tower_limit(company, score, losses)
bench_total = recommend_limit(company, score)
total = ts["total"] or rec_total
tower = build_tower(company, score, ts["layer"], ts["side_a"], total=total)
metrics = layer_metrics(tower, losses, ts["tlr"] / 100, ts["minrol"] / 100)
premium = float(metrics["Indicated premium ($M)"].sum())
exhaust = exhaustion_probability(losses, tower)
need100 = limit_needed(losses, tower.retention, 100)
need250 = limit_needed(losses, tower.retention, 250)
color = TIER_COLORS[tier]

st.title("D&O Program Risk Scorer")
intro("Scores a public company's directors and officers liability risk, simulates a year of claims, checks "
           "whether the insurance tower is big enough, and benchmarks carrier quotes. Sample companies are fictional; "
           "model settings are illustrative and every one can be changed.")

st.markdown(
    f"<div style='padding:16px 20px;border-radius:10px;border:2px solid {color};background:#FFFFFF'>"
    f"<div style='font-size:15px;color:#2B2F36;font-weight:600'>RISK TIER</div>"
    f"<div style='font-size:34px;font-weight:800;color:{color};line-height:1.2'>{tier} ({score:.0f} / 100)</div>"
    f"<div style='font-size:18px;color:#111'>{html.escape(name)} has a <b>{freq:.1%}</b> chance of a securities claim this year, "
    f"{freq / SECTOR_BASE_FREQ[sector]:.1f}x the {sector.lower()} average. A <b>${tower.total_limit:,.0f}M</b> tower runs out in "
    f"<b>{exhaust:.1%}</b> of simulated years.</div></div>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Claim odds this year", f"{freq:.1%}")
m2.metric("Tower limit", f"${tower.total_limit:,.0f}M",
          help="Total limit across primary and excess layers, above the company's retention (deductible). "
               "The recommendation covers a 1 in 100 year loss and is never below the size benchmark.")
note(m2, f"Over a ${tower.retention:g}M retention, {'recommended' if not ts['total'] else 'your choice'}.")
m3.metric("Size benchmark", f"${bench_total:,.0f}M",
          help="A rule of thumb limit from market cap and risk score, the way peers of this size are often insured.")
note(m3, f"The 1 in 100 year loss needs <b>${need100:,.0f}M</b> above the retention.")
m4.metric("Program premium", f"${premium:,.2f}M", help="Risk based: expected loss divided by the target loss ratio, with a minimum rate on line.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Risk drivers", "How to lower the risk", "Losses and tower", "Quote benchmark", "Portfolio screen", "Methodology"])

with tab1:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " / 100", "font": {"size": 40, "color": "#111"}},
        gauge={"axis": {"range": [0, 100], "tickfont": {"size": 14}}, "bar": {"color": "#111", "thickness": 0.25},
               "steps": [{"range": [0, 20], "color": "#BFE3CC"}, {"range": [20, 40], "color": "#F3DFA2"},
                         {"range": [40, 65], "color": "#F5C49C"}, {"range": [65, 100], "color": "#F0A8A0"}]}))
    gauge.update_layout(height=270, margin=dict(t=30, b=10, l=30, r=30))
    bar = px.bar(breakdown.sort_values("Contribution"), x="Contribution", y="Factor", orientation="h",
                 hover_data={"Score": ":.0f", "Weight": ":.0%"}, color_discrete_sequence=[NAVY],
                 labels={"Contribution": "Points added to the score", "Factor": ""})
    bar.update_layout(height=330, margin=dict(t=40, b=10), title="What drives the score")
    a, b = st.columns([1, 1.4])
    a.plotly_chart(gauge, width=W, theme=None)
    b.plotly_chart(bar, width=W, theme=None)
    top = breakdown.iloc[0]
    takeaway(f"<b>{top['Factor']}</b> adds the most risk ({top['Contribution']:.0f} of {score:.0f} points). "
             f"Tiers: Low under 20, Moderate 20 to 40, Elevated 40 to 65, High 65 and up.")

with tab2:
    st.subheader("Fastest ways to lower the score")
    paths = improvement_paths(company, weights)
    if paths.empty:
        st.success("No controllable red flags. The score is driven by sector, size and listing history, which management cannot quickly change.")
    else:
        pf = px.bar(paths.iloc[::-1], x="Points saved", y="Change", orientation="h", color_discrete_sequence=[GREEN],
                    text=paths.iloc[::-1]["New tier"], labels={"Change": ""})
        pf.update_traces(textposition="outside", textfont=dict(size=14))
        pf.update_layout(height=80 + 55 * len(paths), margin=dict(t=10, b=10))
        st.plotly_chart(pf, width=W, theme=None)
        best = paths.iloc[0]
        if best["Points saved"] < 0.5:
            st.info("With the current factor weights none of these fixes changes the score. Check the weights in the left panel.")
        lever = best["Change"][0].lower() + best["Change"][1:]
        takeaway(f"The single biggest lever is to <b>{lever}</b>: the score drops <b>{best['Points saved']:.0f}</b> points to "
                 f"{best['New score']:.0f} ({best['New tier']}) and claim odds fall from {freq:.1%} to <b>{best['New claim probability']:.1%}</b>. "
                 "Use this in renewal meetings to show underwriters what has improved.")
        st.dataframe(paths.style.format({"New score": "{:.0f}", "Points saved": "{:.1f}", "New claim probability": "{:.1%}"}),
                     hide_index=True, width=W)

with tab3:
    st.subheader("Simulated annual D&O loss")
    hit = losses[losses > 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average annual loss", f"${losses.mean():,.2f}M", help="Defense costs plus settlements, averaged over every simulated year.")
    c2.metric("Typical claim if one hits", f"${np.median(hit):,.1f}M" if len(hit) else "n/a")
    c3.metric("1 in 100 year loss", f"${np.percentile(losses, 99):,.1f}M")
    c4.metric("1 in 250 year loss", f"${np.percentile(losses, 99.6):,.1f}M")
    if len(hit):
        hist = px.histogram(x=hit, nbins=60, log_y=True, color_discrete_sequence=[NAVY], labels={"x": "Loss when a claim hits ($M)"})
        hist.add_vline(x=tower.retention + tower.total_limit, line_color=RED, line_width=3,
                       annotation_text=f"Top of tower ${tower.retention + tower.total_limit:,.0f}M", annotation_font=dict(color=RED, size=14))
        hist.update_layout(height=300, margin=dict(t=30, b=10), yaxis_title="Simulated years (log scale)")
        st.plotly_chart(hist, width=W, theme=None)

    st.subheader("Tower design")
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    tot = t1.number_input("Total limit ($M), 0 = recommended", 0.0, 500.0, float(ts["total"]), 5.0,
                          help=f"Leave at 0 to use the recommended ${rec_total:,.0f}M, or type the limit you want to test.")
    layer = t2.selectbox("Layer size ($M)", [5.0, 10.0, 15.0, 25.0], index=[5.0, 10.0, 15.0, 25.0].index(ts["layer"]))
    side_a = t3.selectbox("Side A limit ($M)", [0.0, 5.0, 10.0, 15.0, 25.0], index=[0.0, 5.0, 10.0, 15.0, 25.0].index(ts["side_a"]),
                          help="Side A protects individual directors when the company cannot pay, for example in bankruptcy.")
    tlr = t4.slider("Target loss ratio", 30, 90, ts["tlr"], 5, format="%d%%",
                    help="Share of premium a carrier expects to pay out in claims. Lower means pricier insurance.")
    minrol = t5.slider("Minimum rate on line", 0.25, 5.0, ts["minrol"], 0.25, format="%.2f%%",
                       help="Floor price for remote excess layers, as a % of the layer's limit.")
    sims = t6.select_slider("Simulations", [10_000, 25_000, 50_000, 100_000], ts["sims"])
    new = dict(layer=layer, side_a=side_a, tlr=tlr, minrol=minrol, sims=sims, total=tot)
    if new != ts:
        st.session_state["tower"] = new
        st.rerun()

    fig = go.Figure()
    for _, r in metrics.iloc[::-1].iterrows():
        sa = r["Layer"].startswith("Side A")
        fig.add_trace(go.Bar(x=[r["Limit ($M)"]], y=[r["Layer"]], base=[r["Attaches at ($M)"]], orientation="h",
                             marker_color="#6B4FA8" if sa else NAVY, showlegend=False,
                             text=f"${r['Limit ($M)']:.0f}M xs ${r['Attaches at ($M)']:.1f}M · hit {r['Prob. layer is hit']:.1%}",
                             textposition="inside", insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=14)))
    fig.add_vline(x=tower.retention, line_dash="dot", line_width=2, annotation_text=f"Retention ${tower.retention:g}M")
    fig.update_layout(height=80 + 42 * len(metrics), margin=dict(t=40, b=10), xaxis_title="$ millions",
                      title=f"${tower.total_limit:,.0f}M tower over a ${tower.retention:g}M retention")
    st.plotly_chart(fig, width=W, theme=None)
    st.dataframe(metrics.style.format({
        "Attaches at ($M)": "{:,.1f}", "Limit ($M)": "{:,.0f}", "Prob. layer is hit": "{:.2%}", "Expected loss ($M)": "{:,.3f}",
        "Rate on line": "{:.2%}", "Indicated premium ($M)": "{:,.3f}", "Loss ratio": "{:.0%}"}), width=W, hide_index=True)
    abc = metrics[~metrics["Layer"].str.startswith("Side A")]
    msg = (f"The tower runs out in <b>{exhaust:.1%}</b> of years. Covering a 1 in 100 year loss takes <b>${need100:,.0f}M</b>, "
           + (f"<b>${need100 - tower.total_limit:,.0f}M more</b> than this tower. " if need100 > tower.total_limit else "which this tower does. "))
    if freq < 0.01:
        msg += (f"Claims are rare for this company ({freq:.1%} a year), so the 1 in 100 year loss is small; the 1 in 250 year "
                f"loss needs ${need250:,.0f}M, and a claim that does hit typically costs ${np.median(losses[losses > 0]):,.0f}M. ")
    floor = abc[abc["Rate on line"] <= ts["minrol"] / 100 + 1e-9]
    if len(floor):
        msg += (f"<b>{', '.join(floor['Layer'])}</b> {'is' if len(floor) == 1 else 'are'} priced at the minimum rate, so expected losses there are tiny "
                "and any quote well above the floor is worth pushing back on.")
    takeaway(msg)
    st.session_state["metrics_for_quotes"] = metrics

with tab4:
    st.subheader("Benchmark carrier quotes")
    rng = np.random.default_rng(3)
    carriers = ["Carrier A", "Carrier B", "Carrier C", "Carrier D", "Carrier E", "Carrier F", "Carrier G"]
    default = pd.DataFrame({
        "Layer": metrics["Layer"],
        "Carrier": [carriers[i % len(carriers)] for i in range(len(metrics))],
        "Quoted premium ($M)": (metrics["Indicated premium ($M)"] * rng.uniform(0.85, 1.25, len(metrics))).round(3)})
    quotes = st.data_editor(default, num_rows="dynamic", width=W, hide_index=True, key=f"quotes_{len(metrics)}",
                            column_config={"Layer": st.column_config.SelectboxColumn(options=list(metrics["Layer"])),
                                           "Quoted premium ($M)": st.column_config.NumberColumn(min_value=0.0, format="%.3f")})
    q = quotes.dropna(subset=["Layer", "Quoted premium ($M)"])
    q = q[q["Quoted premium ($M)"] > 0]
    if q.empty:
        st.warning("Enter at least one quote above zero to benchmark.")
    else:
        bench = benchmark_quotes(q, metrics)
        st.dataframe(bench.style.format({
            "Quoted premium ($M)": "{:,.3f}", "Indicated premium ($M)": "{:,.3f}", "Expected loss ($M)": "{:,.3f}",
            "Quote vs indicated": "{:+.0%}", "Quote loss ratio": "{:.0%}"}, na_rep="n/a")
            .map(lambda v: f"color:{RED};font-weight:700" if v == "Above indication, push back" else
                 (f"color:{GREEN};font-weight:700" if v == "Below indication, best value" else ""), subset=["Verdict"]),
            width=W, hide_index=True)
        bq = best_quotes(bench)
        pushes = int((bq["Verdict"] == "Above indication, push back").sum())
        if len(bq):
            diff = float((bq["Quoted premium ($M)"] - bq["Indicated premium ($M)"]).sum())
            word = "above" if diff > 0 else "below"
            takeaway(f"Taking the lowest quote for each of the <b>{len(bq)}</b> quoted layers, the program costs <b>${abs(diff):,.3f}M {word}</b> "
                     f"the model's price. <b>{pushes}</b> of those layers {'is' if pushes == 1 else 'are'} 15% or more above indication and worth pushing back on. "
                     "Default quotes are made up for demonstration; type in real ones.")
        bad = bench[bench["Verdict"].isin(["No matching layer", "Check quote"])]
        if len(bad):
            st.warning(f"{len(bad)} quote row{'s' if len(bad) > 1 else ''} could not be compared: pick a layer from the list and enter a premium above zero.")
        st.download_button("Download quote comparison (CSV)", bench.to_csv(index=False), f"{name}_quote_benchmark.csv")

with tab5:
    st.subheader("Screen a book of companies")
    up = st.file_uploader("Upload a CSV with the same columns as the template below, or use the 12 samples", type="csv")
    book = sample
    if up:
        try:
            book = pd.read_csv(up)
            missing = [c for c in REQUIRED if c not in book.columns]
            if missing:
                st.error(f"Your file is missing these columns: {', '.join(missing)}. Showing the samples instead.")
                book = sample
            elif not book["sector"].isin(SECTOR_BASE_FREQ).all():
                st.warning("Some sectors were not recognized and use the average base rate. Valid sectors: " + ", ".join(SECTOR_BASE_FREQ))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file ({e}). Showing the samples instead.")
            book = sample
    try:
        port = score_portfolio(book, weights)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not score that file ({e}). Showing the samples instead.")
        port = score_portfolio(sample, weights)
    if port.attrs.get("skipped"):
        st.warning("Skipped rows that could not be read: " + "; ".join(port.attrs["skipped"][:8])
                   + (" ..." if len(port.attrs["skipped"]) > 8 else ""))
    if port.empty:
        st.error("No rows could be scored. Showing the samples instead.")
        port = score_portfolio(sample, weights)
    sc = px.scatter(port, x="Market cap ($M)", y="Risk score", color="Tier", size="Benchmark limit ($M)",
                    hover_name="Company", log_x=True, color_discrete_map=TIER_COLORS)
    sc.update_layout(height=390, margin=dict(t=10, b=10))
    st.plotly_chart(sc, width=W, theme=None)
    high = port[port["Tier"].isin(["High", "Elevated"])]
    takeaway(f"<b>{len(high)} of {len(port)}</b> companies are Elevated or High. Highest: <b>{port.iloc[0]['Company']}</b> "
             f"at {port.iloc[0]['Risk score']:.0f}.")
    st.dataframe(port.style.format({"Market cap ($M)": "{:,.0f}", "Claim probability": "{:.1%}",
                                    "Benchmark limit ($M)": "{:,.0f}", "Retention ($M)": "{:,.2f}"}), width=W, hide_index=True)
    st.download_button("Download results (CSV)", port.to_csv(index=False), "do_portfolio_scores.csv")
    st.download_button("Download blank template (CSV)", sample.head(0).to_csv(index=False), "do_template.csv")

with tab6:
    st.markdown((ROOT / "HOW_IT_WORKS.md").read_text())
