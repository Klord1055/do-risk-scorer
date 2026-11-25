import numpy as np
import pandas as pd

from risk.model import (Company, benchmark_quotes, build_tower, claim_frequency, company_from_row, layer_metrics,
                        recommend_limit, recommended_tower_limit, limit_needed, recommend_retention, risk_score,
                        score_portfolio, simulate_losses)
import pytest

SAFE = Company("Safe", "Utilities", 14000, 5000, 0.17, 0.05, 1.0, 40)
RISKY = Company("Risky", "Biotech / Pharma", 900, 40, 0.85, 0.6, 0.2, 0.8, spac=True, restatement=True, prior_claims=1)


def test_score_bounds_and_order():
    s_safe, _ = risk_score(SAFE)
    s_risky, _ = risk_score(RISKY)
    assert 0 <= s_safe < s_risky <= 100


def test_breakdown_sums_to_score():
    s, df = risk_score(RISKY)
    assert abs(df["Contribution"].sum() - s) < 1e-9


def test_tower_covers_recommended_limit():
    s, _ = risk_score(RISKY)
    t = build_tower(RISKY, s, 10, 10)
    assert abs(t.total_limit - recommend_limit(RISKY, s)) < 1e-9
    abc = [l for l in t.layers if not l.side_a]
    for lower, upper in zip(abc, abc[1:]):
        assert abs(lower.attachment + lower.limit - upper.attachment) < 1e-9


def test_higher_layers_are_hit_less_often():
    s, _ = risk_score(RISKY)
    t = build_tower(RISKY, s, 10, 0)
    m = layer_metrics(t, simulate_losses(RISKY, s, 20000))
    probs = m["Prob. layer is hit"].tolist()
    assert all(a >= b for a, b in zip(probs, probs[1:]))


def test_benchmark_flags_expensive_quote():
    ind = pd.DataFrame({"Layer": ["Primary"], "Indicated premium ($M)": [1.0], "Expected loss ($M)": [0.5]})
    q = pd.DataFrame({"Layer": ["Primary"], "Carrier": ["X"], "Quoted premium ($M)": [1.3]})
    assert benchmark_quotes(q, ind)["Verdict"].iloc[0].startswith("Above")


def test_portfolio_sorted():
    df = score_portfolio(pd.read_csv("data/sample_companies.csv"))
    assert df["Risk score"].is_monotonic_decreasing and len(df) == 12


def test_simulated_claim_rate_matches_shown_probability():
    s, _ = risk_score(RISKY)
    L = simulate_losses(RISKY, s, 400_000)
    assert (L > 0).mean() == pytest.approx(claim_frequency(RISKY, s), rel=0.02)


def test_premium_is_risk_based():
    sr, _ = risk_score(RISKY); ss, _ = risk_score(SAFE)
    tr, ts = build_tower(RISKY, sr, 10, 0, total=10), build_tower(SAFE, ss, 10, 0, total=10)
    pr = layer_metrics(tr, simulate_losses(RISKY, sr))["Indicated premium ($M)"].iloc[0]
    ps = layer_metrics(ts, simulate_losses(SAFE, ss))["Indicated premium ($M)"].iloc[0]
    assert pr > ps


def test_recommended_tower_covers_1_in_100():
    s, _ = risk_score(RISKY)
    L = simulate_losses(RISKY, s)
    lim = recommended_tower_limit(RISKY, s, L)
    assert lim >= limit_needed(L, recommend_retention(RISKY), 100) or lim == 500


def test_unmatched_quote_is_flagged():
    ind = pd.DataFrame({"Layer": ["Primary"], "Indicated premium ($M)": [1.0], "Expected loss ($M)": [0.5]})
    q = pd.DataFrame({"Layer": ["Bogus"], "Carrier": ["X"], "Quoted premium ($M)": [1.0]})
    assert benchmark_quotes(q, ind)["Verdict"].iloc[0] == "No matching layer"


def test_text_flags_and_bad_rows():
    df = pd.read_csv("data/sample_companies.csv").head(3).astype(object)
    df.loc[0, "spac"] = "no"
    df.loc[1, "volatility"] = "abc"
    out = score_portfolio(df)
    assert len(out) == 2 and len(out.attrs["skipped"]) == 1
    assert company_from_row(df.iloc[0]).spac is False
