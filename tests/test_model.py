import numpy as np
import pandas as pd

from risk.model import (Company, benchmark_quotes, build_tower, layer_metrics, recommend_limit,
                        risk_score, score_portfolio, simulate_losses)

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
    m = layer_metrics(t, simulate_losses(RISKY, s, 20000), 0.06, 0.75, 0.025)
    probs = m["Prob. layer is hit"].tolist()
    assert all(a >= b for a, b in zip(probs, probs[1:]))


def test_benchmark_flags_expensive_quote():
    ind = pd.DataFrame({"Layer": ["Primary"], "Indicated premium ($M)": [1.0], "Expected loss ($M)": [0.5]})
    q = pd.DataFrame({"Layer": ["Primary"], "Carrier": ["X"], "Quoted premium ($M)": [1.3]})
    assert benchmark_quotes(q, ind)["Verdict"].iloc[0].startswith("Above")


def test_portfolio_sorted():
    df = score_portfolio(pd.read_csv("data/sample_companies.csv"))
    assert df["Risk score"].is_monotonic_decreasing and len(df) == 12
