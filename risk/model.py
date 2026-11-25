"""Core D&O risk model: exposure scoring, loss simulation, and tower layering.

All functions are pure so they can be tested and reused outside the Streamlit app.
Sector frequencies and pricing parameters are illustrative assumptions, editable in the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Illustrative annual securities class action filing probabilities by sector.
SECTOR_BASE_FREQ = {
    "Technology": 0.055,
    "Biotech / Pharma": 0.075,
    "Financials": 0.040,
    "Consumer": 0.035,
    "Industrials": 0.025,
    "Energy": 0.030,
    "Healthcare Services": 0.040,
    "Real Estate": 0.020,
    "Utilities": 0.015,
}

# Factor weights (sum to 1). Each factor is scored 0 to 100 before weighting.
DEFAULT_WEIGHTS = {
    "Stock volatility": 0.18,
    "Price drawdown": 0.18,
    "Leverage": 0.12,
    "Sector": 0.12,
    "Recent IPO / SPAC": 0.12,
    "M&A activity": 0.08,
    "Restatement / SEC inquiry": 0.12,
    "Prior securities claims": 0.08,
}


@dataclass
class Company:
    name: str
    sector: str
    market_cap_m: float  # USD millions
    revenue_m: float
    volatility: float  # annualized stock volatility, e.g. 0.45
    drawdown: float  # 52 week peak to current decline, e.g. 0.30
    debt_to_equity: float
    years_public: float
    spac: bool = False
    major_ma: bool = False
    restatement: bool = False
    prior_claims: int = 0


def _clip(x: float) -> float:
    return float(np.clip(x, 0, 100))


def factor_scores(c: Company) -> dict[str, float]:
    """Score each risk driver from 0 (low) to 100 (high)."""
    freqs = SECTOR_BASE_FREQ.values()
    lo, hi = min(freqs), max(freqs)
    sector_score = 100 * (SECTOR_BASE_FREQ.get(c.sector, 0.035) - lo) / (hi - lo)
    ipo = 100 if c.spac else (90 if c.years_public < 1 else 70 if c.years_public < 3 else 20 if c.years_public < 5 else 0)
    return {
        "Stock volatility": _clip((c.volatility - 0.15) / (0.80 - 0.15) * 100),
        "Price drawdown": _clip(c.drawdown / 0.60 * 100),
        "Leverage": _clip(c.debt_to_equity / 3.0 * 100),
        "Sector": _clip(sector_score),
        "Recent IPO / SPAC": float(ipo),
        "M&A activity": 100.0 if c.major_ma else 0.0,
        "Restatement / SEC inquiry": 100.0 if c.restatement else 0.0,
        "Prior securities claims": _clip(c.prior_claims * 50),
    }


def risk_score(c: Company, weights: dict[str, float] | None = None) -> tuple[float, pd.DataFrame]:
    """Weighted 0 to 100 exposure score plus a factor breakdown table."""
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.values()) or 1.0
    scores = factor_scores(c)
    rows = [
        {"Factor": k, "Score": scores[k], "Weight": w[k] / total_w, "Contribution": scores[k] * w[k] / total_w}
        for k in w
    ]
    df = pd.DataFrame(rows).sort_values("Contribution", ascending=False).reset_index(drop=True)
    return float(df["Contribution"].sum()), df


def risk_tier(score: float) -> str:
    if score >= 65:
        return "High"
    if score >= 40:
        return "Elevated"
    if score >= 20:
        return "Moderate"
    return "Low"


def claim_frequency(c: Company, score: float) -> float:
    """Annual probability of a securities claim, scaled from the sector base by the risk score."""
    base = SECTOR_BASE_FREQ.get(c.sector, 0.035)
    multiplier = 0.4 + 2.6 * (score / 100)  # score 0 -> 0.4x, score 100 -> 3.0x
    return float(min(base * multiplier, 0.6))


def severity_params(c: Company) -> tuple[float, float]:
    """Lognormal (mu, sigma) for claim severity in USD millions.

    Median severity scales with market cap: larger issuers face larger investor losses,
    but settlements grow less than proportionally.
    """
    median = max(1.5, 0.6 * c.market_cap_m ** 0.45)
    sigma = 1.0
    return float(np.log(median)), sigma


def simulate_losses(c: Company, score: float, n: int = 50_000, seed: int = 7) -> np.ndarray:
    """Annual aggregate D&O loss (defense plus settlement), USD millions, via Monte Carlo."""
    rng = np.random.default_rng(seed)
    p = claim_frequency(c, score)
    mu, sigma = severity_params(c)
    claims = rng.binomial(1, p, n) + rng.binomial(1, p * 0.15, n)  # rare second claim
    losses = np.zeros(n)
    for k in (1, 2):
        idx = claims >= k
        losses[idx] += rng.lognormal(mu, sigma, idx.sum())
    # A single year's D&O loss rarely exceeds a fraction of the company's value; cap the tail.
    return np.minimum(losses, 0.25 * c.market_cap_m)


@dataclass
class Layer:
    name: str
    attachment: float  # USD millions
    limit: float  # USD millions
    side_a: bool = False


@dataclass
class Tower:
    retention: float
    layers: list[Layer] = field(default_factory=list)

    @property
    def total_limit(self) -> float:
        return sum(l.limit for l in self.layers if not l.side_a)


def recommend_limit(c: Company, score: float) -> float:
    """Total ABC limit recommendation in USD millions, rounded to a marketable $5M."""
    base = 10 + 6.5 * (c.market_cap_m / 1000) ** 0.6
    adj = base * (0.8 + 0.6 * score / 100)
    return float(max(10, min(300, 5 * round(adj / 5))))


def recommend_retention(c: Company) -> float:
    r = 0.25 + 1.5 * (c.market_cap_m / 1000) ** 0.5
    for step in (0.25, 0.5, 1, 2.5, 5, 10, 15, 25):
        if r <= step:
            return float(step)
    return 25.0


def build_tower(c: Company, score: float, layer_size: float = 10.0, side_a_limit: float = 10.0) -> Tower:
    total = recommend_limit(c, score)
    retention = recommend_retention(c)
    layers, att, i = [], retention, 0
    while att - retention < total - 1e-9:
        lim = min(layer_size, total - (att - retention))
        layers.append(Layer("Primary" if i == 0 else f"Excess {i}", att, lim))
        att += lim
        i += 1
    if side_a_limit > 0:
        layers.append(Layer("Side A DIC", att, side_a_limit, side_a=True))
    return Tower(retention, layers)


def layer_metrics(tower: Tower, losses: np.ndarray, primary_rol: float, ilf_decay: float, side_a_rol: float) -> pd.DataFrame:
    """Expected loss, attachment probability, and indicated premium for each layer.

    Premium uses a rate on line that decays by `ilf_decay` per layer moving up the tower,
    the way excess pricing falls with attachment.
    """
    rows = []
    rol = primary_rol
    for i, l in enumerate(tower.layers):
        if l.side_a:
            # Side A responds when the company cannot indemnify; modeled as 15% of losses above the ABC tower.
            excess = np.clip(losses - l.attachment, 0, l.limit) * 0.15
            layer_rol = side_a_rol
        else:
            excess = np.clip(losses - l.attachment, 0, l.limit)
            layer_rol = rol
            rol *= ilf_decay
        el = float(excess.mean())
        premium = layer_rol * l.limit
        rows.append({
            "Layer": l.name,
            "Attaches at ($M)": l.attachment,
            "Limit ($M)": l.limit,
            "Prob. layer is hit": float((excess > 0).mean()),
            "Expected loss ($M)": el,
            "Rate on line": layer_rol,
            "Indicated premium ($M)": premium,
            "Loss ratio": el / premium if premium else np.nan,
        })
    return pd.DataFrame(rows)


def benchmark_quotes(quotes: pd.DataFrame, indicated: pd.DataFrame) -> pd.DataFrame:
    """Compare carrier quotes to the model's indicated premium for the same layer."""
    m = quotes.merge(indicated[["Layer", "Indicated premium ($M)", "Expected loss ($M)"]], on="Layer", how="left")
    m["Quote vs indicated"] = m["Quoted premium ($M)"] / m["Indicated premium ($M)"] - 1
    m["Quote loss ratio"] = m["Expected loss ($M)"] / m["Quoted premium ($M)"]
    m["Verdict"] = np.where(m["Quote vs indicated"] <= -0.05, "Below indication, best value",
                    np.where(m["Quote vs indicated"] >= 0.15, "Above indication, push back", "In line"))
    return m


def company_from_row(r: pd.Series) -> Company:
    return Company(
        name=str(r["name"]), sector=str(r["sector"]), market_cap_m=float(r["market_cap_m"]),
        revenue_m=float(r["revenue_m"]), volatility=float(r["volatility"]), drawdown=float(r["drawdown"]),
        debt_to_equity=float(r["debt_to_equity"]), years_public=float(r["years_public"]),
        spac=bool(r["spac"]), major_ma=bool(r["major_ma"]), restatement=bool(r["restatement"]),
        prior_claims=int(r["prior_claims"]),
    )


def score_portfolio(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        c = company_from_row(r)
        s, _ = risk_score(c, weights)
        out.append({
            "Company": c.name, "Sector": c.sector, "Market cap ($M)": c.market_cap_m,
            "Risk score": round(s, 1), "Tier": risk_tier(s),
            "Claim probability": claim_frequency(c, s),
            "Recommended limit ($M)": recommend_limit(c, s),
            "Retention ($M)": recommend_retention(c),
        })
    return pd.DataFrame(out).sort_values("Risk score", ascending=False).reset_index(drop=True)
