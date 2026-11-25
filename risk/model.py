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
    # Calibrate the first claim chance so the chance of at least one claim (first or a rarer second) equals p:
    # 1 - (1 - p1)(1 - 0.15 p1) = p  ->  0.15 p1^2 - 1.15 p1 + p = 0
    p = (1.15 - np.sqrt(1.15 ** 2 - 4 * 0.15 * p)) / (2 * 0.15)
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


def build_tower(c: Company, score: float, layer_size: float = 10.0, side_a_limit: float = 10.0,
                total: float | None = None) -> Tower:
    total = recommend_limit(c, score) if total is None else float(total)
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


def layer_metrics(tower: Tower, losses: np.ndarray, target_loss_ratio: float = 0.60,
                  min_rol: float = 0.01) -> pd.DataFrame:
    """Expected loss, attachment probability, and risk based indicated premium for each layer.

    Premium = expected loss / target loss ratio, with a minimum rate on line so remote excess
    layers still carry a market minimum. Riskier companies therefore pay more for the same layer.
    """
    rows = []
    for l in tower.layers:
        if l.side_a:
            # Side A responds when the company cannot indemnify; modeled as 15% of losses above the ABC tower.
            excess = np.clip(losses - l.attachment, 0, l.limit) * 0.15
        else:
            excess = np.clip(losses - l.attachment, 0, l.limit)
        el = float(excess.mean())
        premium = max(el / target_loss_ratio, min_rol * l.limit)
        layer_rol = premium / l.limit
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
    """Compare carrier quotes to the model's indicated premium for the same layer.
    15% or more above indication: push back. 5% or more below: best value."""
    m = quotes.merge(indicated[["Layer", "Indicated premium ($M)", "Expected loss ($M)"]], on="Layer", how="left")
    q = pd.to_numeric(m["Quoted premium ($M)"], errors="coerce")
    m["Quote vs indicated"] = q / m["Indicated premium ($M)"] - 1
    m["Quote loss ratio"] = m["Expected loss ($M)"] / q
    m["Verdict"] = np.select(
        [m["Indicated premium ($M)"].isna(), q.isna() | (q <= 0), m["Quote vs indicated"] <= -0.05, m["Quote vs indicated"] >= 0.15],
        ["No matching layer", "Check quote", "Below indication, best value", "Above indication, push back"], "In line")
    return m


def best_quotes(bench: pd.DataFrame) -> pd.DataFrame:
    """Lowest valid quote per layer, so a layer quoted by several carriers is counted once."""
    ok = bench[~bench["Verdict"].isin(["No matching layer", "Check quote"])]
    return ok.loc[ok.groupby("Layer")["Quoted premium ($M)"].idxmin()] if len(ok) else ok


_TRUE, _FALSE = {"true", "yes", "y", "1", "1.0", "t"}, {"false", "no", "n", "0", "0.0", "f", ""}


def _flag(v, col: str) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    t = "" if pd.isna(v) else str(v).strip().lower()
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    raise ValueError(f"{col} should be yes or no, got {v!r}")


def _num(v, col: str, lo: float = 0.0) -> float:
    x = float(v)
    if not np.isfinite(x) or x < lo:
        raise ValueError(f"{col} should be a number of at least {lo:g}, got {v!r}")
    return x


def company_from_row(r: pd.Series) -> Company:
    """Build a Company from a CSV row, rejecting blanks, text in number columns and impossible values."""
    return Company(
        name=str(r["name"]), sector=str(r["sector"]), market_cap_m=_num(r["market_cap_m"], "market_cap_m", 1.0),
        revenue_m=_num(r["revenue_m"], "revenue_m"), volatility=_num(r["volatility"], "volatility"),
        drawdown=_num(r["drawdown"], "drawdown"), debt_to_equity=_num(r["debt_to_equity"], "debt_to_equity"),
        years_public=_num(r["years_public"], "years_public"),
        spac=_flag(r["spac"], "spac"), major_ma=_flag(r["major_ma"], "major_ma"),
        restatement=_flag(r["restatement"], "restatement"), prior_claims=int(_num(r["prior_claims"], "prior_claims")),
    )


PORTFOLIO_COLUMNS = ["Company", "Sector", "Market cap ($M)", "Risk score", "Tier", "Claim probability",
                     "Benchmark limit ($M)", "Retention ($M)"]


def score_portfolio(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Score every valid row. Rows that cannot be read are skipped and listed in .attrs["skipped"]."""
    out, skipped = [], []
    for i, r in df.iterrows():
        try:
            c = company_from_row(r)
        except (ValueError, TypeError, OverflowError, KeyError) as e:
            skipped.append(f"row {i + 2} ({r.get('name', '?')}): {e}")
            continue
        s, _ = risk_score(c, weights)
        s = round(s, 1)
        out.append({
            "Company": c.name, "Sector": c.sector, "Market cap ($M)": c.market_cap_m,
            "Risk score": s, "Tier": risk_tier(s),
            "Claim probability": claim_frequency(c, s),
            "Benchmark limit ($M)": recommend_limit(c, s),
            "Retention ($M)": recommend_retention(c),
        })
    res = pd.DataFrame(out, columns=PORTFOLIO_COLUMNS).sort_values("Risk score", ascending=False).reset_index(drop=True)
    res.attrs["skipped"] = skipped
    return res


def improvement_paths(c: Company, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """What a company could change to lower its score, one fix at a time, ranked by points saved.
    Only factors management can influence are tested."""
    from dataclasses import replace as _r
    base, _ = risk_score(c, weights)
    fixes = []
    if c.restatement:
        fixes.append(("Close out the restatement or SEC inquiry", _r(c, restatement=False)))
    if c.debt_to_equity > 1.0:
        fixes.append(("Cut debt to 1.0x equity", _r(c, debt_to_equity=1.0)))
    if c.drawdown > 0.15:
        fixes.append(("Stock recovers to within 15% of its high", _r(c, drawdown=0.15)))
    if c.volatility > 0.35:
        fixes.append(("Volatility falls to 35%", _r(c, volatility=0.35)))
    if c.major_ma:
        fixes.append(("Finish integrating the recent deal (no major M&A)", _r(c, major_ma=False)))
    rows = []
    for label, alt in fixes:
        s, _ = risk_score(alt, weights)
        rows.append(dict(Change=label, **{"New score": s, "Points saved": base - s,
                                          "New tier": risk_tier(s), "New claim probability": claim_frequency(alt, s)}))
    df = pd.DataFrame(rows, columns=["Change", "New score", "Points saved", "New tier", "New claim probability"])
    return df.sort_values("Points saved", ascending=False).reset_index(drop=True)


def recommended_tower_limit(c: Company, score: float, losses: np.ndarray, return_period: float = 100) -> float:
    """Tower limit that covers a 1 in N year loss, and never less than the size benchmark. Capped at $500M."""
    need = limit_needed(losses, recommend_retention(c), return_period)
    return float(min(500.0, max(recommend_limit(c, score), need)))


def limit_needed(losses: np.ndarray, retention: float, return_period: float) -> float:
    """Tower limit needed above the retention so a 1 in N year loss is fully covered, rounded up to $5M."""
    q = float(np.percentile(losses, 100 * (1 - 1 / return_period)))
    need = max(0.0, q - retention)
    return float(5 * np.ceil(need / 5))


def exhaustion_probability(losses: np.ndarray, tower: "Tower") -> float:
    """Chance a year's loss blows through the retention plus every ABC layer."""
    return float((losses > tower.retention + tower.total_limit).mean())
