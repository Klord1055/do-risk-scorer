## How it works

**1. Risk score (0 to 100).** Eight drivers of securities class action risk are each scored 0 to 100, then weighted:

| Driver | Why it matters | Default weight |
|---|---|---|
| Stock volatility | Volatile stocks produce the price drops that trigger suits | 18% |
| Price drawdown | Most securities claims follow a sharp stock decline | 18% |
| Leverage | Financial stress raises disclosure and insolvency risk | 12% |
| Sector | Biotech and tech are sued far more often than utilities | 12% |
| Recent IPO or SPAC | Newly public companies face Section 11 and de-SPAC claims | 12% |
| M&A activity | Merger objection suits and deal disclosure claims | 8% |
| Restatement or SEC inquiry | Strong predictor of a follow-on securities suit | 12% |
| Prior securities claims | Past claims signal repeat exposure | 8% |

Weights are editable in the sidebar.

**2. Claim probability.** Each sector starts from a base annual filing rate. The risk score scales it from 0.4x (score 0) to 3.0x (score 100), capped at 60%. This is the chance of at least one claim in the year.

**3. Loss simulation.** A Monte Carlo model runs up to 100,000 simulated years. Each year draws whether a claim occurs, with a smaller chance of a second claim, calibrated so the chance of at least one claim matches the claim probability above. Each claim's size is lognormal, with the median scaling with market cap to the 0.45 power, since settlements grow slower than company size. A year's total loss is capped at 25% of market cap. The output gives expected annual loss and 1 in 100 and 1 in 250 year losses.

**4. Tower design.** The recommended total limit is the larger of two numbers: the limit needed to cover a 1 in 100 year loss above the retention, and a size benchmark from market cap and risk score. It is capped at $500M and you can type any other limit to test it. The retention steps up with market cap. Primary and excess layers are stacked on top, plus an optional Side A DIC layer, which is modeled as paying 15% of losses above the ABC tower (the share where the company cannot indemnify its directors). For each layer the app reports how often it is hit and its expected loss. Exhaustion probability is the share of simulated years where the loss exceeds the retention plus every primary and excess layer.

**5. Pricing and benchmarking.** Indicated premium is risk based: each layer's expected loss divided by a target loss ratio (60% by default), with a minimum rate on line (1% of limit) so remote excess layers still carry a market minimum. A riskier company therefore pays more for the same layer. Carrier quotes are compared layer by layer: 15% or more above indication is flagged to push back, 5% or more below is flagged as best value, and a quote with no matching layer or no premium is flagged to check. When several carriers quote one layer, the summary uses the lowest quote.

**6. How to lower the risk.** Each controllable red flag is fixed one at a time (close the restatement or SEC inquiry, cut debt to 1.0x equity, stock recovers to within 15% of its high, volatility falls to 35%, finish integrating the deal) and the score, tier and claim probability are recomputed. Fixes are ranked by points saved.

**7. Portfolio screen.** Upload a CSV of companies to rank a whole book by risk score, claim probability and benchmark limit. Yes or no columns accept true, false, yes, no, y, n, 1 or 0. Rows with blanks, text in number columns or negative values are skipped and listed.

Weights are normalized to sum to 100%, and factor scores stop rising past 80% volatility, a 60% drop and 3.0x debt to equity.

**Limits.** Sector base rates, severity and pricing parameters are illustrative assumptions, not carrier filed rates. Sample companies are fictional. The tool is built for structuring and negotiation analysis, not binding pricing.

