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

**2. Claim probability.** Each sector starts from a base annual filing rate. The risk score scales it from 0.4x (score 0) to 3.0x (score 100).

**3. Loss simulation.** A Monte Carlo model runs up to 100,000 simulated years. Each year draws whether a claim occurs (binomial) and how large it is (lognormal, with the median scaling with market cap to the 0.45 power, capped at 25% of market cap, since settlements grow slower than company size). The output gives expected annual loss and 1 in 100 and 1 in 250 year losses.

**4. Tower design.** The model recommends a total ABC limit from market cap and risk score, sets a retention, and stacks primary and excess layers plus an optional Side A DIC layer. For each layer it reports how often the layer is hit and its expected loss.

**5. Pricing and benchmarking.** Indicated premium equals rate on line times limit, with each excess layer priced at a set share of the layer below it. Carrier quotes are compared layer by layer to the indication: more than 15% above is flagged to push back, more than 5% below is flagged as best value.

**6. Portfolio screen.** Upload a CSV of companies to rank a whole book by risk score, claim probability and recommended limit.

**Limits.** Sector base rates, severity and pricing parameters are illustrative assumptions, not carrier filed rates. Sample companies are fictional. The tool is built for structuring and negotiation analysis, not binding pricing.
