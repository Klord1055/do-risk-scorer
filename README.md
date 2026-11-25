# D&O Program Risk Scorer

A Python and Streamlit app that scores a public company's directors and officers (D&O) liability exposure, simulates annual losses with Monte Carlo, designs a layered insurance tower, and benchmarks carrier quotes against a modeled price.

**Live demo:** https://klord-do-risk-scorer.streamlit.app

## Why I built it

During my summer 2026 internship on WTW's FINEX Commercial Lines team, I rebuilt multilayer D&O towers, benchmarked premium and retention across carriers, and analyzed renewal data across a large book of corporate clients. Most of that work lived in spreadsheets and was rebuilt account by account. This tool turns the same thinking into one repeatable model: score the risk, size the program, price each layer, and show the client where a quote is out of line.

No client or WTW data is used. Every sample company is fictional and every pricing parameter is an editable assumption.

## What it does

| Tab | Output |
|---|---|
| Risk score | 0 to 100 exposure score, tier, claim probability, and a breakdown of which factors drive it |
| Loss model and tower | Monte Carlo loss distribution, 1 in 100 and 1 in 250 year losses, recommended tower with hit probability and expected loss per layer |
| Quote benchmark | Editable carrier quotes compared layer by layer to the indicated premium, with push back and best value flags |
| Portfolio screen | Upload a CSV to rank a whole book of companies, then download the results |
| Methodology | How every number is calculated |

## Tech

Python, pandas, NumPy, Plotly, Streamlit, pytest. The model lives in `risk/model.py` as pure, tested functions, separate from the interface in `app.py`.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

Built by Klord Fernandez Cayetano.
