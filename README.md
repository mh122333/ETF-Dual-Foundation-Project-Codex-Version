# ETF Duel Foundation Model Project — Experiment 0

This repo implements **Experiment 0: pipeline sanity** for the ETF Duel Foundation Model Project.

## What Experiment 0 produces

When you run the Colab notebook, it will:

- Download 30-minute split-adjusted bars from Alpaca (RTH 09:30–16:00 US/Eastern)
- Cache raw bars to Google Drive as Parquet (one file per symbol)
- Compute ATR + triple-barrier labels
- Build minimal baseline features
- Run sanity checks + leakage smoke test
- Save a labeled dataset parquet and summary JSON to Drive

## Google Drive output folder

All artifacts are written under:

```
/content/drive/MyDrive/ETF Duel Foundation Model Project/codex_build/
```

Suggested subfolders are created automatically:

- `data/raw/`
- `data/processed/`
- `runs/`
- `models/`

## Colab run steps (A100)

1. Open `notebooks/01_experiment0.ipynb` in Google Colab.
2. Clone or copy this repo into Google Drive (for example, `/content/drive/MyDrive/ETF Duel Foundation Model Project/codex_build/repo`).
3. Run the notebook top-to-bottom without edits. The notebook auto-detects the repo path, but you can update it in the cell if you store the repo elsewhere.
4. The notebook mounts Google Drive and reads Alpaca keys from Colab Secrets:

   ```python
   from google.colab import userdata
   key = userdata.get("PAPER_KEY")
   sec = userdata.get("PAPER_SEC")
   ```

5. Outputs are written to the Drive path above.

## Default configuration

- Symbols: `SPY`, `QQQ`, `IWM`, `AAPL`, `MSFT`
- Timeframe: 30-minute bars
- Date range: last ~18 months (explicit start/end in notebook)
- ATR window: 14
- Triple-barrier params: `k_up=2.0`, `k_dn=1.0`, `N=26`

## Repo layout

- `src/etf_duel/`: data download, labels, features, sanity checks, and pipeline helpers
- `notebooks/01_experiment0.ipynb`: end-to-end Experiment 0 notebook
