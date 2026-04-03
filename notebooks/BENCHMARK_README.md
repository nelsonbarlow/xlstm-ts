# Foundation Model Benchmark

## What this is

A benchmark comparing modern time series foundation models against the results from the xLSTM-TS paper ([arXiv:2408.12408](https://arxiv.org/abs/2408.12408)) on S&P 500 daily close price directional prediction.

## Why

We replicated the paper's notebook and found:
- **All baseline models (TCN, TiDE, N-BEATS, etc.) reproduce exactly** from their GitHub code
- **xLSTM-TS results vary between runs** (no fixed random seed) -- their GitHub shows ~66% test accuracy, not the 71.28% claimed in the paper
- The best reproducible directional accuracy from ANY model in the paper is ~65-66% on denoised data
- The wavelet denoising is doing most of the heavy lifting (models jump from ~50% to ~64% with denoising)

**Question:** Can modern foundation models beat 65% directional accuracy *zero-shot* (no training on this data)?

## Models tested

| Model | Org | Params | Architecture | Why included |
|-------|-----|--------|-------------|-------------|
| Chronos-2 | Amazon | 120M | Encoder-only Transformer | Current GIFT-Eval benchmark leader |
| TimesFM 2.5 | Google | 200M | Decoder-only Transformer | Strong zero-shot, BigQuery integration |
| TiRex | NX-AI | 35M | xLSTM-based | Direct successor to xLSTM, NeurIPS 2025 |
| Moirai 2.0 | Salesforce | 11M | Decoder-only Transformer | Smallest competitive model |

## How to run

**Hardware:** Mac Mini M4 24GB is more than sufficient. All models fit comfortably.

### Python script (recommended)

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tqdm torch
pip install chronos-forecasting    # Chronos-2
pip install timesfm                # TimesFM 2.5
pip install tirex-forecasting      # TiRex
pip install uni2ts einops          # Moirai 2.0

# Run all models
python scripts/benchmark.py

# Run specific models only
python scripts/benchmark.py --models chronos tirex

# Custom context window
python scripts/benchmark.py --context-length 200

# Force CPU if MPS causes issues
python scripts/benchmark.py --device cpu
```

Results are saved to `results/foundation_model_results.csv` and `results/benchmark_chart.png`.

### Jupyter notebook (alternative)

```bash
cd notebooks
jupyter notebook foundation_model_benchmark.ipynb
```

## Data

Uses the same `data/datasets/sp500_daily.csv` from the original repo. Same date splits:
- Train: 2000-01-01 to 2020-12-31
- Validation: 2021-01-01 to 2022-06-30
- Test: 2022-07-01 to 2023-12-31

## What to look for

- **Test Accuracy > 65%** means the foundation model beats the paper's best reproducible result *without any training*
- **Test Accuracy > 50%** means it's better than a coin flip
- Regression metrics (MAE, RMSE) show price prediction quality
- The paper baselines used denoised data + full training; foundation models here use raw data + zero-shot

## Known issues

- Some model APIs may change -- check the respective GitHub repos for latest usage if imports fail
- Moirai 2.0 may need `gluonts` as an additional dependency
- TiRex import path may differ depending on version -- check https://huggingface.co/NX-AI/TiRex
- MPS (Apple Silicon) support varies by model -- fall back to CPU if you get MPS errors

## Context for Claude Code

If you're picking this up in a new session: the user is investigating whether pre-trained time series foundation models can match or beat a custom xLSTM-TS model that was trained specifically on this S&P 500 data. The key metric is **directional prediction accuracy** (predicting whether tomorrow's close is higher or lower than today's). The bar to beat is ~65% from the paper's own reproducible results. The notebook may need API adjustments as foundation model libraries evolve quickly.
