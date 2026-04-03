#!/usr/bin/env python3
"""
Fine-tune Chronos-2 on wavelet-denoised S&P 500 data, then evaluate
on original test data — same protocol as the xLSTM-TS paper.

Usage:
    python scripts/finetune_chronos.py
    python scripts/finetune_chronos.py --num-steps 2000 --learning-rate 1e-5
    python scripts/finetune_chronos.py --device cpu
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import pywt
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, root_mean_squared_error,
    mean_absolute_percentage_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'datasets', 'sp500_daily.csv')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

STOCK = 'S&P 500'
TRAIN_END_DATE = '2021-01-01'
VAL_END_DATE = '2022-07-01'

# ---------------------------------------------------------------------------
# Wavelet denoising (same as paper: src/ml/data/preprocessing.py)
# ---------------------------------------------------------------------------
def wavelet_denoising(data, wavelet='db4', level=1):
    padded_data = np.pad(data, 100, mode='edge')
    coeff = pywt.wavedec(padded_data, wavelet, mode='per', level=level)
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(padded_data)))
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
    coeff[-level] = np.zeros_like(coeff[-level])
    denoised = pywt.waverec(coeff, wavelet, mode='per')
    denoised = denoised[100:-100]
    if len(denoised) > len(data):
        denoised = denoised[:len(data)]
    elif len(denoised) < len(data):
        denoised = np.pad(denoised, (0, len(data) - len(denoised)), 'edge')
    return denoised

# ---------------------------------------------------------------------------
# Metrics (same as paper)
# ---------------------------------------------------------------------------
def _naive_forecast(actual, seasonality=1):
    return actual[:-seasonality]

def evaluate(actual, predicted, model_name):
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()

    metrics = {
        'MAE':   mean_absolute_error(actual, predicted),
        'MSE':   mean_squared_error(actual, predicted),
        'RMSE':  root_mean_squared_error(actual, predicted),
        'RMSSE': np.sqrt(mean_squared_error(actual, predicted) /
                         mean_squared_error(actual[1:], _naive_forecast(actual))),
        'MAPE':  mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE':  mean_absolute_error(actual, predicted) /
                 mean_absolute_error(actual[1:], _naive_forecast(actual)),
        'R2':    r2_score(actual, predicted),
    }

    actual_dirs = (np.diff(actual) > 0).astype(int)
    pred_dirs = (np.diff(predicted) > 0).astype(int)
    metrics.update({
        'Test Accuracy':    accuracy_score(actual_dirs, pred_dirs) * 100,
        'Recall':           recall_score(actual_dirs, pred_dirs, pos_label=1) * 100,
        'Precision (Rise)': precision_score(actual_dirs, pred_dirs, pos_label=1) * 100,
        'Precision (Fall)': precision_score(actual_dirs, pred_dirs, pos_label=0) * 100,
        'F1 Score':         f1_score(actual_dirs, pred_dirs, pos_label=1) * 100,
    })

    pct = {'MAPE', 'Test Accuracy', 'Recall', 'Precision (Rise)', 'Precision (Fall)', 'F1 Score'}
    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    for k, v in metrics.items():
        print(f'  {k:>20s}: {v:.2f}{"%" if k in pct else ""}')
    return metrics

# ---------------------------------------------------------------------------
# Rolling prediction
# ---------------------------------------------------------------------------
def rolling_predict(pipeline, full_close, test_start_idx, test_len, ctx_len, desc):
    preds = []
    for i in tqdm(range(test_len), desc=desc):
        ctx = torch.tensor(
            full_close[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        )
        forecasts = pipeline.predict(ctx.unsqueeze(0), prediction_length=1)
        # Chronos-2 predict returns list[Tensor] of shape (n_variates, n_quantiles, pred_len)
        # Middle quantile index = median
        f = forecasts[0]
        median_idx = f.shape[1] // 2
        preds.append(f[0, median_idx, 0].item())
    return np.array(preds)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Chronos-2 on denoised S&P 500')
    parser.add_argument('--model', type=str, default='amazon/chronos-2',
                        help='Chronos-2 model ID (default: amazon/chronos-2)')
    parser.add_argument('--context-length', type=int, default=150,
                        help='Context window (default: 150, same as xLSTM-TS)')
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Fine-tuning steps (default: 1000)')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--finetune-mode', type=str, default='full', choices=['full', 'lora'],
                        help='Fine-tuning mode (default: full)')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda/mps/cpu)')
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f'Device: {device}')
    print(f'Model: {args.model}')
    print(f'Fine-tune mode: {args.finetune_mode}')
    print(f'Steps: {args.num_steps}, LR: {args.learning_rate}, Batch: {args.batch_size}')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH, header=0, index_col='Date')
    df.index = pd.to_datetime(df.index, utc=True).normalize().tz_localize(None)
    close_raw = df['Close'].values.astype(np.float32)

    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_end = pd.Timestamp(VAL_END_DATE)
    train_mask = df.index < train_end
    val_mask = (df.index >= train_end) & (df.index < val_end)

    train_len = train_mask.sum()
    val_len = val_mask.sum()
    test_start_idx = train_len + val_len
    test_actuals = close_raw[test_start_idx:]
    test_len = len(test_actuals)

    print(f'\nDataset: {len(close_raw)} rows')
    print(f'Train: {train_len}, Val: {val_len}, Test: {test_len}')

    # ------------------------------------------------------------------
    # Apply wavelet denoising to train+val (same as paper)
    # ------------------------------------------------------------------
    print('\nApplying wavelet denoising...')
    train_val_raw = close_raw[:test_start_idx]
    train_val_denoised = wavelet_denoising(train_val_raw).astype(np.float32)

    # Build the full denoised series (denoised train+val, original test)
    close_denoised = np.concatenate([train_val_denoised, close_raw[test_start_idx:]])

    print(f'  Raw train+val range: [{train_val_raw.min():.1f}, {train_val_raw.max():.1f}]')
    print(f'  Denoised range:      [{train_val_denoised.min():.1f}, {train_val_denoised.max():.1f}]')

    # ------------------------------------------------------------------
    # Load Chronos-2 base model
    # ------------------------------------------------------------------
    from chronos import Chronos2Pipeline

    print(f'\nLoading {args.model}...')
    base_pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Evaluate base model (zero-shot on raw data) for comparison
    # ------------------------------------------------------------------
    print('\n--- Zero-shot baseline (raw data) ---')
    base_preds = rolling_predict(base_pipeline, close_raw, test_start_idx,
                                 test_len, args.context_length, 'Chronos-2 (zero-shot)')
    base_metrics = evaluate(test_actuals, base_preds, 'Chronos-2 (zero-shot, raw)')

    # ------------------------------------------------------------------
    # Fine-tune on denoised training data
    # ------------------------------------------------------------------
    # Prepare training data: the denoised training series
    train_denoised = train_val_denoised[:train_len]
    val_denoised = train_val_denoised[train_len:train_len + val_len]

    print(f'\n{"━"*60}')
    print(f'  Fine-tuning on denoised data ({args.num_steps} steps)...')
    print(f'{"━"*60}')

    ft_kwargs = dict(
        inputs=[train_denoised],
        validation_inputs=[val_denoised],
        prediction_length=1,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        finetune_mode=args.finetune_mode,
        output_dir=os.path.join(OUTPUT_DIR, 'chronos2-finetuned'),
    )

    finetuned_pipeline = base_pipeline.fit(**ft_kwargs)

    # ------------------------------------------------------------------
    # Evaluate fine-tuned model on original test data
    # (same as paper: train on denoised, test on original)
    # ------------------------------------------------------------------
    # Use denoised context for test predictions (same as paper approach)
    print('\n--- Fine-tuned (denoised context -> original test) ---')
    ft_preds = rolling_predict(finetuned_pipeline, close_denoised, test_start_idx,
                               test_len, args.context_length, 'Chronos-2 (fine-tuned)')
    ft_metrics = evaluate(test_actuals, ft_preds, 'Chronos-2 (fine-tuned, denoised)')

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    paper_best = {
        'xLSTM-TS (denoised)': {'MAE': 55.84, 'Test Accuracy': 66.22, 'F1 Score': 68.33},
        'TiDE (denoised)':     {'MAE': 23.31, 'Test Accuracy': 64.86, 'F1 Score': 67.12},
        'TCN (denoised)':      {'MAE': 30.01, 'Test Accuracy': 63.41, 'F1 Score': 66.89},
    }

    print(f'\n{"="*70}')
    print(f'  COMPARISON SUMMARY')
    print(f'{"="*70}')
    print(f'  {"Model":<35s} {"MAE":>8s} {"Accuracy":>10s} {"F1":>8s}')
    print(f'  {"─"*35} {"─"*8} {"─"*10} {"─"*8}')
    print(f'  {"Chronos-2 (zero-shot, raw)":<35s} {base_metrics["MAE"]:>8.2f} {base_metrics["Test Accuracy"]:>9.2f}% {base_metrics["F1 Score"]:>7.2f}%')
    print(f'  {"Chronos-2 (fine-tuned, denoised)":<35s} {ft_metrics["MAE"]:>8.2f} {ft_metrics["Test Accuracy"]:>9.2f}% {ft_metrics["F1 Score"]:>7.2f}%')
    for name, m in paper_best.items():
        print(f'  {name:<35s} {m["MAE"]:>8.2f} {m["Test Accuracy"]:>9.2f}% {m["F1 Score"]:>7.2f}%')

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        'Chronos-2 (zero-shot)': base_metrics,
        'Chronos-2 (fine-tuned)': ft_metrics,
    }
    results_df = pd.DataFrame(results).T.round(2)
    csv_path = os.path.join(OUTPUT_DIR, 'finetune_results.csv')
    results_df.to_csv(csv_path)
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
