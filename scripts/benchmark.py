#!/usr/bin/env python3
"""
Foundation Model Benchmark: S&P 500 Daily Directional Prediction

Benchmarks Chronos-2, TimesFM 2.5, TiRex, and Moirai 2.0 (zero-shot)
against the xLSTM-TS paper (arXiv:2408.12408) on the same dataset and metrics.

Usage:
    python scripts/benchmark.py                        # run all models
    python scripts/benchmark.py --models chronos tirex # run specific models
    python scripts/benchmark.py --context-length 200   # custom context window
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
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
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'datasets', 'sp500_daily.csv')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results')

STOCK = 'S&P 500'
TRAIN_END_DATE = '2021-01-01'
VAL_END_DATE = '2022-07-01'

AVAILABLE_MODELS = ['chronos', 'tirex', 'moirai', 'timesfm']  # timesfm requires Python <3.12

# ---------------------------------------------------------------------------
# Metrics (same as paper: src/ml/models/shared/metrics.py +
#          src/ml/models/shared/directional_prediction.py)
# ---------------------------------------------------------------------------
def _naive_forecast(actual, seasonality=1):
    return actual[:-seasonality]

def rmsse(actual, predicted, seasonality=1):
    q = mean_squared_error(actual, predicted) / mean_squared_error(
        actual[seasonality:], _naive_forecast(actual, seasonality))
    return np.sqrt(q)

def mase(actual, predicted, seasonality=1):
    return mean_absolute_error(actual, predicted) / mean_absolute_error(
        actual[seasonality:], _naive_forecast(actual, seasonality))

def evaluate(actual, predicted, model_name):
    """Compute all metrics matching the paper's output format."""
    actual = np.asarray(actual).squeeze()
    predicted = np.asarray(predicted).squeeze()

    # Regression
    metrics = {
        'MAE':   mean_absolute_error(actual, predicted),
        'MSE':   mean_squared_error(actual, predicted),
        'RMSE':  root_mean_squared_error(actual, predicted),
        'RMSSE': rmsse(actual, predicted),
        'MAPE':  mean_absolute_percentage_error(actual, predicted) * 100,
        'MASE':  mase(actual, predicted),
        'R2':    r2_score(actual, predicted),
    }

    # Directional (same logic as paper: np.diff on both series, compare signs)
    actual_dirs = (np.diff(actual) > 0).astype(int)
    pred_dirs   = (np.diff(predicted) > 0).astype(int)

    metrics.update({
        'Test Accuracy':    accuracy_score(actual_dirs, pred_dirs) * 100,
        'Recall':           recall_score(actual_dirs, pred_dirs, pos_label=1) * 100,
        'Precision (Rise)': precision_score(actual_dirs, pred_dirs, pos_label=1) * 100,
        'Precision (Fall)': precision_score(actual_dirs, pred_dirs, pos_label=0) * 100,
        'F1 Score':         f1_score(actual_dirs, pred_dirs, pos_label=1) * 100,
    })

    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')
    pct = {'MAPE', 'Test Accuracy', 'Recall', 'Precision (Rise)', 'Precision (Fall)', 'F1 Score'}
    for k, v in metrics.items():
        print(f'  {k:>20s}: {v:.2f}{"%" if k in pct else ""}')

    return metrics

# ---------------------------------------------------------------------------
# Paper baselines (from their GitHub notebook, exact reproduced numbers)
# ---------------------------------------------------------------------------
PAPER_ORIGINAL = {
    'xLSTM-TS':  {'MAE': 38.25, 'MSE': 2325.60, 'RMSE': 48.22, 'RMSSE': 1.11, 'MAPE': 0.94, 'MASE': 1.16, 'R2': 0.97,
                   'Test Accuracy': 49.47, 'Recall': 53.68, 'Precision (Rise)': 50.00, 'Precision (Fall)': 48.84, 'F1 Score': 51.78,
                   'Validation Accuracy': 49.87, 'Train Accuracy': 48.10},
    'TSMixer':   {'MAE': 90.66, 'MSE': 15021.84, 'RMSE': 122.56, 'RMSSE': 3.38, 'MAPE': 2.15, 'MASE': 3.19, 'R2': 0.75,
                   'Test Accuracy': 46.38, 'Recall': 47.26, 'Precision (Rise)': 49.29, 'Precision (Fall)': 43.38, 'F1 Score': 48.25,
                   'Validation Accuracy': 56.00, 'Train Accuracy': 48.91},
    'N-HiTS':    {'MAE': 45.41, 'MSE': 3404.14, 'RMSE': 58.34, 'RMSSE': 1.61, 'MAPE': 1.08, 'MASE': 1.60, 'R2': 0.94,
                   'Test Accuracy': 50.72, 'Recall': 52.74, 'Precision (Rise)': 53.47, 'Precision (Fall)': 47.73, 'F1 Score': 53.10,
                   'Validation Accuracy': 57.45, 'Train Accuracy': 48.68},
    'TiDE':      {'MAE': 31.43, 'MSE': 1512.92, 'RMSE': 38.90, 'RMSSE': 1.07, 'MAPE': 0.75, 'MASE': 1.11, 'R2': 0.97,
                   'Test Accuracy': 51.45, 'Recall': 56.16, 'Precision (Rise)': 53.95, 'Precision (Fall)': 48.39, 'F1 Score': 55.03,
                   'Validation Accuracy': 53.82, 'Train Accuracy': 53.39},
    'TFT':       {'MAE': 52.56, 'MSE': 4924.10, 'RMSE': 70.17, 'RMSSE': 1.93, 'MAPE': 1.25, 'MASE': 1.85, 'R2': 0.92,
                   'Test Accuracy': 51.09, 'Recall': 52.74, 'Precision (Rise)': 53.85, 'Precision (Fall)': 48.12, 'F1 Score': 53.29,
                   'Validation Accuracy': 59.27, 'Train Accuracy': 48.31},
    'N-BEATS':   {'MAE': 50.18, 'MSE': 4029.71, 'RMSE': 63.48, 'RMSSE': 1.75, 'MAPE': 1.19, 'MASE': 1.77, 'R2': 0.93,
                   'Test Accuracy': 48.91, 'Recall': 48.63, 'Precision (Rise)': 51.82, 'Precision (Fall)': 46.04, 'F1 Score': 50.18,
                   'Validation Accuracy': 57.82, 'Train Accuracy': 50.03},
    'DeepTCN':   {'MAE': 50.08, 'MSE': 3471.25, 'RMSE': 58.92, 'RMSSE': 1.62, 'MAPE': 1.18, 'MASE': 1.76, 'R2': 0.94,
                   'Test Accuracy': 48.55, 'Recall': 51.37, 'Precision (Rise)': 51.37, 'Precision (Fall)': 45.38, 'F1 Score': 51.37,
                   'Validation Accuracy': 50.55, 'Train Accuracy': 47.06},
    'TCN':       {'MAE': 29.03, 'MSE': 1336.56, 'RMSE': 36.56, 'RMSSE': 1.01, 'MAPE': 0.69, 'MASE': 1.02, 'R2': 0.98,
                   'Test Accuracy': 49.64, 'Recall': 53.42, 'Precision (Rise)': 52.35, 'Precision (Fall)': 46.46, 'F1 Score': 52.88,
                   'Validation Accuracy': 49.09, 'Train Accuracy': 47.83},
}

PAPER_DENOISED = {
    'xLSTM-TS':  {'MAE': 55.84, 'MSE': 4600.42, 'RMSE': 67.83, 'RMSSE': 1.56, 'MAPE': 1.37, 'MASE': 1.69, 'R2': 0.94,
                   'Test Accuracy': 66.22, 'Recall': 72.11, 'Precision (Rise)': 64.93, 'Precision (Fall)': 67.88, 'F1 Score': 68.33,
                   'Validation Accuracy': 64.80, 'Train Accuracy': 64.21},
    'TSMixer':   {'MAE': 85.58, 'MSE': 13163.98, 'RMSE': 114.73, 'RMSSE': 3.16, 'MAPE': 2.03, 'MASE': 3.02, 'R2': 0.78,
                   'Test Accuracy': 47.83, 'Recall': 49.32, 'Precision (Rise)': 50.70, 'Precision (Fall)': 44.78, 'F1 Score': 50.00,
                   'Validation Accuracy': 56.36, 'Train Accuracy': 49.80},
    'N-HiTS':    {'MAE': 33.96, 'MSE': 1835.65, 'RMSE': 42.84, 'RMSSE': 1.18, 'MAPE': 0.81, 'MASE': 1.20, 'R2': 0.97,
                   'Test Accuracy': 56.52, 'Recall': 56.85, 'Precision (Rise)': 59.29, 'Precision (Fall)': 53.68, 'F1 Score': 58.04,
                   'Validation Accuracy': 64.36, 'Train Accuracy': 56.38},
    'TiDE':      {'MAE': 23.31, 'MSE': 865.17, 'RMSE': 29.41, 'RMSSE': 0.81, 'MAPE': 0.56, 'MASE': 0.82, 'R2': 0.99,
                   'Test Accuracy': 64.86, 'Recall': 67.81, 'Precision (Rise)': 66.44, 'Precision (Fall)': 62.99, 'F1 Score': 67.12,
                   'Validation Accuracy': 68.36, 'Train Accuracy': 67.55},
    'TFT':       {'MAE': 47.97, 'MSE': 4100.21, 'RMSE': 64.03, 'RMSSE': 1.77, 'MAPE': 1.14, 'MASE': 1.69, 'R2': 0.93,
                   'Test Accuracy': 53.26, 'Recall': 54.79, 'Precision (Rise)': 55.94, 'Precision (Fall)': 50.38, 'F1 Score': 55.36,
                   'Validation Accuracy': 60.36, 'Train Accuracy': 52.02},
    'N-BEATS':   {'MAE': 32.14, 'MSE': 1666.18, 'RMSE': 40.82, 'RMSSE': 1.13, 'MAPE': 0.76, 'MASE': 1.13, 'R2': 0.97,
                   'Test Accuracy': 59.78, 'Recall': 62.33, 'Precision (Rise)': 61.90, 'Precision (Fall)': 57.36, 'F1 Score': 62.12,
                   'Validation Accuracy': 63.64, 'Train Accuracy': 58.56},
    'DeepTCN':   {'MAE': 54.97, 'MSE': 3924.06, 'RMSE': 62.64, 'RMSSE': 1.73, 'MAPE': 1.29, 'MASE': 1.94, 'R2': 0.93,
                   'Test Accuracy': 63.77, 'Recall': 71.23, 'Precision (Rise)': 64.20, 'Precision (Fall)': 63.16, 'F1 Score': 67.53,
                   'Validation Accuracy': 65.45, 'Train Accuracy': 64.65},
    'TCN':       {'MAE': 30.01, 'MSE': 1333.10, 'RMSE': 36.51, 'RMSSE': 1.01, 'MAPE': 0.71, 'MASE': 1.06, 'R2': 0.98,
                   'Test Accuracy': 63.41, 'Recall': 69.86, 'Precision (Rise)': 64.15, 'Precision (Fall)': 62.39, 'F1 Score': 66.89,
                   'Validation Accuracy': 65.82, 'Train Accuracy': 64.23},
}

# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------
def run_chronos(full_close, test_start_idx, test_len, ctx_len, pred_len, device):
    from chronos import ChronosPipeline
    model = ChronosPipeline.from_pretrained(
        'amazon/chronos-t5-base',
        device_map=device,
        dtype=torch.float32,
    )
    preds = []
    for i in tqdm(range(test_len), desc='Chronos-2'):
        ctx = torch.tensor(
            full_close[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        )
        forecast = model.predict(ctx.unsqueeze(0), prediction_length=pred_len)
        # forecast shape: (1, num_samples, pred_len) — take median across samples
        median_val = forecast[:, :, 0].median(dim=1).values.squeeze().item()
        preds.append(median_val)
    return np.array(preds)


def run_timesfm(full_close, test_start_idx, test_len, ctx_len, pred_len, device):
    import timesfm
    backend = 'gpu' if device == 'cuda' else 'cpu'
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=1,
            horizon_len=pred_len,
            input_patch_len=32,
            output_patch_len=128,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id='google/timesfm-2.0-200m-pytorch',
        ),
    )
    preds = []
    for i in tqdm(range(test_len), desc='TimesFM 2.5'):
        ctx = full_close[test_start_idx + i - ctx_len : test_start_idx + i].tolist()
        point_forecast, _ = tfm.forecast([ctx])
        preds.append(point_forecast[0][0])
    return np.array(preds)


def run_tirex(full_close, test_start_idx, test_len, ctx_len, pred_len, device):
    from tirex import load_model
    model = load_model('NX-AI/TiRex')
    preds = []
    for i in tqdm(range(test_len), desc='TiRex'):
        ctx = torch.tensor(
            full_close[test_start_idx + i - ctx_len : test_start_idx + i],
            dtype=torch.float32,
        ).unsqueeze(0)  # shape: (1, ctx_len)
        quantiles, mean = model.forecast(context=ctx, prediction_length=pred_len)
        preds.append(mean.squeeze().item())
    return np.array(preds)


def run_moirai(full_close, test_start_idx, test_len, ctx_len, pred_len, device):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained('Salesforce/moirai-2.0-R-small'),
        prediction_length=pred_len,
        context_length=ctx_len,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    preds = []
    for i in tqdm(range(test_len), desc='Moirai 2.0'):
        ctx_vals = full_close[test_start_idx + i - ctx_len : test_start_idx + i]
        # predict returns (batch, num_quantiles, pred_len) — index 4 = median (0.5)
        forecast = model.predict([ctx_vals])
        preds.append(forecast[0, 4, 0])
    return np.array(preds)


MODEL_RUNNERS = {
    'chronos': ('Chronos-2',    run_chronos),
    'timesfm': ('TimesFM 2.5',  run_timesfm),
    'tirex':   ('TiRex',        run_tirex),
    'moirai':  ('Moirai 2.0',   run_moirai),
}

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def print_table(title, results_dict):
    df = pd.DataFrame(results_dict).T.round(2)
    pct_cols = ['MAPE', 'Test Accuracy', 'Recall', 'Precision (Rise)',
                'Precision (Fall)', 'F1 Score', 'Validation Accuracy', 'Train Accuracy']
    display = df.copy()
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f'{x:.2f}%')
    print(f'\n{"="*80}')
    print(f'  {title}')
    print(f'{"="*80}')
    print(display.to_string())
    return df


def save_chart(fm_df, orig_df, den_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    plot_data = {}
    for name, row in fm_df.iterrows():
        plot_data[f'{name} (zero-shot)'] = {
            'Test Accuracy': row['Test Accuracy'], 'MAE': row['MAE'],
            'group': 'Foundation (zero-shot)',
        }
    for name, row in orig_df.iterrows():
        plot_data[f'{name} (original)'] = {
            'Test Accuracy': row['Test Accuracy'], 'MAE': row['MAE'],
            'group': 'Paper (original data)',
        }
    for name, row in den_df.iterrows():
        plot_data[f'{name} (denoised)'] = {
            'Test Accuracy': row['Test Accuracy'], 'MAE': row['MAE'],
            'group': 'Paper (denoised data)',
        }

    plot_df = pd.DataFrame(plot_data).T.sort_values('Test Accuracy', ascending=True)
    cmap = {
        'Foundation (zero-shot)': '#FF5722',
        'Paper (original data)': '#9E9E9E',
        'Paper (denoised data)': '#2196F3',
    }
    colors = [cmap[plot_df.loc[n, 'group']] for n in plot_df.index]

    axes[0].barh(plot_df.index, plot_df['Test Accuracy'].astype(float), color=colors)
    axes[0].axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Coin flip (50%)')
    axes[0].set_xlabel('Test Accuracy (%)')
    axes[0].set_title('Directional Prediction Accuracy', fontweight='bold')
    axes[0].legend()
    axes[0].tick_params(axis='y', labelsize=9)

    plot_mae = plot_df.sort_values('MAE', ascending=False)
    colors_mae = [cmap[plot_mae.loc[n, 'group']] for n in plot_mae.index]
    axes[1].barh(plot_mae.index, plot_mae['MAE'].astype(float), color=colors_mae)
    axes[1].set_xlabel('MAE (lower is better)')
    axes[1].set_title('Mean Absolute Error', fontweight='bold')
    axes[1].tick_params(axis='y', labelsize=9)

    legend_elements = [
        Patch(facecolor='#FF5722', label='Foundation models (zero-shot)'),
        Patch(facecolor='#9E9E9E', label='Paper baselines (original data)'),
        Patch(facecolor='#2196F3', label='Paper baselines (denoised data)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f'{STOCK} Daily Close - Foundation Models vs xLSTM-TS Paper',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nChart saved to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Foundation model benchmark for S&P 500 daily')
    default_models = ['chronos', 'tirex', 'moirai']
    parser.add_argument('--models', nargs='+', choices=AVAILABLE_MODELS, default=default_models,
                        help='Which models to run (default: chronos tirex moirai). '
                             'timesfm available but requires Python <3.12')
    parser.add_argument('--context-length', type=int, default=150,
                        help='Context window size (default: 150, same as xLSTM-TS)')
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda/mps/cpu). Auto-detected if omitted.')
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
    print(f'Context length: {args.context_length}')
    print(f'Models: {", ".join(args.models)}')

    # Load data
    df = pd.read_csv(DATA_PATH, header=0, index_col='Date')
    df.index = pd.to_datetime(df.index, utc=True).normalize().tz_localize(None)
    close = df['Close'].values

    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_end = pd.Timestamp(VAL_END_DATE)
    train_len = len(df[df.index < train_end])
    val_len = len(df[(df.index >= train_end) & (df.index < val_end)])
    test_start_idx = train_len + val_len
    test_actuals = close[test_start_idx:]
    test_len = len(test_actuals)

    print(f'\nDataset: {len(close)} rows')
    print(f'Test set: {test_len} rows (from index {test_start_idx})')

    # Run models
    fm_results = {}
    for model_key in args.models:
        display_name, runner = MODEL_RUNNERS[model_key]
        print(f'\n{"━"*60}')
        print(f'  Running {display_name}...')
        print(f'{"━"*60}')
        try:
            preds = runner(close, test_start_idx, test_len, args.context_length, 1, device)
            fm_results[display_name] = evaluate(test_actuals, preds, display_name)
        except Exception as e:
            print(f'  FAILED: {e}')
            print(f'  Skipping {display_name}. Check that the package is installed.')

    if not fm_results:
        print('\nNo models ran successfully. Install at least one:')
        print('  pip install chronos-forecasting torch')
        print('  pip install timesfm')
        print('  pip install tirex-forecasting')
        print('  pip install uni2ts einops')
        sys.exit(1)

    # Print all three tables
    fm_df   = print_table('FOUNDATION MODELS (zero-shot, no training, raw data)', fm_results)
    orig_df = print_table('PAPER BASELINES — ORIGINAL DATA (trained, no denoising)', PAPER_ORIGINAL)
    den_df  = print_table('PAPER BASELINES — DENOISED DATA (trained, wavelet denoised)', PAPER_DENOISED)

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, 'foundation_model_results.csv')
    fm_df.to_csv(csv_path)
    print(f'\nResults CSV saved to {csv_path}')

    chart_path = os.path.join(OUTPUT_DIR, 'benchmark_chart.png')
    save_chart(fm_df, orig_df, den_df, chart_path)


if __name__ == '__main__':
    main()
