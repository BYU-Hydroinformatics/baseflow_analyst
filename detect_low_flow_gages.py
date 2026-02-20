"""
Detect USGS stream gages with sustained low-flow periods.

A gage qualifies if it has at least one consecutive run of days where:
  - Flow stays below the Nth percentile of its own record
  - The run lasts at least `min_duration` days
  - Flow is relatively constant during the run (CV < cv_threshold)

Usage:
    python detect_low_flow_gages.py [--pct 10] [--cv 0.15] [--dur 60]
"""
import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMFLOW_DIR = os.path.join(BASE_DIR, "usgs_daily_streamflow")


def detect_low_flow(filepath, pct_threshold, cv_threshold, min_duration):
    """Analyze a single gage's streamflow for sustained low-flow periods."""
    site_no = os.path.basename(filepath).replace('.csv', '')
    result = {
        'site_no': site_no,
        'has_lowflow': False,
        'max_lowflow_duration': 0,
        'max_run_cv': np.nan,
        'threshold_cfs': np.nan,
        'record_days': 0,
    }
    try:
        df = pd.read_csv(filepath, usecols=['date', 'streamflow'])
        df['streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce')
        q = df['streamflow'].dropna().values
        result['record_days'] = len(q)

        if len(q) < 365:
            return result

        threshold = np.percentile(q, pct_threshold)
        result['threshold_cfs'] = threshold
        if threshold <= 0:
            return result

        # Find consecutive runs below threshold
        below = q < threshold
        best_duration = 0
        best_cv = np.nan

        run_start = None
        for i in range(len(below)):
            if below[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_len = i - run_start
                    if run_len >= min_duration:
                        run_vals = q[run_start:i]
                        cv = run_vals.std() / run_vals.mean() if run_vals.mean() > 0 else np.inf
                        if cv < cv_threshold and run_len > best_duration:
                            best_duration = run_len
                            best_cv = cv
                    run_start = None
        # Check final run
        if run_start is not None:
            run_len = len(below) - run_start
            if run_len >= min_duration:
                run_vals = q[run_start:]
                cv = run_vals.std() / run_vals.mean() if run_vals.mean() > 0 else np.inf
                if cv < cv_threshold and run_len > best_duration:
                    best_duration = run_len
                    best_cv = cv

        if best_duration >= min_duration:
            result['has_lowflow'] = True
            result['max_lowflow_duration'] = best_duration
            result['max_run_cv'] = round(best_cv, 4)

    except Exception:
        pass

    return result


def _worker(args):
    return detect_low_flow(*args)


def main():
    parser = argparse.ArgumentParser(description="Detect gages with sustained low-flow periods")
    parser.add_argument('--pct', type=float, default=10, help='Percentile threshold (default: 10)')
    parser.add_argument('--cv', type=float, default=0.15, help='Max CV for constant flow (default: 0.15)')
    parser.add_argument('--dur', type=int, default=60, help='Min run duration in days (default: 60)')
    args = parser.parse_args()

    # Collect all streamflow CSV files (exclude site_info.csv)
    files = []
    for f in sorted(os.listdir(STREAMFLOW_DIR)):
        if f.endswith('.csv') and f != 'site_info.csv':
            files.append(os.path.join(STREAMFLOW_DIR, f))

    print(f"Analyzing {len(files)} gages (pct={args.pct}, cv={args.cv}, dur={args.dur})...")

    worker_args = [(f, args.pct, args.cv, args.dur) for f in files]
    n_workers = max(1, cpu_count() - 1)

    results = []
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, worker_args, chunksize=50)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(files)}...")

    df = pd.DataFrame(results)
    out_path = os.path.join(BASE_DIR, "low_flow_gages.csv")
    df.to_csv(out_path, index=False)

    n_lowflow = df['has_lowflow'].sum()
    n_total = len(df)
    pct = 100 * n_lowflow / n_total if n_total > 0 else 0
    print(f"Done. {n_lowflow}/{n_total} gages ({pct:.1f}%) have low-flow periods.")
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
