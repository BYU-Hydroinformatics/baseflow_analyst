import pandas as pd
import numpy as np
import os
import sys
import traceback

import pybfs

# Unit conversions
CFS_TO_M3_PER_DAY = 0.0283168 * 86400  # 1 cfs = 2446.58 m³/day
SQMI_TO_M2 = 2_589_988.11  # 1 sq mile = 2,589,988.11 m²


def load_drainage_areas(site_info_path):
    """
    Load drainage areas from site_info.csv (drain_area_va column, in sq miles).
    Converts to m² and skips sites with missing values.

    Returns:
        Dict mapping site_no -> drainage area in m²
    """
    df = pd.read_csv(site_info_path, dtype=str)

    if 'drain_area_va' not in df.columns:
        print(f"Error: 'drain_area_va' column not found in {site_info_path}")
        return {}

    area_dict = {}
    skipped = 0
    for _, row in df.iterrows():
        site = row['site_no']
        area_str = row.get('drain_area_va', '')
        if pd.notna(area_str) and str(area_str).strip():
            try:
                area_sqmi = float(area_str)
                if area_sqmi > 0:
                    area_dict[site] = area_sqmi * SQMI_TO_M2
                else:
                    skipped += 1
            except ValueError:
                skipped += 1
        else:
            skipped += 1

    print(f"Loaded drainage areas for {len(area_dict)} sites "
          f"({skipped} skipped due to missing/invalid values)")
    return area_dict


def calibrate_site(site_no, streamflow_path, area_m2):
    """
    Run BFS calibration for a single site.

    Args:
        site_no: USGS site number
        streamflow_path: Path to the streamflow CSV (date, streamflow in cfs)
        area_m2: Drainage area in m²

    Returns:
        Tuple of (bf_params, bff) DataFrames, or (None, None) on failure
    """
    df = pd.read_csv(streamflow_path, dtype={'date': str, 'streamflow': str})

    # Parse dates and streamflow
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['streamflow'] = pd.to_numeric(df['streamflow'], errors='coerce')

    # Drop rows with missing streamflow
    df = df.dropna(subset=['streamflow'])

    if len(df) < 365:
        print(f"  Site {site_no}: only {len(df)} days of data, need at least 365. Skipping.")
        return None, None

    # Convert cfs to m³/day
    tmp_q = df['streamflow'].values * CFS_TO_M3_PER_DAY
    dys = df['date'].values

    # Skip if all zeros or negative
    if np.nanmax(tmp_q) <= 0:
        print(f"  Site {site_no}: no positive streamflow values. Skipping.")
        return None, None

    bf_params, bff, ci_table, bfs_out = pybfs.bfs_calibrate(
        tmp_site=site_no,
        tmp_area=area_m2,
        tmp_q=tmp_q,
        dys=dys
    )

    return bf_params, bff


def main():
    streamflow_dir = "./usgs_daily_streamflow"
    output_dir = "./usgs_calibration_results"
    os.makedirs(output_dir, exist_ok=True)

    params_dir = os.path.join(output_dir, "params")
    bff_dir = os.path.join(output_dir, "bff")
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(bff_dir, exist_ok=True)

    # Step 1: Load site info and drainage areas
    site_info_path = os.path.join(streamflow_dir, "site_info.csv")
    if not os.path.exists(site_info_path):
        print(f"Error: {site_info_path} not found. Run usgs_download_all_daily.py first.")
        return

    area_dict = load_drainage_areas(site_info_path)
    if not area_dict:
        print("No valid drainage areas found. Exiting.")
        return

    # Get list of downloaded gages
    gauges_with_data_path = os.path.join(streamflow_dir, "gauges_with_data.csv")
    if not os.path.exists(gauges_with_data_path):
        print(f"Error: {gauges_with_data_path} not found. Run usgs_download_all_daily.py first.")
        return

    gauges_df = pd.read_csv(gauges_with_data_path, dtype=str)
    all_sites = gauges_df['site_no'].tolist()
    print(f"Found {len(all_sites)} gages with downloaded streamflow data")

    # Filter to sites that have both streamflow data and drainage area
    valid_sites = [s for s in all_sites if s in area_dict]
    skipped_no_area = len(all_sites) - len(valid_sites)
    if skipped_no_area > 0:
        print(f"Skipping {skipped_no_area} sites with no drainage area available")
    print(f"Sites ready for calibration: {len(valid_sites)}")

    # Step 3: Check which sites are already calibrated (resume support)
    already_calibrated = set()
    for f in os.listdir(params_dir):
        if f.startswith('params_') and f.endswith('.csv'):
            already_calibrated.add(f.replace('params_', '').replace('.csv', ''))

    # Also track previously failed sites
    failed_log_path = os.path.join(output_dir, "calibration_failed.csv")
    already_failed = set()
    if os.path.exists(failed_log_path):
        failed_df = pd.read_csv(failed_log_path, dtype=str)
        already_failed = set(failed_df['site_no'].tolist())

    sites_to_calibrate = [
        s for s in valid_sites
        if s not in already_calibrated and s not in already_failed
    ]

    print(f"\nAlready calibrated: {len(already_calibrated)}")
    print(f"Previously failed: {len(already_failed)}")
    print(f"Remaining to calibrate: {len(sites_to_calibrate)}")

    if not sites_to_calibrate:
        print("All sites already processed. Nothing to do.")
    else:
        failed_sites = list(already_failed)

        for idx, site_no in enumerate(sites_to_calibrate):
            streamflow_path = os.path.join(streamflow_dir, f"{site_no}.csv")
            if not os.path.exists(streamflow_path):
                print(f"[{idx + 1}/{len(sites_to_calibrate)}] Site {site_no}: "
                      f"streamflow file not found. Skipping.")
                continue

            area_m2 = area_dict[site_no]
            print(f"\n[{idx + 1}/{len(sites_to_calibrate)}] Calibrating site {site_no} "
                  f"(area={area_m2:.2e} m²)...")

            try:
                bf_params, bff = calibrate_site(site_no, streamflow_path, area_m2)

                if bf_params is not None:
                    bf_params.to_csv(
                        os.path.join(params_dir, f"params_{site_no}.csv"),
                        index=False, float_format='%.6g'
                    )
                    bff.to_csv(
                        os.path.join(bff_dir, f"bff_{site_no}.csv"),
                        index=False, float_format='%.6g'
                    )
                    already_calibrated.add(site_no)
                    print(f"  Site {site_no}: calibration complete "
                          f"(BFF={bff['BFF'].iloc[0]:.4f}, Error={bff['Error'].iloc[0]:.4f})")
                else:
                    print(f"  Site {site_no}: calibration returned no results")
                    failed_sites.append(site_no)

            except Exception as e:
                print(f"  Site {site_no}: calibration failed - {e}")
                traceback.print_exc()
                failed_sites.append(site_no)

            # Save failed list after each site for resume
            if failed_sites:
                pd.DataFrame({'site_no': failed_sites}).to_csv(
                    failed_log_path, index=False
                )

    # Step 4: Combine all calibrated params into one summary file
    print("\nCombining all calibration results...")
    all_params = []
    all_bff = []

    for f in sorted(os.listdir(params_dir)):
        if f.startswith('params_') and f.endswith('.csv'):
            all_params.append(pd.read_csv(os.path.join(params_dir, f)))

    for f in sorted(os.listdir(bff_dir)):
        if f.startswith('bff_') and f.endswith('.csv'):
            all_bff.append(pd.read_csv(os.path.join(bff_dir, f)))

    if all_params:
        combined_params = pd.concat(all_params, ignore_index=True)
        combined_params.to_csv(
            os.path.join(output_dir, "all_params.csv"),
            index=False, float_format='%.6g'
        )
        print(f"Saved combined parameters: {len(combined_params)} sites -> all_params.csv")

    if all_bff:
        combined_bff = pd.concat(all_bff, ignore_index=True)
        combined_bff.to_csv(
            os.path.join(output_dir, "all_bff.csv"),
            index=False, float_format='%.6g'
        )
        print(f"Saved combined baseflow fractions: {len(combined_bff)} sites -> all_bff.csv")

    print(f"\nDone. Results saved to: {output_dir}")
    print(f"  - params/params_<site_no>.csv: per-site calibration parameters")
    print(f"  - bff/bff_<site_no>.csv: per-site baseflow fractions")
    print(f"  - all_params.csv: combined parameters for all calibrated sites")
    print(f"  - all_bff.csv: combined baseflow fractions for all sites")
    print(f"  - calibration_failed.csv: sites that failed calibration")


if __name__ == '__main__':
    main()
