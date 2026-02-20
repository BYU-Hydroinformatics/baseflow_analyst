import pandas as pd
import os
import ssl
import urllib.request
import time
from io import StringIO
from datetime import datetime

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context


def fetch_sites_by_state():
    """
    Fetch USGS sites with discharge data (00060), state by state.
    Returns a DataFrame with combined site information for all CONUS states + DC.
    """
    print("Fetching USGS sites by state...")

    states = [
        'AL', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC'
    ]

    all_sites = []

    for state in states:
        print(f"Fetching sites for state: {state}")
        url = (
            f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
            f"&stateCd={state}&parameterCd=00060&siteType=ST"
            f"&siteStatus=active&hasDataTypeCd=dv"
            f"&siteOutput=expanded"
        )

        try:
            time.sleep(1)

            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')

            lines = content.split('\n')
            data_lines = []
            header_line = None

            for line in lines:
                if line.startswith('#'):
                    continue
                elif header_line is None:
                    header_line = line
                    data_lines.append(line)
                else:
                    data_lines.append(line)

            if len(data_lines) <= 2:
                print(f"No sites found for state {state}")
                continue

            state_df = pd.read_csv(StringIO('\n'.join(data_lines)), delimiter='\t', dtype=str)

            if len(state_df) > 1:
                state_df = state_df.iloc[1:, :]
                print(f"Found {len(state_df)} sites for {state}")
                all_sites.append(state_df)
            else:
                print(f"No valid data for state {state}")

        except Exception as e:
            print(f"Error fetching sites for state {state}: {e}")

    if not all_sites:
        print("No sites found for any state.")
        return pd.DataFrame()

    combined_df = pd.concat(all_sites, ignore_index=True)

    selected_columns = ['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va', 'alt_va', 'drain_area_va']
    available_columns = [col for col in selected_columns if col in combined_df.columns]

    missing_columns = [col for col in selected_columns if col not in combined_df.columns]
    if missing_columns:
        print(f"Warning: The following columns were not found: {missing_columns}")

    combined_df['parm_cd'] = '00060'
    result_df = combined_df[available_columns + ['parm_cd']]

    print(f"Total sites found: {len(result_df)}")
    result_df = result_df.drop_duplicates(subset=['site_no'])
    print(f"After removing duplicates: {len(result_df)} unique sites")

    return result_df


def download_streamflow_data(site_no, save_dir):
    """
    Download all available historical daily streamflow data for a site.

    Args:
        site_no: The USGS site number
        save_dir: Directory to save the data

    Returns:
        True if data was downloaded successfully, False otherwise
    """
    base_url = 'https://waterservices.usgs.gov/nwis/dv/?'
    start_date = '1800-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    url = (
        f'{base_url}sites={site_no}&parameterCd=00060'
        f'&startDT={start_date}&endDT={end_date}&format=rdb'
    )

    try:
        time.sleep(0.5)

        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')

        lines = content.split('\n')
        data_lines = [line for line in lines if not line.startswith('#')]

        if len(data_lines) <= 2:
            return False

        df = pd.read_csv(StringIO('\n'.join(data_lines)), delimiter='\t', dtype=str)
        if len(df) <= 1:
            return False

        df = df.iloc[1:, :]

        if len(df) == 0:
            return False

        columns_to_remove = ['site_no', 'agency_cd']
        for col in df.columns:
            if '_00060_00003_cd' in col:
                columns_to_remove.append(col)

        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

        rename_dict = {}
        if 'datetime' in df.columns:
            rename_dict['datetime'] = 'date'
        for col in df.columns:
            if '_00060_00003' in col and '_00060_00003_cd' not in col:
                rename_dict[col] = 'streamflow'

        df = df.rename(columns=rename_dict)

        output_file = os.path.join(save_dir, f"{site_no}.csv")
        df.to_csv(output_file, index=False)
        print(f"Downloaded data for site {site_no} ({len(df)} records)")

        return True

    except Exception as e:
        print(f"Error downloading data for site {site_no}: {e}")
        return False


def process_all_sites(site_numbers, save_dir, batch_size=50):
    """
    Process all sites in batches, downloading historical daily streamflow data.
    Skips sites that already have a downloaded CSV file (resume support).

    Args:
        site_numbers: List of all site numbers
        save_dir: Directory to save the output files
        batch_size: Number of sites to process in each batch

    Returns:
        List of site numbers that have data
    """
    total_sites = len(site_numbers)

    # Check which sites already have downloaded data
    already_downloaded = set()
    for f in os.listdir(save_dir):
        if f.endswith('.csv') and f not in ('site_info.csv', 'gauges_with_data.csv', 'gauges_without_data.csv'):
            already_downloaded.add(f.replace('.csv', ''))

    sites_to_download = [s for s in site_numbers if s not in already_downloaded]

    print(f"\nTotal sites: {total_sites}")
    print(f"Already downloaded: {len(already_downloaded)}")
    print(f"Remaining to download: {len(sites_to_download)}")

    # Start with already downloaded sites as successful
    sites_with_data = [s for s in site_numbers if s in already_downloaded]
    sites_without_data = []

    if not sites_to_download:
        print("All sites already downloaded. Nothing to do.")
    else:
        for i in range(0, len(sites_to_download), batch_size):
            batch = sites_to_download[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(sites_to_download) + batch_size - 1) // batch_size
            print(f"\nBatch {batch_num}/{total_batches}: processing {len(batch)} sites "
                  f"(sites {i + 1}-{i + len(batch)} of {len(sites_to_download)} remaining)")

            for site_no in batch:
                if download_streamflow_data(site_no, save_dir):
                    sites_with_data.append(site_no)
                else:
                    print(f"Site {site_no} has no data available - skipping")
                    sites_without_data.append(site_no)

            print(f"Batch {batch_num} done: {len(sites_with_data)} successful, "
                  f"{len(sites_without_data)} failed so far")

    # Save summary CSVs
    sites_with_data_df = pd.DataFrame({'site_no': sites_with_data})
    output_csv_path = os.path.join(save_dir, "gauges_with_data.csv")
    sites_with_data_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved {len(sites_with_data)} gauge IDs to {output_csv_path}")

    if sites_without_data:
        sites_without_data_df = pd.DataFrame({'site_no': sites_without_data})
        failed_csv_path = os.path.join(save_dir, "gauges_without_data.csv")
        sites_without_data_df.to_csv(failed_csv_path, index=False)
        print(f"Saved {len(sites_without_data)} failed gauge IDs to {failed_csv_path}")

    return sites_with_data


def main():
    save_dir = "./usgs_daily_streamflow"
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Fetch all gage site numbers (skip if site_info.csv already exists)
    sites_info_path = os.path.join(save_dir, "site_info.csv")

    if os.path.exists(sites_info_path):
        print(f"Resuming: loading existing site list from {sites_info_path}")
        sites_df = pd.read_csv(sites_info_path, dtype=str)
    else:
        sites_df = fetch_sites_by_state()

        if sites_df.empty:
            print("No sites found. Exiting.")
            return

        sites_df.to_csv(sites_info_path, index=False)
        print(f"Saved site information to {sites_info_path}")

    site_numbers = sites_df['site_no'].tolist()
    print(f"\nFound {len(site_numbers)} unique gages across all states")
    print("Downloading all available historical daily streamflow data for each gage...\n")

    # Uncomment to limit sites for testing
    # site_numbers = site_numbers[:10]

    # Step 2: Download all available historical daily data for each gage
    process_all_sites(site_numbers, save_dir, batch_size=50)

    print("\nDownload complete. Results saved to:", save_dir)
    print("- site_info.csv: metadata for all discovered gages")
    print("- [site_no].csv: daily streamflow data per gage")
    print("- gauges_with_data.csv: list of gages with successful downloads")
    print("- gauges_without_data.csv: list of gages with no data")


if __name__ == '__main__':
    main()
