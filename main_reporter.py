# main_reporter.py
import pandas as pd
import logging
from pathlib import Path
import webbrowser
import argparse
import re  # Was in load_portfolio_data, ensure it's here if used at module level
import numpy as np  # Was in load_portfolio_data, ensure it's here if used at module level

from portfolio_analyzer.data_processing import process_portfolio_data
from portfolio_analyzer.report_generator import generate_html_report
from portfolio_analyzer.section_builders import (
    build_key_metrics_section,
    build_holdings_summary_section,
    build_historical_networth_section,
    build_summary_by_symbol_section,
    build_summary_by_risk_level_section,
    build_summary_by_asset_class_section,
    build_top_movers_section
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parent


def clean_currency_string(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip()
    if not s: return np.nan
    is_negative = False
    if s.startswith('(') and s.endswith(')'): is_negative = True; s = s[1:-1]
    s = re.sub(r'[$\s,]', '', s)
    try:
        num = float(s)
        return -num if is_negative else num
    except ValueError:
        logging.debug(f"Could not convert currency string '{value}' to float.")
        return np.nan


def load_portfolio_data(input_file_path: str | None) -> pd.DataFrame:
    core_expected_cols = ["Account", "Symbol", "Amount", "Cost"]
    possible_numeric_cols_from_csv = ["Amount", "Cost", "price", "Value"]
    if input_file_path:
        try:
            input_path = Path(input_file_path)
            if not input_path.exists(): raise FileNotFoundError("Input CSV file not found.")
            if not input_path.is_file() or input_path.suffix.lower() != ".csv": raise ValueError(
                "Invalid input file type, must be .csv.")
            temp_df = pd.read_csv(input_path, dtype={"Account": str, "Symbol": str})
            logging.info(f"Successfully loaded raw data from: {input_path}")
            standardized_columns = {}
            for col in temp_df.columns:
                col_lower = col.lower().replace(" ", "")
                if col_lower == "account":
                    standardized_columns[col] = "Account"
                elif col_lower == "symbol" or col_lower == "ticker":
                    standardized_columns[col] = "Symbol"
                elif col_lower == "amount" or col_lower == "quantity" or col_lower == "shares":
                    standardized_columns[col] = "Amount"
                elif col_lower == "cost" or col_lower == "costbasis":
                    standardized_columns[col] = "Cost"
                elif col_lower == "price":
                    standardized_columns[col] = "price"
                elif col_lower == "value" or col_lower == "marketvalue":
                    standardized_columns[col] = "Value_from_csv"
            temp_df.rename(columns=standardized_columns, inplace=True)
            if not all(col in temp_df.columns for col in core_expected_cols):
                missing = [col for col in core_expected_cols if col not in temp_df.columns]
                raise ValueError(f"CSV core columns missing: {missing}")
            cols_to_keep = list(core_expected_cols)
            for pc in possible_numeric_cols_from_csv:
                if pc in temp_df.columns and pc not in cols_to_keep: cols_to_keep.append(pc)
            if "name" in temp_df.columns and "name" not in cols_to_keep: cols_to_keep.append("name")
            df = temp_df[cols_to_keep].copy()
            for col_name in ["Amount", "Cost", "price", "Value_from_csv"]:
                if col_name in df.columns:
                    df[col_name] = df[col_name].apply(clean_currency_string)
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

            # Identify rows with NaN in 'Amount' or 'Symbol' *before* dropping them
            # Note: 'Symbol' is loaded as string, so it's unlikely to be NaN unless it was empty in CSV.
            # 'Amount' is numeric, so it can become NaN after to_numeric if it wasn't a valid number.
            invalid_rows_mask = df['Amount'].isna() | df['Symbol'].isna() | (df['Symbol'] == '')
            if 'Symbol' in df.columns:  # Ensure Symbol column exists
                invalid_rows_mask = df['Amount'].isna() | df['Symbol'].isna() | (
                        df['Symbol'].astype(str).str.strip() == '')

            invalid_df = df[invalid_rows_mask]

            initial_rows = len(df)
            df.dropna(subset=['Amount', 'Symbol'], inplace=True)
            # Also drop rows where Symbol might be an empty string after stripping
            if 'Symbol' in df.columns:
                df = df[df['Symbol'].astype(str).str.strip() != '']

            dropped_count = initial_rows - len(df)

            if dropped_count > 0:
                logging.warning(
                    f"{dropped_count} rows dropped due to invalid/missing 'Amount' or 'Symbol'.")
                if not invalid_df.empty:
                    logging.warning("First few invalid rows (before dropping):")
                    # Log only a few to avoid spamming the console if there are many
                    log_limit = min(len(invalid_df), 3)
                    for i in range(log_limit):
                        logging.warning(f"  Row index {invalid_df.index[i]}: {invalid_df.iloc[i].to_dict()}")
                else:
                    # This case might happen if rows were dropped due to empty Symbol strings that weren't initially NaN
                    logging.warning(
                        "  (Could not display specific invalid rows, check for empty Symbol strings or Amount conversion issues)")

            if df.empty: raise ValueError("No valid data rows after initial cleaning.")

            if 'Cost' in df.columns:
                df['Cost'] = df['Cost'].fillna(0.0)
            else:
                df['Cost'] = 0.0
                logging.warning("Cost column was missing, defaulted to 0.0.")
            df['Symbol'] = df['Symbol'].astype(str).str.upper().str.strip()
            logging.info(f"Cleaned and prepared portfolio data. Shape: {df.shape}")
            return df
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"{e}. Proceeding with default sample data.")
        except Exception as e:
            logging.error(f"Unexpected error loading CSV '{input_file_path}': {e}")
            logging.warning(
                "Proceeding with default sample data.")
    logging.info("Using default sample data.")
    default_data = {"Account": ["Retirement 401k", "Retirement 401k", "Brokerage", "Brokerage", "Brokerage", "Roth IRA",
                                "Checking Account"],
                    "Symbol": ["FXAIX", "VEMAX", "AAPL", "MSFT", "GOOG", "VOO", "CASH"],
                    "Amount": [50.0, 30.0, 10.0, 15.0, 5.0, 20.0, 5000.0],
                    "Cost": [7000.0, 2500.0, 1500.0, 3000.0, 4000.0, 8000.0, 0.0]}
    df_default = pd.DataFrame(default_data)
    df_default['Symbol'] = df_default['Symbol'].astype(str).str.upper()
    return df_default


def main():
    parser = argparse.ArgumentParser(description="Generate a portfolio report.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Ignore existing cache and fetch all data from APIs, then update cache.")
    parser.add_argument("-i", "--input-file", type=str, help="Path to a CSV file containing the portfolio data.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Disable interactive prompts for missing symbols.")
    args = parser.parse_args()
    interactive_mode = not args.non_interactive

    if args.rebuild_cache: logging.info("REBUILD_CACHE flag is set. Cache will be ignored and overwritten.")

    portfolio_df_input = load_portfolio_data(args.input_file)
    if portfolio_df_input.empty: logging.error("Portfolio data is empty. Cannot generate report."); return

    logging.info("Processing portfolio data...")
    processed_df = process_portfolio_data(portfolio_df_input, rebuild_cache=args.rebuild_cache,
                                          interactive_fallback=interactive_mode)
    if processed_df.empty or 'Market Value' not in processed_df.columns: logging.error(
        "Failed to process portfolio data. Exiting."); return

    logging.info("\nProcessed Portfolio Data (first 5 rows):")
    logging.info(processed_df.head().to_string())

    report_sections = {
        "Key Performance Indicators": build_key_metrics_section(
            processed_df, portfolio_df_input, rebuild_cache=args.rebuild_cache, interactive_fallback=interactive_mode
        ),
        "Historical Net Worth": build_historical_networth_section(
            portfolio_df_input, default_period="1y", rebuild_cache=args.rebuild_cache,
            interactive_fallback=interactive_mode
        ),
        "Top Asset Movers": build_top_movers_section(
            portfolio_df_input, rebuild_cache=args.rebuild_cache, interactive_fallback=interactive_mode
        ),
        "Summary by Symbol": build_summary_by_symbol_section(
            processed_df, rebuild_cache=args.rebuild_cache, interactive_fallback=interactive_mode
        ),
        "Summary by Asset Risk Level": build_summary_by_risk_level_section(
            processed_df, rebuild_cache=args.rebuild_cache, interactive_fallback=interactive_mode
        ),
        "Summary by Asset Class": build_summary_by_asset_class_section(
            processed_df, rebuild_cache=args.rebuild_cache, interactive_fallback=interactive_mode
        ),
        "Holdings Details": build_holdings_summary_section(processed_df)
    }

    # Prepare data for JS drill-down to make it more robust
    drilldown_data_df = processed_df.copy()
    drilldown_data_df.rename(columns={
        'Market Value': 'Market_Value_raw',
        'Cost': 'Cost_raw',
        'price': 'price_raw',
        'Amount': 'Amount_raw'
    }, inplace=True)
    drilldown_cols = ['Account', 'Symbol', 'name', 'Amount_raw', 'price_raw', 'Market_Value_raw', 'Cost_raw',
                      'Risk Level', 'Asset Class']
    # Ensure all required columns exist before trying to select them
    final_drilldown_cols = [col for col in drilldown_cols if col in drilldown_data_df.columns]
    drilldown_json = drilldown_data_df[final_drilldown_cols].to_json(orient='records', date_format='iso')

    logging.info("Generating HTML report...")
    report_file = PROJECT_ROOT / "portfolio_report.html"
    generate_html_report(report_sections, output_path=str(report_file), drilldown_data=drilldown_json)

    try:
        import os
        if os.path.exists(str(report_file)):
            webbrowser.open(f"file://{report_file.resolve()}")
    except Exception as e:
        logging.warning(f"Could not auto-open report: {e}. Please open '{report_file.name}' manually.")


if __name__ == "__main__":
    main()
