# portfolio_analyzer/cache_utils.py
import pandas as pd
import json
from pathlib import Path
import logging
import hashlib
from datetime import datetime, timedelta, timezone

CACHE_BASE_DIR = Path(__file__).resolve().parent.parent / ".cache"
ASSET_INFO_CACHE_DIR = CACHE_BASE_DIR / "asset_info"
HISTORICAL_PRICES_CACHE_DIR = CACHE_BASE_DIR / "historical_prices"
SYMBOL_MAPPINGS_CACHE_DIR = CACHE_BASE_DIR / "symbol_mappings"

ASSET_INFO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SYMBOL_MAPPINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SYMBOL_MAPPINGS_FILE = SYMBOL_MAPPINGS_CACHE_DIR / "user_symbol_mappings.json"

ASSET_INFO_MAX_AGE_HOURS = 1  # Re-fetch current price info if older than X hours


def get_cached_asset_info(symbol: str, rebuild_cache: bool = False) -> dict | None:
    if rebuild_cache:
        return None

    cache_file = ASSET_INFO_CACHE_DIR / f"{symbol.upper()}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            cache_timestamp_str = cached_data.get("_cache_timestamp_utc")
            if cache_timestamp_str:
                cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                if datetime.now(timezone.utc) - cache_timestamp < timedelta(hours=ASSET_INFO_MAX_AGE_HOURS):
                    logging.debug(f"Asset info for {symbol} is fresh from cache.")
                    return cached_data.get("data")
                else:
                    logging.info(
                        f"Cached asset info for {symbol} is STALE (older than {ASSET_INFO_MAX_AGE_HOURS}h). Will re-fetch.")
                    return None
            else:
                logging.warning(f"Cached asset info for {symbol} missing timestamp. Will re-fetch.")
                return None
        except Exception as e:
            logging.warning(f"Error loading/validating cached asset info for {symbol}: {e}. Will re-fetch.")
    return None


def cache_asset_info(symbol: str, info_data: dict):
    if not isinstance(info_data, dict): return
    cache_file = ASSET_INFO_CACHE_DIR / f"{symbol.upper()}.json"
    data_to_cache = {
        "_cache_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": info_data
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(data_to_cache, f, indent=4, default=str)
        logging.debug(f"Cached asset info for {symbol}.")
    except Exception as e:
        logging.error(f"Error caching asset info for {symbol}: {e}")


def _generate_historical_prices_cache_filename(symbols: list, period_str: str, end_date_str: str) -> str:
    """
    Generates a consistent filename for historical price cache, including an end date.
    end_date_str: YYYY-MM-DD string representing the "as of" date for the period.
    """
    if not symbols: return f"NO_SYMBOLS_{period_str}_{end_date_str}.parquet"
    # Sort symbols for consistency, use uppercase
    sorted_symbols_str = "-".join(sorted([s.upper() for s in symbols]))

    filename_base = f"{sorted_symbols_str}_{period_str}_{end_date_str}"
    if len(filename_base) > 150:  # Keep filenames manageable
        filename_base = f"{hashlib.md5(sorted_symbols_str.encode()).hexdigest()}_{period_str}_{end_date_str}"
    return f"HIST_{filename_base}.parquet"


def get_cached_historical_prices(symbols: list, period: str, end_date_for_period: datetime,
                                 rebuild_cache: bool = False) -> pd.DataFrame | None:
    if rebuild_cache: return None

    end_date_str = end_date_for_period.strftime('%Y-%m-%d')
    filename = _generate_historical_prices_cache_filename(symbols, period, end_date_str)
    cache_file = HISTORICAL_PRICES_CACHE_DIR / filename

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index).tz_localize(None)  # Ensure naive for consistency
            logging.info(
                f"Historical prices for {symbols} (period: {period}, end: {end_date_str}): Using CACHED data ({filename}).")
            return df
        except Exception as e:
            logging.warning(f"Error loading cached historical prices ({filename}): {e}. Will re-fetch.")
    return None


def cache_historical_prices(symbols: list, period: str, end_date_for_period: datetime, prices_df: pd.DataFrame):
    if not isinstance(prices_df, pd.DataFrame): return
    # Do not cache if prices_df is empty, as this might mean a failed fetch for valid symbols.
    # Let it retry next time. Cache only successful, non-empty fetches.
    if prices_df.empty and symbols:
        logging.debug(
            f"Skipping cache for empty historical prices_df for {symbols} (period: {period}). Might be fetch error.")
        return

    end_date_str = end_date_for_period.strftime('%Y-%m-%d')
    filename = _generate_historical_prices_cache_filename(symbols, period, end_date_str)
    cache_file = HISTORICAL_PRICES_CACHE_DIR / filename
    try:
        df_to_cache = prices_df.copy()
        if isinstance(df_to_cache.index, pd.DatetimeIndex) and df_to_cache.index.tz is not None:
            df_to_cache.index = df_to_cache.index.tz_localize(None)

        df_to_cache.to_parquet(cache_file, index=True)
        logging.debug(f"Cached historical prices for {symbols} (period: {period}, end: {end_date_str}) to {filename}.")
    except Exception as e:
        logging.error(f"Error caching historical prices to ({filename}): {e}")


def load_symbol_mappings() -> dict:
    if SYMBOL_MAPPINGS_FILE.exists():
        try:
            with open(SYMBOL_MAPPINGS_FILE, 'r') as f:
                mappings = json.load(f)
            logging.debug("Loaded symbol mappings from cache.")
            return mappings
        except Exception as e:
            logging.warning(f"Error loading symbol mappings: {e}. Starting with empty mappings.")
    return {}


def save_symbol_mappings(mappings: dict):
    try:
        with open(SYMBOL_MAPPINGS_FILE, 'w') as f:
            json.dump(mappings, f, indent=4)
        logging.debug("Saved symbol mappings to cache.")
    except Exception as e:
        logging.error(f"Error saving symbol mappings: {e}")


SYMBOL_MAPPINGS = load_symbol_mappings()
