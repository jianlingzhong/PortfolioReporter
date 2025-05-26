# portfolio_analyzer/cache_utils.py
import pandas as pd
import json
from pathlib import Path
import logging
import hashlib

CACHE_BASE_DIR = Path(__file__).resolve().parent.parent / ".cache"

ASSET_INFO_CACHE_DIR = CACHE_BASE_DIR / "asset_info"
ASSET_INFO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORICAL_PRICES_CACHE_DIR = CACHE_BASE_DIR / "historical_prices"
HISTORICAL_PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SYMBOL_MAPPINGS_CACHE_DIR = CACHE_BASE_DIR / "symbol_mappings"
SYMBOL_MAPPINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SYMBOL_MAPPINGS_FILE = SYMBOL_MAPPINGS_CACHE_DIR / "user_symbol_mappings.json"


# --- Asset Info (Ticker.info) Cache ---

def get_cached_asset_info(symbol: str, rebuild_cache: bool = False) -> dict | None:  # Added rebuild_cache
    """Loads asset info from cache if available and rebuild_cache is False."""
    if rebuild_cache:
        return None  # Skip cache lookup if rebuilding

    cache_file = ASSET_INFO_CACHE_DIR / f"{symbol.upper()}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                info = json.load(f)
            # logging.debug(f"Loaded asset info for {symbol} from cache.") # Moved to data_processing
            return info
        except Exception as e:
            logging.warning(f"Error loading cached asset info for {symbol}: {e}. Will re-fetch.")
            return None
    return None


def cache_asset_info(symbol: str, info_data: dict):
    """Saves asset info to cache."""
    if not isinstance(info_data, dict):
        logging.error(f"Invalid data type for caching asset info of {symbol}. Expected dict.")
        return
    cache_file = ASSET_INFO_CACHE_DIR / f"{symbol.upper()}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump(info_data, f, indent=4, default=str)
        logging.debug(f"Cached asset info for {symbol}.")
    except Exception as e:
        logging.error(f"Error caching asset info for {symbol}: {e}")


# --- Historical Prices Cache ---

def _generate_historical_prices_cache_filename(symbols: list, period: str) -> str:
    if not symbols: return f"NO_SYMBOLS_{period}.parquet"
    sorted_symbols_str = "-".join(sorted([s.upper() for s in symbols]))
    if len(sorted_symbols_str) > 100:
        return f"MULTI_{hashlib.md5(sorted_symbols_str.encode()).hexdigest()}_{period}.parquet"
    return f"MULTI_{sorted_symbols_str}_{period}.parquet"


def get_cached_historical_prices(symbols: list, period: str,
                                 rebuild_cache: bool = False) -> pd.DataFrame | None:  # Added rebuild_cache
    """Loads historical prices DataFrame from cache if available and rebuild_cache is False."""
    if rebuild_cache:
        return None  # Skip cache lookup if rebuilding

    filename = _generate_historical_prices_cache_filename(symbols, period)
    cache_file = HISTORICAL_PRICES_CACHE_DIR / filename
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # logging.debug(f"Loaded historical prices for {symbols} (period: {period}) from cache: {filename}") # Moved
            return df
        except Exception as e:
            logging.warning(
                f"Error loading cached historical prices for {symbols} (period: {period}): {e}. Will re-fetch.")
            return None
    return None


def cache_historical_prices(symbols: list, period: str, prices_df: pd.DataFrame):
    """Saves historical prices DataFrame to cache."""
    if not isinstance(prices_df, pd.DataFrame):
        logging.error(f"Invalid data type for caching historical prices of {symbols}. Expected DataFrame.")
        return
    if prices_df.empty:
        logging.debug(f"Skipping cache for empty historical prices_df for {symbols} (period: {period}).")
        return
    filename = _generate_historical_prices_cache_filename(symbols, period)
    cache_file = HISTORICAL_PRICES_CACHE_DIR / filename
    try:
        prices_df.to_parquet(cache_file, index=True)
        logging.debug(f"Cached historical prices for {symbols} (period: {period}) to {filename}.")
    except Exception as e:
        logging.error(f"Error caching historical prices for {symbols} (period: {period}): {e}")


def load_symbol_mappings() -> dict:
    """Loads user-defined symbol mappings from cache."""
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
    """Saves user-defined symbol mappings to cache."""
    try:
        with open(SYMBOL_MAPPINGS_FILE, 'w') as f:
            json.dump(mappings, f, indent=4)
        logging.debug("Saved symbol mappings to cache.")
    except Exception as e:
        logging.error(f"Error saving symbol mappings: {e}")


# Global variable to hold mappings in memory for the current session
# Initialize by loading from disk
SYMBOL_MAPPINGS = load_symbol_mappings()
