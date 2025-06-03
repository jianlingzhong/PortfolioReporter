# portfolio_analyzer/data_processing.py
import pandas as pd
import yfinance as yf
import logging
import time
import numpy as np
from datetime import datetime, timedelta, timezone
import requests  # Still useful for requests.exceptions
from .cache_utils import (
  get_cached_asset_info, cache_asset_info,
  get_cached_historical_prices, cache_historical_prices,
  SYMBOL_MAPPINGS, save_symbol_mappings
)

try:
  # import requests.exceptions # Already imported by 'import requests'
  NETWORK_EXCEPTIONS_TUPLE = (requests.exceptions.RequestException,)
  if hasattr(yf, 'YFinanceError') and getattr(yf, 'YFinanceError') not in NETWORK_EXCEPTIONS_TUPLE:
    NETWORK_EXCEPTIONS_TUPLE += (yf.YFinanceError,)
except ImportError:
  if hasattr(yf, 'YFinanceError'):
    NETWORK_EXCEPTIONS_TUPLE = (yf.YFinanceError,)
  else:
    NETWORK_EXCEPTIONS_TUPLE = (IOError, OSError, Exception)
  logging.warning("Could not import requests.exceptions directly for specific network error handling.")
NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS_TUPLE

TICKER_CACHE = {}  # Stores yf.Ticker objects
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30  # Define a default timeout for yf.download

CASH_EQUIVALENT_SYMBOLS = {"CASH", "SPAXX", "VMFXX", "SWVXX"}

# --- CONSTANTS FOR RISK/ASSET CLASS DETERMINATION ---
# Moved to module level to fix pylint C0103 (invalid-name) and improve performance.
RISK_LEVEL_CASH_KEYWORDS = [
  "money market", "treasury bill", "spaxx", "vmfxx", "swvxx", "fdrxx", "govmmkt",
  "fgmxx", "t-bill"
]
RISK_LEVEL_BOND_KEYWORDS = [
  "bond", "fixed income", "treasury", "government debt", "corporate debt", "aggregate bond",
  "muni", "bnd", "agg"
]
RISK_LEVEL_TARGET_DATE_KEYWORDS = [
  "target date", "target-date", "target retirement", "freedom index",
  "vffsx", "vdadx", "vforx", "vthrx", "vtwax",
  "fitfx", "ffwax", "fdeex"
]
RISK_LEVEL_BROAD_EQUITY_KEYWORDS = [
  "total stock market", "s&p 500", "500 index", "large blend", "large cap",
  "russell 1000", "russell 3000", "developed markets", "world stock", "global equity",
  "broad market", "nasdaq-100",
  "vfiax", "vtsax", "voo", "vti",
  "fxaix", "fskax", "fzilx",
  "swppx", "swtsx",
  "qqq", "spy"
]
RISK_LEVEL_SECTOR_THEMATIC_KEYWORDS = [
  "technology", "health", "financial", "real estate", "communications", "utilities",
  "consumer discretionary", "growth", "value", "small cap", "mid cap", "small-cap",
  "mid-cap", "flapx", "flxsx", "fdgrx"
]
RISK_LEVEL_COMMODITY_KEYWORDS = ["gold", "commodities", "gld"]

ASSET_CLASS_TICKER_MAP = {
  # Broad Market Equity
  'vti': 'Broad Market Equity', 'vtsax': 'Broad Market Equity',
  'voo': 'Broad Market Equity', 'vfiax': 'Broad Market Equity',
  'fxaix': 'Broad Market Equity', 'fskax': 'Broad Market Equity',
  'swppx': 'Broad Market Equity', 'swtsx': 'Broad Market Equity',
  'spy': 'Broad Market Equity', 'qqq': 'Broad Market Equity',

  # Target Date Funds
  'vffsx': 'Target Date Fund', 'vdadx': 'Target Date Fund',
  'vforx': 'Target Date Fund', 'vthrx': 'Target Date Fund',
  'vtwax': 'Target Date Fund',
  'fitfx': 'Target Date Fund', 'fdfix': 'Target Date Fund',
  'ffwax': 'Target Date Fund', 'fdeex': 'Target Date Fund',

  # Bonds / Fixed Income
  'ftabx': 'Bonds / Fixed Income', 'bnd': 'Bonds / Fixed Income',
  'agg': 'Bonds / Fixed Income',

  # Sector / Thematic Funds
  'fuenx': 'Sector / Thematic Fund (Energy)',
  'flapx': 'Sector / Thematic Fund (Large Cap)',
  'flxsx': 'Sector / Thematic Fund (Large Cap)',
  'fdgrx': 'Sector / Thematic Fund (Large Growth)',

  # Cash & Equivalents
  'spaxx': 'Cash & Equivalents', 'vmfxx': 'Cash & Equivalents',
  'swvxx': 'Cash & Equivalents', 'fdrxx': 'Cash & Equivalents',
}
ASSET_CLASS_CASH_KEYWORDS = ["money market", "treasury bill", "govmmkt", "fgmxx", "t-bill"]
ASSET_CLASS_BOND_KEYWORDS = ["bond", "fixed income", "treasury", "government debt", "corporate debt", "aggregate bond",
                             "muni"]
ASSET_CLASS_COMMODITY_KEYWORDS = ["gold", "commodities", "gld"]
ASSET_CLASS_REAL_ESTATE_KEYWORDS = ["real estate", "reit"]
ASSET_CLASS_TARGET_DATE_KEYWORDS = ["target-date", "target retirement", "freedom index"]
ASSET_CLASS_BROAD_MARKET_KEYWORDS = ["large blend", "large cap", "s&p 500", "total stock market", "world stock"]


def get_user_symbol_mapping_input(original_symbol: str, context_msg: str) -> tuple[str | None, float | None, bool]:
  print("-" * 40)
  logging.warning(f"DATA FETCH FAILED for symbol: '{original_symbol}' (context: {context_msg})")
  print(f"Could not automatically fetch data for symbol: '{original_symbol}'")
  print("This might be an incorrect ticker, a delisted asset, or a non-ticker identifier.")
  print("Examples include CUSIPs (like '31617E471'), or internal fund numbers.")
  print("\nYou can provide an alternative mapping for this session and future runs:")
  print("  1. Enter a VALID Yahoo Finance ticker to use as a proxy (e.g., VTI, SPAXX).")
  print("  2. (If proxy) Enter a price factor (e.g., 1.0 if direct proxy, 0.1 if proxy is 10x value).")
  print("  3. Enter 'cash' to treat this symbol as a cash equivalent (price 1.0).")
  print("  4. Enter 'ignore' or just press Enter to skip this symbol (will assign 0 value).")
  mapped_symbol_input = input(f"  Proxy symbol for '{original_symbol}' (or 'cash'/'ignore'/Enter): ").strip().upper()
  if not mapped_symbol_input or mapped_symbol_input == 'IGNORE':
    print(f"'{original_symbol}' will be ignored and treated as having zero value.")
    print("-" * 40)
    return None, 1.0, False
  if mapped_symbol_input == 'CASH' or mapped_symbol_input in CASH_EQUIVALENT_SYMBOLS:
    print(f"'{original_symbol}' will be treated as CASH (price 1.0).")
    print("-" * 40)
    return "CASH", 1.0, True
  price_factor_str = input(
    f"  Enter price factor for '{original_symbol}' (using proxy '{mapped_symbol_input}') [default 1.0]: ").strip()
  price_factor = 1.0
  if price_factor_str:
    try:
      price_factor = float(price_factor_str)
    except ValueError:
      print(f"Invalid price factor '{price_factor_str}'. Defaulting to 1.0.")
  print(f"Mapping for '{original_symbol}': use '{mapped_symbol_input}' with factor {price_factor}.")
  print("-" * 40)
  return mapped_symbol_input, price_factor, False


def get_effective_symbol_info(original_symbol: str, context_msg: str) -> tuple[
  str, float, bool, bool]:
  original_symbol_upper = original_symbol.strip().upper()
  if original_symbol_upper in CASH_EQUIVALENT_SYMBOLS: return "CASH", 1.0, True, False
  if original_symbol_upper in SYMBOL_MAPPINGS:
    mapping = SYMBOL_MAPPINGS[original_symbol_upper]
    maps_to = mapping.get("maps_to", original_symbol_upper).upper()
    factor = float(mapping.get("factor", 1.0))
    is_cash = maps_to == "CASH" or maps_to in CASH_EQUIVALENT_SYMBOLS or mapping.get("treat_as_cash", False)
    if mapping.get("ignore_interactive"): logging.info(
      f"'{original_symbol_upper}' is set to be ignored based on prior input for {context_msg}."); return original_symbol_upper, 0.0, False, True
    if maps_to != original_symbol_upper or factor != 1.0 or is_cash: logging.info(
      f"Using pre-existing mapping for '{original_symbol_upper}': maps to '{maps_to}' with factor {factor} for {context_msg}.")
    return maps_to, factor, is_cash, False
  return original_symbol_upper, 1.0, False, False


def get_asset_details(symbol: str, rebuild_cache: bool = False, interactive_fallback: bool = True, max_retries: int = 3,
                      retry_delay_seconds: int = 2):
  original_symbol_cleaned = symbol.strip().upper()
  effective_symbol, price_factor, is_cash_equivalent, was_ignored = get_effective_symbol_info(original_symbol_cleaned,
                                                                                              "current details")
  # Return the effective symbol used for the lookup
  return_dict = {"price": 0.0, "type": "Ignored", "sector": "N/A", "name": original_symbol_cleaned,
                 "currency": "USD", "category": "N/A", "beta": None, "effective_symbol": effective_symbol}

  if was_ignored and price_factor == 0.0:
    return return_dict
  if is_cash_equivalent:
    name_for_cash = original_symbol_cleaned if original_symbol_cleaned != "CASH" else "Cash Holdings"
    if original_symbol_cleaned != effective_symbol: name_for_cash = f"{original_symbol_cleaned} (as Cash)"
    return {"price": 1.0 * price_factor, "type": "Cash", "sector": "Cash", "name": name_for_cash, "currency": "USD",
            "category": "Cash", "beta": 0.0, "effective_symbol": effective_symbol}

  cached_info_dict = get_cached_asset_info(effective_symbol, rebuild_cache=rebuild_cache)
  if cached_info_dict:
    if all(k in cached_info_dict for k in ["price", "type", "sector", "name", "currency", "category", "beta"]):
      logging.info(f"Asset details for {original_symbol_cleaned} (using {effective_symbol}): Using CACHED data.")
      cached_info_dict["price"] = float(cached_info_dict["price"]) * price_factor
      if cached_info_dict["name"] == effective_symbol and original_symbol_cleaned != effective_symbol:
        cached_info_dict["name"] = original_symbol_cleaned
      # Add the effective symbol to the cached dict before returning
      cached_info_dict["effective_symbol"] = effective_symbol
      return cached_info_dict
    logging.warning(
      f"Cached info for {effective_symbol} (from {original_symbol_cleaned}) is incomplete. Re-fetching.")

  logging.info(f"Asset details for {original_symbol_cleaned} (trying {effective_symbol}): Fetching from API.")
  symbols_to_try = [effective_symbol]
  if '.' in effective_symbol and effective_symbol.upper() not in ['BF.B']: symbols_to_try.append(
    effective_symbol.replace('.', '-'))
  api_info_data, fetched_api_symbol = None, None
  current_retry_delay = retry_delay_seconds
  for attempt in range(max_retries):
    for sym_variant in symbols_to_try:
      logging.debug(
        f"Attempt {attempt + 1}/{max_retries} for info: {sym_variant} (original: {original_symbol_cleaned})")
      try:
        if sym_variant not in TICKER_CACHE: TICKER_CACHE[sym_variant] = yf.Ticker(sym_variant)
        ticker = TICKER_CACHE[sym_variant]
        current_info = ticker.info
        if current_info and any(k in current_info and current_info[k] is not None for k in
                                ['regularMarketPrice', 'currentPrice', 'navPrice', 'previousClose']):
          api_info_data, fetched_api_symbol = current_info, sym_variant
          logging.info(
            f"Successfully fetched info for API symbol '{sym_variant}' (original: '{original_symbol_cleaned}') on attempt {attempt + 1}.")
          break
        logging.debug(
          f"Info for '{sym_variant}' empty or lacked key price fields for {original_symbol_cleaned}.")
      except NETWORK_EXCEPTIONS as e_net:
        logging.warning(
          f"Network/API error on attempt {attempt + 1} for '{sym_variant}': {type(e_net).__name__} - {e_net}")
      except Exception as e_variant:
        logging.error(
          f"Unexpected error fetching info for '{sym_variant}': {type(e_variant).__name__} - {e_variant}")
        continue
      time.sleep(0.2)
    if api_info_data: break
    if attempt < max_retries - 1: logging.info(
      f"Waiting {current_retry_delay}s before next retry for {original_symbol_cleaned} info..."); time.sleep(
      current_retry_delay); current_retry_delay = min(current_retry_delay * 2, 30)
  if not api_info_data:
    if interactive_fallback:
      mapped_sym_input, factor_input, treat_as_cash = get_user_symbol_mapping_input(original_symbol_cleaned,
                                                                                    "current details")
      if mapped_sym_input:
        SYMBOL_MAPPINGS[original_symbol_cleaned] = {"maps_to": mapped_sym_input, "factor": factor_input,
                                                    "treat_as_cash": treat_as_cash}
        save_symbol_mappings(SYMBOL_MAPPINGS)
        return get_asset_details(original_symbol_cleaned, rebuild_cache, False, 1)
      SYMBOL_MAPPINGS[original_symbol_cleaned] = {"maps_to": original_symbol_cleaned, "factor": 0.0,
                                                  "treat_as_cash": False, "ignore_interactive": True}
      save_symbol_mappings(SYMBOL_MAPPINGS)
      return return_dict  # Return dict with original symbol as effective
    return_dict["type"] = "Unknown (API Fail)"
    return return_dict

  price = 0.0
  price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose', 'navPrice']
  for field in price_fields:
    if field in api_info_data and api_info_data[field] is not None:
      try:
        price = float(api_info_data[field])
        break
      except (ValueError, TypeError):
        continue
  if price == 0.0 or price is None:
    try:
      hist_ticker = yf.Ticker(fetched_api_symbol)
      hist = hist_ticker.history(period="2d")
      if not hist.empty and 'Close' in hist.columns and not hist['Close'].empty: price = float(
        hist['Close'].iloc[-1])
    except Exception as e_hist:
      logging.warning(f"Could not fetch hist for backup price for {fetched_api_symbol}: {e_hist}")
    if price == 0.0 or price is None: price = 0.0; logging.warning(
      f"Could not get valid price for {fetched_api_symbol}. Using 0.")

  asset_type = api_info_data.get("quoteType", "Other")
  asset_type = asset_type.title() if asset_type in ("EQUITY", "ETF", "MUTUALFUND") else asset_type
  sector = api_info_data.get("sector", "N/A")
  category = api_info_data.get("category", "N/A")
  beta = api_info_data.get("beta")
  name_from_api = api_info_data.get("longName", api_info_data.get("shortName"))
  name_to_use = original_symbol_cleaned if not name_from_api or name_from_api.upper() == fetched_api_symbol.upper() else name_from_api
  currency = api_info_data.get("currency", "USD")

  details_to_return = {"price": price * price_factor, "type": asset_type, "sector": sector, "name": name_to_use,
                       "currency": currency, "category": category, "beta": beta, "effective_symbol": effective_symbol}

  cacheable_details = {k: v for k, v in details_to_return.items() if k != 'effective_symbol'}
  if price_factor != 1.0 and price != 0.0: cacheable_details["price"] = price
  cache_asset_info(fetched_api_symbol, cacheable_details)
  return details_to_return


def get_historical_prices(
    symbols: list,
    period: str,
    end_date_dt: datetime,
    rebuild_cache: bool = False,
    interactive_fallback: bool = True,
    max_retries: int = 2,
    retry_delay_seconds: int = 5
) -> pd.DataFrame:
  if not symbols: return pd.DataFrame(index=pd.DatetimeIndex([])).astype(float)
  end_date_normalized = end_date_dt.replace(hour=0, minute=0, second=0, microsecond=0)
  end_date_cache_key_str = end_date_normalized.strftime('%Y-%m-%d')
  end_date_normalized_naive = end_date_normalized.astimezone(timezone.utc).replace(tzinfo=None)
  final_df_columns, symbols_to_fetch_api, symbol_info_map = [], [], {}
  for s_orig in symbols:
    s_orig_cleaned = s_orig.strip().upper()
    final_df_columns.append(s_orig_cleaned)
    eff_sym, factor, is_cash, was_ignored = get_effective_symbol_info(s_orig_cleaned,
                                                                      f"hist prices for {period} ending {end_date_cache_key_str}")
    symbol_info_map[s_orig_cleaned] = {"effective_symbol": eff_sym, "factor": factor, "is_cash": is_cash,
                                       "ignored": was_ignored}
    if not is_cash and not was_ignored and eff_sym not in symbols_to_fetch_api: symbols_to_fetch_api.append(eff_sym)
  if not symbols_to_fetch_api: return pd.DataFrame(columns=final_df_columns).astype(float)
  cached_df = get_cached_historical_prices(symbols_to_fetch_api, period, end_date_normalized_naive,
                                           rebuild_cache=rebuild_cache)
  if cached_df is not None:
    # REFACTORED BLOCK 1: Avoid PerformanceWarning by building a dict of series first.
    reconstructed_cols = {}
    for s_orig in final_df_columns:
      map_info = symbol_info_map[s_orig]
      eff_sym, factor = map_info["effective_symbol"], map_info["factor"]

      # Default to a column of NaNs if the symbol is cash, ignored, or data is missing
      col_data = np.nan
      if not map_info["is_cash"] and not map_info["ignored"] and eff_sym in cached_df.columns:
        col_data = cached_df[eff_sym] * factor
      reconstructed_cols[s_orig] = col_data

    # Create the DataFrame from the dictionary of series/arrays in one go.
    reconstructed_df = pd.DataFrame(reconstructed_cols, index=cached_df.index)
    return reconstructed_df.astype(float)

  logging.info(
    f"Hist prices for {symbols_to_fetch_api} (period: {period}, end: {end_date_cache_key_str}): Fetching API.")
  raw_fetched_data = pd.DataFrame()
  current_retry_delay = retry_delay_seconds
  for attempt in range(max_retries):
    symbols_still_to_try = [s for s in symbols_to_fetch_api if
                            s not in raw_fetched_data.columns or raw_fetched_data[s].isnull().all()]
    if not symbols_still_to_try: break
    logging.debug(f"API Attempt {attempt + 1}/{max_retries} for hist prices: {symbols_still_to_try}")
    try:
      dl_end_str = (end_date_normalized_naive + timedelta(days=1)).strftime('%Y-%m-%d')
      data = yf.download(symbols_still_to_try, period=period, end=dl_end_str, progress=False, auto_adjust=True,
                         group_by='ticker', timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
      if not data.empty:
        temp_dfs = []
        for sym_eff in symbols_still_to_try:
          df_sym_data = None
          if sym_eff in data and isinstance(data[sym_eff], pd.DataFrame) and 'Close' in data[sym_eff].columns:
            df_sym_data = data[sym_eff][['Close']].rename(columns={'Close': sym_eff})
          elif sym_eff.replace('.', '-') in data and isinstance(data[sym_eff.replace('.', '-')],
                                                                pd.DataFrame) and 'Close' in data[
            sym_eff.replace('.', '-')].columns:
            df_sym_data = data[sym_eff.replace('.', '-')][['Close']].rename(columns={'Close': sym_eff})
          if df_sym_data is not None and not df_sym_data[sym_eff].isnull().all(): temp_dfs.append(df_sym_data)
        if temp_dfs:
          batch_data = pd.concat(temp_dfs, axis=1)
          if raw_fetched_data.empty:
            raw_fetched_data = batch_data
          else:
            raw_fetched_data = raw_fetched_data.combine_first(batch_data)
      if all(s in raw_fetched_data.columns and not raw_fetched_data[s].isnull().all() for s in
             symbols_to_fetch_api): break
    except NETWORK_EXCEPTIONS as e_net:
      logging.warning(
        f"Network/API error attempt {attempt + 1} for {symbols_still_to_try}: {type(e_net).__name__} - {e_net}")
    except Exception as e:
      logging.error(
        f"Unexpected yf.download error attempt {attempt + 1} for {symbols_still_to_try}: {type(e).__name__} - {e}")
    if attempt < max_retries - 1 and any(
        s not in raw_fetched_data.columns or raw_fetched_data[s].isnull().all() for s in symbols_still_to_try):
      logging.info(f"Waiting {current_retry_delay}s before next hist. price retry...")
      time.sleep(current_retry_delay)
      current_retry_delay = min(current_retry_delay * 2, 60)
    elif attempt == max_retries - 1:
      logging.error(
        f"Failed hist prices after {max_retries} attempts: {[s for s in symbols_still_to_try if s not in raw_fetched_data.columns or raw_fetched_data[s].isnull().all()]}")
  symbols_needing_map_final = [s for s in symbols_to_fetch_api if
                               s not in raw_fetched_data.columns or raw_fetched_data[s].isnull().all()]
  if symbols_needing_map_final and interactive_fallback:
    for s_eff_failed in list(symbols_needing_map_final):
      orig_s = next((o for o, mi in symbol_info_map.items() if
                     mi["effective_symbol"] == s_eff_failed and not mi["is_cash"] and not mi["ignored"]),
                    s_eff_failed)
      if orig_s in SYMBOL_MAPPINGS and SYMBOL_MAPPINGS[orig_s].get("ignore_interactive"):
        if s_eff_failed not in raw_fetched_data.columns: raw_fetched_data[s_eff_failed] = np.nan; continue
      logging.info(f"Prompting for failed hist fetch: {orig_s} (tried as {s_eff_failed})")
      mapped_sym, factor, treat_cash = get_user_symbol_mapping_input(orig_s, f"hist for {period}")
      if mapped_sym:
        SYMBOL_MAPPINGS[orig_s] = {"maps_to": mapped_sym, "factor": factor, "treat_as_cash": treat_cash}
        save_symbol_mappings(SYMBOL_MAPPINGS)
        symbol_info_map[orig_s].update(
          {"effective_symbol": mapped_sym, "factor": factor, "is_cash": treat_cash})
        if not treat_cash:
          new_data = get_historical_prices([mapped_sym], period, end_date_normalized_naive, rebuild_cache,
                                           False, 1)
          if not new_data.empty and mapped_sym in new_data.columns:
            if raw_fetched_data.empty: raw_fetched_data = pd.DataFrame(index=new_data.index)
            raw_fetched_data[s_eff_failed] = new_data[mapped_sym].reindex(
              raw_fetched_data.index).ffill().bfill()
      else:
        SYMBOL_MAPPINGS[orig_s] = {"maps_to": orig_s, "factor": 1.0, "treat_as_cash": False,
                                   "ignore_interactive": True}
        save_symbol_mappings(SYMBOL_MAPPINGS)
        symbol_info_map[orig_s]["ignored"] = True
      if s_eff_failed not in raw_fetched_data.columns: raw_fetched_data[s_eff_failed] = np.nan
  if raw_fetched_data.empty and not any(sinfo['is_cash'] for sinfo in symbol_info_map.values()): return pd.DataFrame(
    columns=final_df_columns).astype(float)
  for s_eff in symbols_to_fetch_api:
    if s_eff not in raw_fetched_data.columns: raw_fetched_data[s_eff] = np.nan
  if not raw_fetched_data.empty:
    raw_fetched_data.index = pd.to_datetime(raw_fetched_data.index).tz_localize(None)
    raw_fetched_data = raw_fetched_data[raw_fetched_data.index <= end_date_normalized_naive]
    cache_historical_prices(symbols_to_fetch_api, period, end_date_normalized_naive, raw_fetched_data)

  # REFACTORED BLOCK 2: Avoid PerformanceWarning by building a dict of series first.
  final_cols_data = {}
  final_index = raw_fetched_data.index if not raw_fetched_data.empty else pd.DatetimeIndex([])

  for s_orig_cleaned in final_df_columns:
    map_info = symbol_info_map[s_orig_cleaned]
    eff_sym, factor = map_info["effective_symbol"], map_info["factor"]

    col_data = np.nan  # Default to NaN
    if not map_info["is_cash"] and not map_info["ignored"] and eff_sym in raw_fetched_data.columns:
      col_data = raw_fetched_data[eff_sym] * factor
    final_cols_data[s_orig_cleaned] = col_data

  final_result_df = pd.DataFrame(final_cols_data, index=final_index)
  return final_result_df.astype(float)


def determine_risk_level(effective_symbol: str, asset_type: str, sector: str, name: str, category: str,
                         beta: float | None) -> str:
  """
  Determines asset risk level using a hybrid data-driven and keyword-based approach.
  """
  asset_type_lower = str(asset_type).lower()

  if beta is not None and asset_type_lower in ("stock", "equity", "etf"):
    try:
      beta_val = float(beta)
      if beta_val >= 1.7: return "4H - Very High Risk (Aggressive)"
      if beta_val >= 1.3: return "4 - High Risk (Volatile)"
      if beta_val >= 1.0: return "3H - Med-High Risk (Above Avg Volatility)"
      if beta_val >= 0.8: return "3 - Medium Risk (Market Volatility)"
      if beta_val < 0.8: return "2 - Low Risk (Below Avg Volatility)"
    except (ValueError, TypeError):
      pass

  effective_symbol_lower = str(effective_symbol).lower()
  search_text = f"{effective_symbol_lower} {str(name).lower()} {str(sector).lower()} {str(category).lower()}"

  if asset_type_lower == "cash" or any(kw in search_text for kw in RISK_LEVEL_CASH_KEYWORDS):
    return "1 - Lowest Risk (Cash/Equiv.)"
  if any(kw in search_text for kw in RISK_LEVEL_BOND_KEYWORDS):
    return "2 - Low Risk (Bonds)"
  if asset_type_lower in ("stock", "equity"):
    return "4 - High Risk (Individual Stocks)"
  if any(kw in search_text for kw in RISK_LEVEL_TARGET_DATE_KEYWORDS):
    return "3M - Medium Risk (Other Equity Fund)"
  if any(kw in search_text for kw in RISK_LEVEL_BROAD_EQUITY_KEYWORDS):
    return "3 - Medium Risk (Broad Equity)"
  if any(kw in search_text for kw in RISK_LEVEL_SECTOR_THEMATIC_KEYWORDS):
    return "3H - Med-High Risk (Sector/Thematic Equity)"
  if any(kw in search_text for kw in RISK_LEVEL_COMMODITY_KEYWORDS):
    return "4 - High Risk (Commodities)"
  if asset_type_lower in ("etf", "mutual fund"):
    return "3H - Med-High Risk (Sector/Thematic Equity)"

  return "5 - Other/Unknown"


def process_portfolio_data(df_input: pd.DataFrame, rebuild_cache: bool = False,
                           interactive_fallback: bool = True) -> pd.DataFrame:
  df = df_input.copy()
  if 'Symbol' not in df.columns: logging.error(
    "'Symbol' column not found for process_portfolio_data."); return pd.DataFrame()
  df['Symbol'] = df['Symbol'].astype(str).str.upper()
  asset_details_list = []
  for symbol_upper in df["Symbol"].unique():
    details = get_asset_details(symbol_upper, rebuild_cache=rebuild_cache,
                                interactive_fallback=interactive_fallback)
    asset_details_list.append({"Symbol": symbol_upper, **details})
  if not asset_details_list:
    df_details = pd.DataFrame(
      columns=["Symbol", "price", "type", "sector", "name", "currency", "category", "beta", "effective_symbol"])
  else:
    df_details = pd.DataFrame(asset_details_list)
    if 'Symbol' in df_details.columns: df_details['Symbol'] = df_details['Symbol'].astype(str).str.upper()

  merged_df = pd.merge(df, df_details, on="Symbol", how="left", suffixes=('_orig', '_details'))
  for col_base in ["price", "type", "sector", "name", "currency", "category", "beta", "effective_symbol"]:
    col_details, col_orig = f"{col_base}_details", f"{col_base}_orig"
    if col_details in merged_df.columns:
      merged_df[col_base] = merged_df[col_details]
    elif col_orig in merged_df.columns and col_base not in merged_df.columns:
      merged_df[col_base] = merged_df[col_orig]
    elif col_base not in merged_df.columns:
      default_val = None
      if col_base == "price":
        default_val = 0.0
      elif col_base not in ["beta", "effective_symbol"]:
        default_val = "N/A"
      merged_df[col_base] = default_val
    if col_details in merged_df.columns: merged_df.drop(columns=[col_details], inplace=True, errors='ignore')
    if col_orig in merged_df.columns: merged_df.drop(columns=[col_orig], inplace=True, errors='ignore')

  merged_df['price'] = pd.to_numeric(merged_df.get('price'), errors='coerce').fillna(0.0)
  merged_df['Amount'] = pd.to_numeric(merged_df.get('Amount'), errors='coerce').fillna(0.0)
  merged_df['Cost'] = pd.to_numeric(merged_df.get('Cost'), errors='coerce').fillna(0.0)
  merged_df["Market Value"] = (merged_df["Amount"] * merged_df["price"]).astype(float)

  merged_df['Risk Level'] = merged_df.apply(
    lambda r: determine_risk_level(
      r.get('effective_symbol'), r.get('type', 'N/A'), r.get('sector', 'N/A'),
      r.get('name', 'N/A'), r.get('category', 'N/A'), r.get('beta')
    ), axis=1)

  # THE FIX: Call determine_asset_class with the effective_symbol, not the original symbol.
  merged_df['Asset Class'] = merged_df.apply(
    lambda r: determine_asset_class(
      r.get('effective_symbol'),  # This was r.get('Symbol')
      r.get('type', 'N/A'),
      r.get('sector', 'N/A'),
      r.get('name', 'N/A'),
      r.get('category', 'N/A')
    ), axis=1)

  final_cols = ["Account", "Symbol", "Amount", "Cost", "price", "Market Value", "name", "type", "sector",
                "category", "Risk Level", "Asset Class", "currency", "beta", "effective_symbol"]
  existing_cols = [c for c in final_cols if c in merged_df.columns]
  other_cols = [c for c in merged_df.columns if c not in existing_cols]
  return merged_df[existing_cols + other_cols]


def determine_asset_class(effective_symbol: str, asset_type: str, sector: str, name: str, category: str) -> str:
  """
  Determines a detailed asset class using a definitive, two-layer approach.
  """
  effective_symbol_lower = str(effective_symbol).lower()

  # Layer 1: Definitive Ticker-to-Class Mapping (Highest Priority)
  if effective_symbol_lower in ASSET_CLASS_TICKER_MAP:
    return ASSET_CLASS_TICKER_MAP[effective_symbol_lower]

  # Layer 2: Hierarchical Fallback for Unmapped Tickers
  asset_type_lower = str(asset_type).lower()
  search_text = f"{effective_symbol_lower} {str(name).lower()} {str(sector).lower()} {str(category).lower()}"

  if asset_type_lower == 'cash' or asset_type_lower == 'moneymarket' or any(
      kw in search_text for kw in ASSET_CLASS_CASH_KEYWORDS):
    return "Cash & Equivalents"
  if any(kw in search_text for kw in ASSET_CLASS_COMMODITY_KEYWORDS):
    return "Commodities"

  if asset_type_lower in ('stock', 'equity'):
    if sector and sector != 'N/A':
      if sector in ["Technology", "Communication Services"]: return "Technology & Communications (Stock)"
      if sector in ["Consumer Cyclical", "Consumer Defensive"]: return "Consumer Goods & Services (Stock)"
      return f"{sector} (Stock)"
    return "Individual Stock"

  if asset_type_lower in ('etf', 'mutual fund'):
    if any(kw in search_text for kw in ASSET_CLASS_BOND_KEYWORDS): return "Bonds / Fixed Income"
    if any(kw in search_text for kw in ASSET_CLASS_REAL_ESTATE_KEYWORDS): return "Real Estate"
    if any(kw in search_text for kw in ASSET_CLASS_TARGET_DATE_KEYWORDS): return "Target Date Fund"
    if any(kw in search_text for kw in ASSET_CLASS_BROAD_MARKET_KEYWORDS): return "Broad Market Equity"
    if sector and sector != 'N/A':
      if sector in ["Technology", "Communication Services"]: return "Technology & Communications (Fund)"
      if sector in ["Consumer Cyclical", "Consumer Defensive"]: return "Consumer Goods & Services (Fund)"
      return f"{sector} (Fund)"
    return "Sector / Thematic Fund"

  return "Other / Unknown"
