import pandas as pd
import yfinance as yf  # Keep this
import logging
import time
import numpy as np
from .cache_utils import (
    get_cached_asset_info, cache_asset_info,
    get_cached_historical_prices, cache_historical_prices,
    SYMBOL_MAPPINGS, save_symbol_mappings
)

# yfinance can sometimes raise errors that are subclasses of requests.exceptions
# For yfinance specific errors, they are usually available directly from yf
# or we might need to catch a more general yfinance exception if specific ones are not exposed.
# Common yfinance exceptions that might indicate transient issues or specific data problems:
# yf.Y διαφορε फाइनेंस (yf.errors.YFinanceError if available, or more specific ones)
# For now, let's broaden the yfinance exception catch or specify known ones.
try:
    import requests.exceptions

    # For yfinance, let's catch specific known errors if possible, or a general one.
    # YFPricesMissingError is a common one. YFNotImplementedError might be less common for retry.
    # It's safer to catch what yfinance actually raises for network-like issues.
    # Often, yfinance wraps underlying errors. Let's try to catch base requests errors and a general yf error for now.
    NETWORK_EXCEPTIONS = (
        requests.exceptions.RequestException,  # Catches ConnectionError, Timeout, HTTPError, etc.
        # Add specific yfinance errors if known to be transient or network-related
        # For example, yfinance might raise its own error that wraps a requests error.
        # Looking at yfinance source, many errors inherit from a base YFinanceError if defined.
        # If yf.YFinanceError exists:
        # yf.YFinanceError
        # For now, a broad approach for yfinance specific might be needed if detailed exceptions are not stable.
        # Let's assume for now requests.exceptions.RequestException covers most network issues yfinance might bubble up.
        # If yfinance wraps and re-raises, we might need to add yfinance's wrapper exception.
        # For example, if yfinance has yf.errors.YFinanceDownloadError
    )
    # Check if yfinance exposes a base error class, often good to catch
    if hasattr(yf, 'YFinanceError'):  # Check if yfinance has a base error class we can use
        NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS + (yf.YFinanceError,)
    elif hasattr(yf, 'shared') and hasattr(yf.shared,
                                           '_ERRORS') and 'YFNotImplementedError' in yf.shared._ERRORS:  # Old way
        NETWORK_EXCEPTIONS = NETWORK_EXCEPTIONS + (yf.shared._ERRORS['YFNotImplementedError'],)


except ImportError:
    # If requests is not an explicit dependency we can import for typing,
    # yfinance still uses it, so its exceptions will be raised.
    # This fallback is less specific.
    if hasattr(yf, 'YFinanceError'):
        NETWORK_EXCEPTIONS = (yf.YFinanceError,)
    elif hasattr(yf, 'shared') and hasattr(yf.shared, '_ERRORS') and 'YFNotImplementedError' in yf.shared._ERRORS:
        NETWORK_EXCEPTIONS = (yf.shared._ERRORS['YFNotImplementedError'],)
    else:
        NETWORK_EXCEPTIONS = (Exception,)  # Very broad fallback, use with caution
    logging.warning("Could not import requests.exceptions directly; network exception handling might be less specific.")

TICKER_CACHE = {}
CASH_EQUIVALENT_SYMBOLS = {"CASH", "SPAXX", "VMFXX", "SWVXX"}


# --- get_user_symbol_mapping_input and get_effective_symbol_info remain the same ---
# ... (copy from previous correct version) ...
def get_user_symbol_mapping_input(original_symbol: str, context_msg: str) -> tuple[str | None, float | None, bool]:
    print("-" * 40)
    logging.warning(f"DATA FETCH FAILED for symbol: '{original_symbol}' (context: {context_msg})")
    print(f"Could not automatically fetch data for symbol: '{original_symbol}'")
    print("This symbol might be incorrect, delisted, or require a specific format (e.g., 'BRK-B' instead of 'BRK.B').")
    print("You can provide an alternative mapping for this session and future runs:")
    print("  1. Enter a VALID Yahoo Finance ticker to use as a proxy (e.g., VTI, SPY).")
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


def get_effective_symbol_info(original_symbol: str, context_msg: str, interactive_fallback: bool) -> tuple[
    str, float, bool, bool]:
    original_symbol_upper = original_symbol.strip().upper()
    if original_symbol_upper in CASH_EQUIVALENT_SYMBOLS: return "CASH", 1.0, True, False
    if original_symbol_upper in SYMBOL_MAPPINGS:
        mapping = SYMBOL_MAPPINGS[original_symbol_upper]
        mapped_sym = mapping.get("maps_to", original_symbol_upper).upper()
        factor = float(mapping.get("factor", 1.0))
        is_cash = mapped_sym == "CASH" or mapped_sym in CASH_EQUIVALENT_SYMBOLS
        if mapping.get("ignore_interactive"):
            logging.info(f"'{original_symbol_upper}' is set to be ignored based on prior input for {context_msg}.")
            return original_symbol_upper, 0.0, False, True
        if mapped_sym != original_symbol_upper or factor != 1.0 or is_cash:
            logging.info(
                f"Using pre-existing mapping for '{original_symbol_upper}': maps to '{mapped_sym}' with factor {factor} for {context_msg}.")
        return mapped_sym, factor, is_cash, False
    return original_symbol_upper, 1.0, False, False


def get_asset_details(symbol: str, rebuild_cache: bool = False, interactive_fallback: bool = True, max_retries: int = 3,
                      retry_delay: int = 2):  # Added retry params
    original_symbol_cleaned = symbol.strip().upper()
    # ... (effective symbol, cash equivalent, ignore checks from previous version) ...
    effective_symbol, price_factor, is_cash_equivalent, was_ignored = get_effective_symbol_info(
        original_symbol_cleaned, "current details", interactive_fallback
        # interactive_fallback here doesn't control retry, only post-retry prompt
    )
    if was_ignored and price_factor == 0.0:
        return {"price": 0.0, "type": "Ignored", "sector": "N/A", "name": original_symbol_cleaned, "currency": "USD"}
    if is_cash_equivalent:
        name_for_cash = original_symbol_cleaned if original_symbol_cleaned != "CASH" else "Cash Holdings"
        if original_symbol_cleaned != effective_symbol: name_for_cash = f"{original_symbol_cleaned} (as Cash)"
        return {"price": 1.0 * price_factor, "type": "Cash", "sector": "Cash", "name": name_for_cash, "currency": "USD"}

    cached_info = get_cached_asset_info(effective_symbol, rebuild_cache=rebuild_cache)
    if cached_info:
        # ... (cache check logic as before) ...
        if all(k in cached_info for k in ["price", "type", "sector", "name", "currency"]):
            logging.info(f"Asset details for {original_symbol_cleaned} (using {effective_symbol}): Using CACHED data.")
            cached_info_copy = cached_info.copy()
            cached_info_copy["price"] = float(cached_info_copy["price"]) * price_factor
            if cached_info_copy["name"] == effective_symbol and original_symbol_cleaned != effective_symbol:
                cached_info_copy["name"] = original_symbol_cleaned
            return cached_info_copy

    logging.info(f"Asset details for {original_symbol_cleaned} (trying {effective_symbol}): Fetching from API.")
    symbols_to_try = [effective_symbol]
    if '.' in effective_symbol and effective_symbol.upper() not in ['BF.B']:
        symbols_to_try.append(effective_symbol.replace('.', '-'))

    api_info_data = None
    fetched_symbol_variant = None

    for attempt in range(max_retries):
        for sym_variant in symbols_to_try:
            logging.debug(
                f"Attempt {attempt + 1}/{max_retries} for info: {sym_variant} (original: {original_symbol_cleaned})")
            try:
                if sym_variant not in TICKER_CACHE: TICKER_CACHE[sym_variant] = yf.Ticker(sym_variant)
                ticker = TICKER_CACHE[sym_variant]
                current_info = ticker.info  # API CALL
                if current_info and any(
                        price_key in current_info and current_info[price_key] is not None for price_key in
                        ['regularMarketPrice', 'currentPrice', 'navPrice']):
                    api_info_data = current_info
                    fetched_symbol_variant = sym_variant
                    logging.info(
                        f"Successfully fetched info for API symbol '{sym_variant}' (original: '{original_symbol_cleaned}') on attempt {attempt + 1}.")
                    break  # Break from sym_variant loop
                else:
                    logging.debug(f"Info for '{sym_variant}' empty or lacked key price fields.")
            except NETWORK_EXCEPTIONS as e_net:  # Catch specific network/API errors for retry
                logging.warning(f"Network/API error on attempt {attempt + 1} for '{sym_variant}': {e_net}")
            except Exception as e_variant:  # Other unexpected errors during Ticker().info
                logging.error(f"Unexpected error fetching info for '{sym_variant}': {e_variant}")
                # For non-network errors, usually don't retry the same variant
                continue  # Try next variant if any
            time.sleep(0.1)  # Small delay even between variants

        if api_info_data: break  # Break from retry loop if successful
        if attempt < max_retries - 1:
            logging.info(f"Waiting {retry_delay}s before next retry for {original_symbol_cleaned} info...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    # --- Post-retry logic (user prompt, processing, caching) ---
    # ... (This part is the same as in your previous data_processing.py, starting from 'if not api_info_data:')
    # Make sure to replace 'symbol_upper' with 'original_symbol_cleaned' or 'fetched_symbol_variant' where appropriate.
    if not api_info_data:
        if interactive_fallback:
            logging.info(f"All API retries for '{original_symbol_cleaned}' failed. Prompting user for mapping.")
            mapped_sym_input, factor_input, treat_as_cash = get_user_symbol_mapping_input(original_symbol_cleaned,
                                                                                          "current details")
            if mapped_sym_input:
                SYMBOL_MAPPINGS[original_symbol_cleaned] = {"maps_to": mapped_sym_input, "factor": factor_input,
                                                            "treat_as_cash": treat_as_cash}
                save_symbol_mappings(SYMBOL_MAPPINGS)
                logging.info(f"Retrying get_asset_details for '{original_symbol_cleaned}' with new mapping.")
                return get_asset_details(original_symbol_cleaned, rebuild_cache, interactive_fallback=False,
                                         max_retries=1)  # Only one try with new mapping
            else:
                SYMBOL_MAPPINGS[original_symbol_cleaned] = {"maps_to": original_symbol_cleaned, "factor": 0.0,
                                                            "treat_as_cash": False, "ignore_interactive": True}
                save_symbol_mappings(SYMBOL_MAPPINGS)
                return {"price": 0.0, "type": "Ignored", "sector": "N/A", "name": original_symbol_cleaned,
                        "currency": "USD"}
        else:
            return {"price": 0.0, "type": "Unknown (API Fail)", "sector": "N/A", "name": original_symbol_cleaned,
                    "currency": "USD"}

    price = 0.0
    price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose', 'navPrice']
    for field in price_fields:
        if field in api_info_data and api_info_data[field] is not None:
            try:
                price = float(api_info_data[field]);
                break
            except (ValueError, TypeError):
                continue
    if price == 0.0 or price is None:
        try:
            hist_ticker_obj = yf.Ticker(fetched_symbol_variant)
            hist = hist_ticker_obj.history(period="2d")
            if not hist.empty and 'Close' in hist.columns and not hist['Close'].empty:
                price = float(hist['Close'].iloc[-1])
        except Exception as e_hist:
            logging.warning(f"Could not fetch history for backup price for {fetched_symbol_variant}: {e_hist}")
        if price == 0.0 or price is None: price = 0.0; logging.warning(
            f"Could not determine valid price for {fetched_symbol_variant}. Using 0.")
    asset_type_map = {"EQUITY": "Stock", "ETF": "ETF", "MUTUALFUND": "Mutual Fund"}
    asset_type = asset_type_map.get(api_info_data.get("quoteType"), "Other")
    sector = api_info_data.get("sector")
    if asset_type == "ETF" and (not sector or sector == "N/A"): sector = api_info_data.get("category",
                                                                                           "N/A ETF Category")
    name_from_api = api_info_data.get("longName", api_info_data.get("shortName"))
    name_to_use = original_symbol_cleaned if not name_from_api or name_from_api.upper() == fetched_symbol_variant.upper() else name_from_api
    currency = api_info_data.get("currency", "USD")
    details = {"price": price * price_factor, "type": asset_type, "sector": sector if sector else "N/A",
               "name": name_to_use, "currency": currency}
    cache_symbol_key = original_symbol_cleaned if original_symbol_cleaned in SYMBOL_MAPPINGS and \
                                                  SYMBOL_MAPPINGS[original_symbol_cleaned][
                                                      "maps_to"] == fetched_symbol_variant else fetched_symbol_variant
    cacheable_details = details.copy()
    if price_factor != 1.0 and price != 0: cacheable_details["price"] = price
    cache_asset_info(cache_symbol_key, cacheable_details)
    return details


def get_historical_prices(symbols: list, period: str = "1y", rebuild_cache: bool = False,
                          interactive_fallback: bool = True, max_retries: int = 2,
                          retry_delay: int = 3):  # Added retry params
    # ... (initial symbol processing and cache check as before) ...
    if not symbols: return pd.DataFrame()
    final_df_columns = []
    symbols_to_fetch_api = []
    symbol_info_map = {}

    for s_orig in symbols:
        s_orig_cleaned = s_orig.strip().upper()
        final_df_columns.append(s_orig_cleaned)
        effective_symbol, factor, is_cash, was_ignored = get_effective_symbol_info(s_orig_cleaned,
                                                                                   f"historical prices for period {period}",
                                                                                   interactive_fallback)
        symbol_info_map[s_orig_cleaned] = {"effective_symbol": effective_symbol, "factor": factor, "is_cash": is_cash,
                                           "ignored": was_ignored}
        if not is_cash and not was_ignored and effective_symbol not in symbols_to_fetch_api:
            symbols_to_fetch_api.append(effective_symbol)

    if not symbols_to_fetch_api:
        logging.info("No non-cash symbols to fetch historical prices for.")
        return pd.DataFrame(columns=final_df_columns)

    cached_df = get_cached_historical_prices(symbols_to_fetch_api, period, rebuild_cache=rebuild_cache)
    if cached_df is not None:
        logging.info(
            f"Historical prices for effective symbols {symbols_to_fetch_api} (period: {period}): Using CACHED data.")
        reconstructed_df = pd.DataFrame(index=cached_df.index)
        for s_orig in final_df_columns:
            map_info = symbol_info_map[s_orig]
            if map_info["is_cash"] or map_info["ignored"]: reconstructed_df[s_orig] = np.nan; continue
            eff_sym, factor = map_info["effective_symbol"], map_info["factor"]
            if eff_sym in cached_df.columns:
                reconstructed_df[s_orig] = cached_df[eff_sym] * factor
            else:
                reconstructed_df[s_orig] = np.nan
        return reconstructed_df

    logging.info(
        f"Historical prices for effective symbols {symbols_to_fetch_api} (period: {period}): Fetching from API.")

    raw_fetched_data = pd.DataFrame()
    symbols_successfully_fetched_effective = []  # Tracks effective symbols that succeeded

    current_retry_delay = retry_delay

    for attempt in range(max_retries):
        symbols_for_this_attempt = [s for s in symbols_to_fetch_api if s not in symbols_successfully_fetched_effective]
        if not symbols_for_this_attempt: break  # All fetched

        logging.debug(f"API Attempt {attempt + 1}/{max_retries} for historical prices: {symbols_for_this_attempt}")
        try:
            data = yf.download(symbols_for_this_attempt, period=period, progress=False, auto_adjust=True,
                               group_by='ticker')
            if not data.empty:
                for sym_eff in symbols_for_this_attempt:
                    data_for_sym = None
                    if sym_eff in data and isinstance(data[sym_eff], pd.DataFrame):
                        data_for_sym = data[sym_eff]
                    elif sym_eff.replace('.', '-') in data and isinstance(data[sym_eff.replace('.', '-')],
                                                                          pd.DataFrame):
                        data_for_sym = data[sym_eff.replace('.', '-')]

                    if data_for_sym is not None and 'Close' in data_for_sym.columns and not data_for_sym[
                        'Close'].isnull().all():
                        if raw_fetched_data.empty and not data_for_sym.index.empty: raw_fetched_data.index = data_for_sym.index

                        # Align current symbol's data to the main index before assigning
                        aligned_series = data_for_sym['Close'].reindex(
                            raw_fetched_data.index if not raw_fetched_data.empty else data_for_sym.index)
                        raw_fetched_data[sym_eff] = aligned_series

                        if sym_eff not in symbols_successfully_fetched_effective:
                            symbols_successfully_fetched_effective.append(sym_eff)

            # Check if all symbols for this attempt are now covered
            if all(s in symbols_successfully_fetched_effective for s in symbols_for_this_attempt):
                break  # Success for this batch

        except NETWORK_EXCEPTIONS as e_net:
            logging.warning(
                f"Network/API error on attempt {attempt + 1} for historical prices {symbols_for_this_attempt}: {e_net}")
        except Exception as e:
            logging.error(
                f"Unexpected error during yf.download for {symbols_for_this_attempt} on attempt {attempt + 1}: {e}")
            # For unexpected errors, maybe don't retry this batch further to avoid repeated strange errors
            # Or decide based on exception type

        if attempt < max_retries - 1 and any(
                s not in symbols_successfully_fetched_effective for s in symbols_for_this_attempt):
            logging.info(f"Waiting {current_retry_delay}s before next retry for remaining historical prices...")
            time.sleep(current_retry_delay)
            current_retry_delay *= 2  # Exponential backoff
        elif attempt == max_retries - 1:
            logging.error(
                f"Failed to fetch some historical prices after {max_retries} attempts: {[s for s in symbols_for_this_attempt if s not in symbols_successfully_fetched_effective]}")

    # Interactive fallback for symbols that are still not fetched
    symbols_still_needing_data = [s for s in symbols_to_fetch_api if s not in symbols_successfully_fetched_effective]
    if symbols_still_needing_data and interactive_fallback:
        # ... (similar interactive fallback logic as in get_asset_details,
        # but adapted for historical data. This might involve fetching for a newly mapped symbol
        # and then adding/merging its data into raw_fetched_data under the original s_eff_failed key)
        for s_eff_failed in list(symbols_still_needing_data):  # Iterate on copy
            original_for_failed = next((orig for orig, map_info in symbol_info_map.items() if
                                        map_info["effective_symbol"] == s_eff_failed and not map_info["is_cash"] and not
                                        map_info["ignored"]), s_eff_failed)
            if original_for_failed in SYMBOL_MAPPINGS and SYMBOL_MAPPINGS[original_for_failed].get(
                    "ignore_interactive"):
                if s_eff_failed not in raw_fetched_data.columns: raw_fetched_data[s_eff_failed] = np.nan
                symbols_still_needing_data.remove(s_eff_failed);
                continue

            logging.info(f"Prompting for failed historical fetch: {original_for_failed} (tried as {s_eff_failed})")
            mapped_sym_input, factor_input, treat_as_cash = get_user_symbol_mapping_input(original_for_failed,
                                                                                          f"historical for period {period}")

            if mapped_sym_input:
                SYMBOL_MAPPINGS[original_for_failed] = {"maps_to": mapped_sym_input, "factor": factor_input,
                                                        "treat_as_cash": treat_as_cash}
                save_symbol_mappings(SYMBOL_MAPPINGS)
                symbol_info_map[original_for_failed].update(
                    {"effective_symbol": mapped_sym_input, "factor": factor_input, "is_cash": treat_as_cash})

                if not treat_as_cash:
                    new_data = get_historical_prices([mapped_sym_input], period=period, rebuild_cache=rebuild_cache,
                                                     interactive_fallback=False, max_retries=1)
                    if not new_data.empty and mapped_sym_input in new_data.columns:
                        if raw_fetched_data.empty: raw_fetched_data.index = new_data.index
                        aligned_new_data = new_data[mapped_sym_input].reindex(raw_fetched_data.index).ffill().bfill()
                        raw_fetched_data[s_eff_failed] = aligned_new_data  # Data for the original s_eff_failed
                        if s_eff_failed in symbols_still_needing_data: symbols_still_needing_data.remove(s_eff_failed)
                    else:
                        if s_eff_failed not in raw_fetched_data.columns: raw_fetched_data[s_eff_failed] = np.nan
                elif s_eff_failed in symbols_still_needing_data:
                    symbols_still_needing_data.remove(s_eff_failed)  # Mapped to cash
            else:  # User chose to ignore
                SYMBOL_MAPPINGS[original_for_failed] = {"maps_to": original_for_failed, "factor": 1.0,
                                                        "treat_as_cash": False, "ignore_interactive": True}
                save_symbol_mappings(SYMBOL_MAPPINGS)
                symbol_info_map[original_for_failed]["ignored"] = True
                if s_eff_failed not in raw_fetched_data.columns: raw_fetched_data[s_eff_failed] = np.nan
                if s_eff_failed in symbols_still_needing_data: symbols_still_needing_data.remove(s_eff_failed)

    if raw_fetched_data.empty and not any(sinfo['is_cash'] for sinfo in symbol_info_map.values()):
        return pd.DataFrame(columns=final_df_columns)

    for s_eff in symbols_to_fetch_api:  # Ensure all columns exist, even if all NaN
        if s_eff not in raw_fetched_data.columns: raw_fetched_data[s_eff] = np.nan

    if not raw_fetched_data.empty:
        raw_fetched_data.index = pd.to_datetime(raw_fetched_data.index)
        cache_historical_prices(symbols_to_fetch_api, period, raw_fetched_data)  # Cache raw data by effective symbols

    final_result_df = pd.DataFrame(index=raw_fetched_data.index if not raw_fetched_data.empty else None,
                                   columns=final_df_columns)
    for s_orig_cleaned in final_df_columns:
        map_info = symbol_info_map[s_orig_cleaned]
        if map_info["is_cash"] or map_info["ignored"]: final_result_df[s_orig_cleaned] = np.nan; continue
        eff_sym, factor = map_info["effective_symbol"], map_info["factor"]
        if eff_sym in raw_fetched_data.columns:
            final_result_df[s_orig_cleaned] = raw_fetched_data[eff_sym] * factor
        else:
            final_result_df[s_orig_cleaned] = np.nan
    return final_result_df


def determine_risk_level(asset_type: str, sector: str, name: str) -> str:
    """
    Determines a simplified risk level based on asset type, sector, and name.
    """
    asset_type_lower = str(asset_type).lower()
    sector_lower = str(sector).lower()
    name_lower = str(name).lower()

    if asset_type_lower == "cash" or any(cash_eq in name_lower for cash_eq in ["money market", "spaxx", "vmfxx"]):
        return "1 - Lowest Risk (Cash/Equiv.)"

    # Bonds / Fixed Income identification (can be expanded)
    bond_keywords = ["bond", "fixed income", "treasury", "government debt", "corporate debt", "aggregate bond", "muni"]
    if asset_type_lower == "etf" or asset_type_lower == "mutual fund":
        if any(keyword in name_lower for keyword in bond_keywords) or \
                any(keyword in sector_lower for keyword in bond_keywords):  # Sector for ETFs might be 'Fixed Income'
            return "2 - Low Risk (Bonds)"
        # Check for specific bond ETF/MF categories if sector is less descriptive
        if "fixed income" in sector_lower or "bond" in sector_lower:  # Common yfinance categories
            return "2 - Low Risk (Bonds)"
        if "diversified portfolio" in sector_lower and any(
                keyword in name_lower for keyword in bond_keywords):  # Some "diversified portfolio" ETFs are bond heavy
            return "2 - Low Risk (Bonds)"

    if "gold" in name_lower or "gold" in sector_lower or "commodities focused" in sector_lower and "gold" in name_lower:
        return "4 - High Risk (Commodities)"  # Or a specific "Commodities" category

    if asset_type_lower == "stock":
        return "4 - High Risk (Individual Stocks)"

    if asset_type_lower == "etf" or asset_type_lower == "mutual fund":
        broad_market_keywords = ["total stock market", "s&p 500", "large cap", "large blend", "russell 1000",
                                 "russell 3000", "developed markets", "world stock", "global equity", "broad market"]
        if any(keyword in name_lower for keyword in broad_market_keywords) or \
                any(keyword in sector_lower for keyword in broad_market_keywords):
            return "3 - Medium Risk (Broad Equity)"

        # More specific sector ETFs or thematic ETFs could be higher risk
        # These are general heuristics and can be refined.
        # If sector is specific like 'technology', 'healthcare', 'financial services', etc.
        if sector_lower not in ["n/a", "n/a etf category", "diversified portfolio",
                                "miscellaneous sector"] and "blend" not in sector_lower and "cap" not in sector_lower:
            return "3H - Med-High Risk (Sector/Thematic Equity)"

        # Default for other ETFs/MFs if not caught above
        return "3M - Medium Risk (Other Equity Fund)"

    return "5 - Other/Unknown"


def process_portfolio_data(df_input: pd.DataFrame, rebuild_cache: bool = False,
                           interactive_fallback: bool = True) -> pd.DataFrame:
    df = df_input.copy()
    if 'Symbol' not in df.columns:
        logging.error("'Symbol' column not found in input DataFrame for process_portfolio_data.")
        return pd.DataFrame()
    df['Symbol'] = df['Symbol'].astype(str).str.upper()

    asset_details_list = []
    unique_symbols = df["Symbol"].unique()
    for symbol_upper in unique_symbols:
        details = get_asset_details(symbol_upper, rebuild_cache=rebuild_cache,
                                    interactive_fallback=interactive_fallback)
        asset_details_list.append({"Symbol": symbol_upper, **details})

    if not asset_details_list:
        df_details = pd.DataFrame(columns=["Symbol", "price", "type", "sector", "name", "currency"])
    else:
        df_details = pd.DataFrame(asset_details_list)
        if 'Symbol' in df_details.columns:
            df_details['Symbol'] = df_details['Symbol'].astype(str).str.upper()

    merged_df = pd.merge(df, df_details, on="Symbol", how="left")

    # Consolidate price
    if 'price_y' in merged_df.columns:
        merged_df['price'] = merged_df['price_y']
        merged_df.drop(columns=['price_y'], inplace=True, errors='ignore')
        if 'price_x' in merged_df.columns: merged_df.drop(columns=['price_x'], inplace=True, errors='ignore')
    elif 'price_x' in merged_df.columns:
        merged_df['price'] = merged_df['price_x']
        merged_df.drop(columns=['price_x'], inplace=True, errors='ignore')
    elif 'price' not in merged_df.columns:
        merged_df['price'] = 0.0
    merged_df['price'] = pd.to_numeric(merged_df['price'], errors='coerce').fillna(0.0)

    # Consolidate other string columns
    string_cols_from_details = ["type", "sector", "name", "currency"]
    for col in string_cols_from_details:
        col_y, col_x = f"{col}_y", f"{col}_x"
        if col_y in merged_df.columns:
            merged_df[col] = merged_df[col_y]
            if col_x in merged_df.columns: merged_df.drop(columns=[col_x], inplace=True, errors='ignore')
            merged_df.drop(columns=[col_y], inplace=True, errors='ignore')
        elif col_x in merged_df.columns and col not in merged_df.columns:
            merged_df[col] = merged_df[col_x]
            merged_df.drop(columns=[col_x], inplace=True, errors='ignore')
        elif col not in merged_df.columns:
            merged_df[col] = "N/A"
        merged_df[col] = merged_df[col].fillna("N/A")

    # Ensure 'Amount' and 'Cost' are numeric
    for col_name in ['Amount', 'Cost']:
        if col_name not in merged_df.columns:
            merged_df[col_name] = 0.0
        else:
            merged_df[col_name] = pd.to_numeric(merged_df[col_name], errors='coerce').fillna(0.0)

    merged_df["Market Value"] = (merged_df["Amount"] * merged_df["price"]).astype(float)

    # --- ADD RISK LEVEL ---
    merged_df['Risk Level'] = merged_df.apply(
        lambda row: determine_risk_level(row.get('type', 'N/A'), row.get('sector', 'N/A'), row.get('name', 'N/A')),
        axis=1
    )

    final_columns_order = ["Account", "Symbol", "Amount", "Cost", "price", "Market Value", "name", "type", "sector",
                           "Risk Level", "currency"]
    existing_final_columns = [col for col in final_columns_order if col in merged_df.columns]
    # Add any other columns that might exist due to unforeseen suffixes, though ideally handled above
    other_cols = [col for col in merged_df.columns if col not in existing_final_columns]
    return merged_df[existing_final_columns + other_cols]
