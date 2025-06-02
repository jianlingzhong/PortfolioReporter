# portfolio_analyzer/section_builders.py

# Configure logging for this module if not configured globally

# portfolio_analyzer/section_builders.py
import pandas as pd
import numpy as np
import logging
import re
import plotly.graph_objects as go
from datetime import datetime, timezone
from .report_components import create_bar_chart_figure, create_html_table
from .data_processing import get_historical_prices, get_effective_symbol_info

logger = logging.getLogger(__name__)  # Keep this commented unless you need module-specific logger


# --- build_asset_allocation_section and build_holdings_summary_section ---
# (These remain the same from your last correct version)
def build_asset_allocation_section(processed_df: pd.DataFrame) -> dict:
    """Builds the Asset Allocation Overview section dictionary using bar charts."""
    logging.info("Building asset allocation section (using bar charts)...")
    charts_html_list = []

    # Chart 1: Allocation by Asset Type
    alloc_type_df = processed_df.groupby("type")["Market Value"].sum().reset_index()
    alloc_type_df = alloc_type_df[alloc_type_df["Market Value"] > 0].sort_values(by="Market Value",
                                                                                 ascending=True)  # Ascending for horizontal bar

    if not alloc_type_df.empty:
        fig_alloc_type = create_bar_chart_figure(
            df_grouped=alloc_type_df,
            x_col="type",  # Categories on Y-axis
            y_col="Market Value",  # Values on X-axis
            title="Asset Allocation by Type",
            x_axis_title=None,  # Asset Type will be y-axis label
            y_axis_title="Market Value (USD)",
            orientation='h'
        )
        fig_alloc_type.update_layout(
            margin=dict(t=50, b=40, l=120, r=40),  # Increased left margin for type labels
            yaxis=dict(title_text="Asset Type")  # Explicitly set y-axis title for horizontal
        )
        charts_html_list.append(fig_alloc_type.to_html(
            full_html=False, include_plotlyjs=False, div_id="chart-alloc-type", config={'responsive': True}
        ))

    # Chart 2: Allocation by Sector
    sector_allocation_df_source = processed_df[
        ~processed_df["type"].isin(["Cash", "Mutual Fund", "Other", "Unknown"]) &
        (processed_df["sector"].notna()) & (processed_df["sector"] != "N/A") &
        (processed_df["sector"] != "N/A ETF Category")
        ]
    if not sector_allocation_df_source.empty:
        alloc_sector_df = sector_allocation_df_source.groupby("sector")["Market Value"].sum().reset_index()
        alloc_sector_df = alloc_sector_df[alloc_sector_df["Market Value"] > 0].sort_values(by="Market Value",
                                                                                           ascending=False)

        if not alloc_sector_df.empty:
            fig_alloc_sector = create_bar_chart_figure(
                df_grouped=alloc_sector_df,
                x_col="sector",
                y_col="Market Value",
                title="Stock & ETF Allocation by Sector",
                x_axis_title="Sector",
                y_axis_title="Market Value (USD)",
                orientation='v'
            )
            fig_alloc_sector.update_layout(
                xaxis_tickangle=-45,
                margin=dict(t=50, b=120, l=60, r=20)  # Increased bottom margin
            )
            charts_html_list.append(fig_alloc_sector.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-alloc-sector", config={'responsive': True}
            ))

    return {"type": "charts_vertical", "charts": charts_html_list}


# --- MODIFIED calculate_net_worth_and_spy_benchmark ---
def calculate_net_worth_and_spy_benchmark(
        portfolio_df: pd.DataFrame,
        period: str,
        rebuild_cache: bool = False,
        interactive_fallback: bool = True
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Calculates portfolio net worth and a SPY benchmark for a single period.
    The 'period' string determines how far back from 'today' data is fetched.
    """
    # ... (holdings, symbols_to_fetch, cash_amount logic - same) ...
    holdings = portfolio_df.groupby('Symbol')['Amount'].sum().reset_index()
    symbols_to_fetch = holdings[holdings['Symbol'] != 'CASH']['Symbol'].tolist()
    cash_holdings = holdings[holdings['Symbol'] == 'CASH']
    cash_amount = cash_holdings['Amount'].sum() if not cash_holdings.empty else 0.0

    yf_fetch_period_arg = "max" if period.lower() == "all" else period  # yfinance argument
    # Determine the target end date for this period calculation (usually today)
    target_end_date_for_period = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    portfolio_historical_prices_df = pd.DataFrame()  # This will hold prices for effective symbols
    if symbols_to_fetch:
        portfolio_historical_prices_df = get_historical_prices(
            symbols_to_fetch,
            period=yf_fetch_period_arg,  # Use the string like "1y", "max"
            end_date_dt=target_end_date_for_period,  # Pass the calculated end date
            rebuild_cache=rebuild_cache,
            interactive_fallback=interactive_fallback
        )
        if portfolio_historical_prices_df.empty and cash_amount == 0:
            logging.warning(f"Period {period}: Failed to fetch historical prices for portfolio and no cash.")
            return None, None

    # Determine date_index based on fetched data or cash-only scenario
    date_index = None
    if not portfolio_historical_prices_df.empty and portfolio_historical_prices_df.index.is_monotonic_increasing:
        date_index = portfolio_historical_prices_df.index  # This index is from yf.download for the given period up to target_end_date
    elif cash_amount > 0:
        # This logic is for when only cash exists, or ALL symbol fetches failed.
        # We need to construct a date_index based on the period string relative to target_end_date_for_period
        today_for_index = target_end_date_for_period  # Use the consistent end date
        num_periods_map = {"max": 252 * 5, "5y": 252 * 5, "1y": 252, "6mo": 126, "3mo": 63, "1mo": 21, "ytd": None}
        effective_period_key_for_index = period.lower() if period.lower() in num_periods_map else "1y"
        num_periods = num_periods_map.get(effective_period_key_for_index)

        if effective_period_key_for_index == "ytd":
            start_of_year = pd.Timestamp(year=today_for_index.year, month=1, day=1, tzinfo=timezone.utc)
            date_index = pd.bdate_range(start_of_year, today_for_index)  # Ensure end date is inclusive
        elif num_periods:
            # date_range 'end' is inclusive. 'periods' counts backwards from 'end'.
            date_index = pd.date_range(end=today_for_index, periods=num_periods, freq='B')
        else:  # Fallback for "max" or other unmapped for cash-only (e.g. 5 years)
            date_index = pd.date_range(end=today_for_index, periods=num_periods_map["5y"], freq='B')

        if date_index.empty: date_index = pd.DatetimeIndex([today_for_index])
        date_index = date_index.tz_localize(None)  # Make naive for consistency
    else:
        logging.warning(
            f"Period {period}: No portfolio assets with historical data or cash. Cannot determine date_index.")
        return None, None

    if date_index.tz is not None: date_index = date_index.tz_localize(None)

    historical_values_dict = {}
    for _, row in holdings.iterrows():
        symbol, amount = row['Symbol'], row['Amount']
        if symbol == 'CASH': continue

        temp_series = pd.Series(0.0, index=date_index, name=symbol)
        # portfolio_historical_prices_df columns are original symbols after reconstruction in get_historical_prices
        if not portfolio_historical_prices_df.empty and symbol in portfolio_historical_prices_df.columns:
            asset_prices = portfolio_historical_prices_df[symbol].reindex(date_index).ffill().bfill()
            if not asset_prices.isnull().all(): temp_series = asset_prices * amount
        historical_values_dict[symbol] = temp_series

    historical_values_dict['CASH_TOTAL'] = pd.Series(cash_amount, index=date_index)
    all_historical_values = pd.DataFrame(historical_values_dict).fillna(0.0)

    portfolio_net_worth_series = all_historical_values.sum(axis=1)
    # ... (rest of portfolio_net_worth_df creation and SPY benchmark calculation)
    # Ensure SPY also uses target_end_date_for_period for its get_historical_prices call
    if portfolio_net_worth_series.empty: return None, None
    is_constant = portfolio_net_worth_series.nunique() <= 1
    is_zero = portfolio_net_worth_series.iloc[0] == 0.0 if not portfolio_net_worth_series.empty else True
    if is_constant and is_zero and not (cash_amount > 0 and not symbols_to_fetch): return None, None
    portfolio_net_worth_df = portfolio_net_worth_series.reset_index()
    portfolio_net_worth_df.columns = ['Date', 'Net Worth']
    portfolio_net_worth_df['Net Worth'] = pd.to_numeric(portfolio_net_worth_df['Net Worth'], errors='coerce').fillna(
        0.0)

    spy_prices_df = get_historical_prices(
        ["SPY"], period=yf_fetch_period_arg, end_date_dt=target_end_date_for_period,  # Pass end_date
        rebuild_cache=rebuild_cache, interactive_fallback=interactive_fallback
    )
    if spy_prices_df.empty or 'SPY' not in spy_prices_df.columns: return portfolio_net_worth_df, None
    spy_prices_aligned = spy_prices_df['SPY'].reindex(date_index).ffill().bfill()
    if spy_prices_aligned.isnull().all(): return portfolio_net_worth_df, None
    if portfolio_net_worth_df.empty or spy_prices_aligned.empty: return portfolio_net_worth_df, None
    initial_portfolio_net_worth = portfolio_net_worth_df['Net Worth'].iloc[0]
    initial_spy_price = spy_prices_aligned.iloc[0]
    if pd.isna(initial_portfolio_net_worth) or pd.isna(
            initial_spy_price) or initial_spy_price == 0: return portfolio_net_worth_df, None
    hypothetical_spy_shares = initial_portfolio_net_worth / initial_spy_price
    spy_benchmark_values = spy_prices_aligned * hypothetical_spy_shares
    spy_benchmark_df = spy_benchmark_values.reset_index()
    spy_benchmark_df.columns = ['Date', 'SPY Benchmark']
    spy_benchmark_df['SPY Benchmark'] = pd.to_numeric(spy_benchmark_df['SPY Benchmark'], errors='coerce').fillna(0.0)
    return portfolio_net_worth_df, spy_benchmark_df


def build_historical_networth_section(
        portfolio_df: pd.DataFrame, default_period: str = "1y",
        rebuild_cache: bool = False, interactive_fallback: bool = True
) -> dict:
    # ... (This function uses calculate_net_worth_and_spy_benchmark, which now handles end_date implicitly via get_historical_prices)
    # ... (The rest of this function should be okay from the previous version) ...
    logging.info(
        f"Building historical net worth section (rebuild_cache={rebuild_cache}, interactive={interactive_fallback}), default: {default_period}...")
    time_periods_map = {"3mo": "3mo", "6mo": "6mo", "1y": "1y", "5y": "5y", "All": "max"}
    display_periods = list(time_periods_map.keys())
    active_display_period = default_period
    if default_period not in display_periods:
        found_default = False
        for disp_key, yf_val in time_periods_map.items():
            if yf_val == default_period: active_display_period = disp_key; found_default = True; break
        if not found_default: display_periods.append(default_period); time_periods_map[
            default_period] = default_period; active_display_period = default_period
    fig = go.Figure()
    period_data_dict = {}
    at_least_one_valid_plot = False
    for display_name, yf_period_val in time_periods_map.items():
        portfolio_df_period, spy_df_period = calculate_net_worth_and_spy_benchmark(portfolio_df, yf_period_val,
                                                                                   rebuild_cache=rebuild_cache,
                                                                                   interactive_fallback=interactive_fallback)
        if portfolio_df_period is not None and not portfolio_df_period.empty:
            period_data_dict[display_name] = {'portfolio': portfolio_df_period, 'spy': spy_df_period}
            at_least_one_valid_plot = True
        else:
            logging.warning(f"No valid net worth data for period: {display_name} (yf: {yf_period_val})")
    if not at_least_one_valid_plot: return {"type": "html_content",
                                            "html": "<p>Insufficient historical data for any selected time periods.</p>"}
    default_display_name_for_visibility = active_display_period
    try:
        active_button_index = display_periods.index(default_display_name_for_visibility)
        if default_display_name_for_visibility not in period_data_dict:
            for i, dn in enumerate(display_periods):
                if dn in period_data_dict: active_button_index = i; default_display_name_for_visibility = dn; break
            else:
                active_button_index = 0
    except ValueError:
        active_button_index = 0
        if display_periods and display_periods[0] in period_data_dict:
            default_display_name_for_visibility = display_periods[0]
        else:
            default_display_name_for_visibility = "1y" if "1y" in period_data_dict else (
                next(iter(period_data_dict)) if period_data_dict else "N/A")
    for display_name in display_periods:  # Iterate using display names
        data = period_data_dict.get(display_name)
        if data:
            fig.add_trace(go.Scatter(x=data['portfolio']['Date'], y=data['portfolio']['Net Worth'], mode='lines',
                                     name='Portfolio', legendgroup=display_name,
                                     legendgrouptitle_text=display_name.upper(),
                                     visible=(display_name == default_display_name_for_visibility),
                                     line=dict(color='royalblue'),
                                     hovertemplate='%{x|%Y-%m-%d} (Portfolio): $%{y:,.2f}<extra></extra>'))
            if data['spy'] is not None and not data['spy'].empty:
                fig.add_trace(go.Scatter(x=data['spy']['Date'], y=data['spy']['SPY Benchmark'], mode='lines',
                                         name='SPY Benchmark', legendgroup=display_name,
                                         visible=(display_name == default_display_name_for_visibility),
                                         line=dict(color='orangered'),
                                         hovertemplate='%{x|%Y-%m-%d} (SPY): $%{y:,.2f}<extra></extra>'))
    buttons = []
    for display_name_button in display_periods:
        if display_name_button in period_data_dict:
            visibility = [False] * len(fig.data)
            for i, trace_data in enumerate(fig.data):
                if trace_data.legendgroup == display_name_button: visibility[i] = True
            buttons.append(dict(label=display_name_button.upper(), method="update", args=[{"visible": visibility}, {
                "title.text": f"Historical Net Worth ({display_name_button.upper()})",
                "legend.title.text": display_name_button.upper()}]))
    fig.update_layout(updatemenus=[
        dict(active=active_button_index, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True,
             x=0.0, xanchor="left", y=1.16, yanchor="top")],
        title_text=f"Historical Net Worth ({default_display_name_for_visibility.upper()})", title_x=0.5,
        xaxis_title=None, yaxis_title="Value (USD)", margin=dict(t=100, b=20, l=60, r=20), autosize=True,
        width=None, height=None, yaxis_autorange=True,
        legend_title_text=default_display_name_for_visibility.upper(), hovermode="x unified")
    return {"type": "charts_vertical", "charts": [
        fig.to_html(full_html=False, include_plotlyjs=True, div_id="chart-net-worth", config={'responsive': True})]}


def get_net_worth_at_previous_day(portfolio_df: pd.DataFrame, days_ago: int,
                                  base_historical_df: pd.DataFrame | None) -> float | None:
    if days_ago <= 0: return None
    if base_historical_df is None or base_historical_df.empty:
        logging.warning(f"No base historical data provided for change calc ({days_ago} days ago).")
        return None
    # base_historical_df should already be sorted by Date descending from its creation in build_key_metrics_section
    if len(base_historical_df) > days_ago:
        return base_historical_df['Net Worth'].iloc[days_ago]
    else:
        logging.warning(f"Not enough hist data points ({len(base_historical_df)}) for value {days_ago} days ago.")
        return None


def build_key_metrics_section(
        current_portfolio_processed_df: pd.DataFrame,
        original_portfolio_input_df: pd.DataFrame,
        rebuild_cache: bool = False,
        interactive_fallback: bool = True
) -> dict:
    # ... (Logic from your previous correct version, ensure calculate_net_worth_and_spy_benchmark calls pass interactive_fallback & rebuild_cache) ...
    # ... and get_net_worth_at_previous_day is called with the pre-fetched base_historical_nw_df.
    logging.info(f"Building key metrics section (rebuild_cache={rebuild_cache}, interactive={interactive_fallback})...")
    metrics = []
    current_net_worth = current_portfolio_processed_df['Market Value'].sum()
    metrics.append(
        {"label": "Current Net Worth", "value": f"${current_net_worth:,.2f}", "sub_value": None, "change_pct": None,
         "change_abs": None, "positive_change": None, "icon": "💰"})
    total_cost_basis = 0.0
    if 'Cost' in current_portfolio_processed_df.columns and pd.api.types.is_numeric_dtype(
            current_portfolio_processed_df['Cost']):
        total_cost_basis = current_portfolio_processed_df['Cost'].sum()
    metrics.append(
        {"label": "Total Cost Basis", "value": f"${total_cost_basis:,.2f}", "sub_value": None, "change_pct": None,
         "change_abs": None, "positive_change": None, "icon": "🧾"})
    total_gain_loss_abs = current_net_worth - total_cost_basis
    is_total_gain_positive = total_gain_loss_abs >= 0
    total_gain_loss_pct_str = "(N/A)"
    if total_cost_basis != 0:
        total_gain_loss_pct = (total_gain_loss_abs / total_cost_basis) * 100
        total_gain_loss_pct_str = f"{'+' if is_total_gain_positive else ''}{total_gain_loss_pct:.2f}%"
    metrics.append({"label": "Total Gain/Loss", "value": None, "sub_value": None,
                    "change_abs": f"{'+' if is_total_gain_positive else ''}${total_gain_loss_abs:,.2f}",
                    "change_pct": total_gain_loss_pct_str, "positive_change": is_total_gain_positive,
                    "icon": "📈" if is_total_gain_positive else "📉"})

    change_definitions = {"1-Day Change": (1, "5d"), "5-Day Change": (5, "10d"), "20-Day Change": (20, "2mo"),
                          "3-Month Change": (63, "4mo"), "6-Month Change": (126, "7mo"), "1-Year Change": (252, "13mo")}
    max_fetch_days = 0
    longest_yf_period_for_changes = "5d"
    period_duration_map = {"d": 1, "mo": 21, "y": 252}
    for _, (days, yf_p_str) in change_definitions.items():
        num_part_match = re.search(r'\d+', yf_p_str)
        unit_part_match = re.search(r'[a-zA-Z]+', yf_p_str)
        if num_part_match and unit_part_match:
            num_part, unit_part = int(num_part_match.group()), unit_part_match.group()
            current_fetch_days = num_part * period_duration_map.get(unit_part, 1)
            if current_fetch_days > max_fetch_days: max_fetch_days, longest_yf_period_for_changes = current_fetch_days, yf_p_str
    logging.info(f"Fetching historical data for KPI changes using period: {longest_yf_period_for_changes}")
    base_historical_nw_df_for_kpi, _ = calculate_net_worth_and_spy_benchmark(original_portfolio_input_df,
                                                                             period=longest_yf_period_for_changes,
                                                                             rebuild_cache=rebuild_cache,
                                                                             interactive_fallback=interactive_fallback)

    last_close_net_worth_for_changes = None
    sorted_base_historical = None
    if base_historical_nw_df_for_kpi is not None and not base_historical_nw_df_for_kpi.empty:
        sorted_base_historical = base_historical_nw_df_for_kpi.sort_values(by='Date', ascending=False).reset_index(
            drop=True)
        if not sorted_base_historical.empty:
            last_close_net_worth_for_changes = sorted_base_historical['Net Worth'].iloc[0]
            logging.info(
                f"Using last historical close for KPI change calculations: ${last_close_net_worth_for_changes:,.2f}")
    else:
        logging.warning("Cannot get recent historical data for KPI change calculations; metrics will be N/A.")

    for label, (days_offset, _) in change_definitions.items():
        prev_net_worth = get_net_worth_at_previous_day(None, days_offset,
                                                       base_historical_df=sorted_base_historical)  # Pass sorted df
        change_abs_val, change_pct_val, is_positive = None, None, None
        if prev_net_worth is not None and last_close_net_worth_for_changes is not None and prev_net_worth != 0:
            change_abs_val = last_close_net_worth_for_changes - prev_net_worth
            change_pct_val = (change_abs_val / prev_net_worth) * 100
            is_positive = change_abs_val >= 0
        metrics.append({"label": label, "value": None,
                        "sub_value": f"Last Close: ${last_close_net_worth_for_changes:,.2f}<br>vs Prev: ${prev_net_worth:,.2f}" if prev_net_worth is not None and last_close_net_worth_for_changes is not None else "vs N/A",
                        "change_abs": f"{'+' if change_abs_val is not None and change_abs_val >= 0 else ''}${change_abs_val:,.2f}" if change_abs_val is not None else "N/A",
                        "change_pct": f"{'+' if change_pct_val is not None and change_pct_val >= 0 else ''}{change_pct_val:.2f}%" if change_pct_val is not None else "N/A",
                        "positive_change": is_positive, "icon": None})
    return {"type": "key_metrics", "metrics": metrics}


# --- get_asset_value_at_offset (modified for robustness) ---
def get_asset_value_at_offset(
        symbol: str,
        amount: float,
        base_historical_prices_df: pd.DataFrame,
        days_offset: int,  # 0 for last available, 1 for one before last, etc.
        is_cash_equivalent: bool
) -> float | None:
    if is_cash_equivalent: return amount * 1.0
    if base_historical_prices_df is None or base_historical_prices_df.empty or symbol not in base_historical_prices_df.columns:
        return None

    valid_prices = base_historical_prices_df[symbol].dropna()
    if valid_prices.empty: return None

    # valid_prices are typically indexed by date, ascending.
    # We want the price 'days_offset' from the *end* of this series.
    if len(valid_prices) > days_offset:
        try:
            # iloc[-(1+days_offset)] gets the Nth from the end.
            # e.g. offset 0 -> iloc[-1] (last)
            #      offset 1 -> iloc[-2] (second to last)
            price_at_offset = valid_prices.iloc[-(1 + days_offset)]
            return amount * price_at_offset
        except IndexError:
            logging.debug(
                f"Index out of bounds for {symbol} with offset {days_offset} from end. Available: {len(valid_prices)}")
            return None
    return None


def build_summary_by_symbol_section(processed_df: pd.DataFrame) -> dict:
    """Builds the Summary by Symbol table section."""
    logging.info("Building summary by symbol section...")
    if processed_df.empty or 'Market Value' not in processed_df.columns:
        return {"type": "html_content", "html": "<p>No data available for symbol summary.</p>"}

    # Ensure 'price' is numeric before aggregation if it's not already
    # This should be handled by process_portfolio_data, but a check here is safe
    df_for_summary = processed_df.copy()
    if 'price' in df_for_summary.columns:
        df_for_summary['price'] = pd.to_numeric(df_for_summary['price'], errors='coerce')
    else:
        df_for_summary['price'] = 0.0  # Default if somehow missing

    if 'Cost' in df_for_summary.columns:
        df_for_summary['Cost'] = pd.to_numeric(df_for_summary['Cost'], errors='coerce').fillna(0.0)
    else:
        df_for_summary['Cost'] = 0.0

    summary_df = df_for_summary.groupby('Symbol').agg(
        Name=('name', 'first'),
        Total_Amount=('Amount', 'sum'),
        Latest_Price=('price', 'first'),  # Get the latest price (should be unique per symbol)
        Total_Market_Value=('Market Value', 'sum'),
        Asset_Type=('type', 'first'),
        Sector=('sector', 'first'),
        Risk_Level=('Risk Level', 'first'),
        Total_Cost=('Cost', 'sum')
    ).reset_index()

    summary_df.rename(columns={
        'Symbol': 'Symbol',
        'Name': 'Asset Name',
        'Total_Amount': 'Total Amount',
        'Latest_Price': 'Latest Price',  # New column
        'Total_Market_Value': 'Total Market Value',
        'Asset_Type': 'Type',
        'Sector': 'Sector',
        'Risk_Level': 'Risk Level',
        'Total_Cost': 'Total Cost'
    }, inplace=True)

    # Formatting for display
    summary_df['Latest Price'] = summary_df['Latest Price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    summary_df['Total Market Value'] = summary_df['Total Market Value'].apply(
        lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    summary_df['Total Cost'] = summary_df['Total Cost'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    summary_df['Total Amount'] = summary_df['Total Amount'].apply(
        lambda x: f"{x:,.4f}".rstrip('0').rstrip('.') if pd.notnull(x) and isinstance(x, float) and x % 1 != 0 else (
            f"{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isnull(x) else str(x)))
    )

    # Define the order of columns, including the new 'Latest Price'
    cols_ordered = ['Symbol', 'Asset Name', 'Total Amount', 'Latest Price', 'Total Market Value', 'Total Cost', 'Type',
                    'Sector', 'Risk Level']

    # Ensure all columns in cols_ordered exist in summary_df, otherwise, they won't be selected
    final_cols_to_display = [col for col in cols_ordered if col in summary_df.columns]
    summary_df = summary_df[final_cols_to_display]

    html_content = create_html_table(
        df_display=summary_df,
        table_title="Summary by Symbol",
        table_id="summaryBySymbolTable"
    )
    return {"type": "table", "html": html_content}


def build_summary_by_risk_level_section(processed_df: pd.DataFrame) -> dict:
    logging.info("Building summary by asset risk level section...")
    if processed_df.empty or 'Risk Level' not in processed_df.columns or 'Market Value' not in processed_df.columns:
        return {"type": "html_content", "html": "<p>No data available for risk level summary.</p>"}

    summary_df = processed_df.groupby('Risk Level').agg(
        Total_Market_Value=('Market Value', 'sum'),
        Asset_Count=('Symbol', 'nunique')  # Count of unique symbols in that risk category
    ).reset_index()

    total_portfolio_value = processed_df['Market Value'].sum()
    if total_portfolio_value > 0:
        summary_df['Allocation (%)'] = (summary_df['Total_Market_Value'] / total_portfolio_value) * 100
        summary_df['Allocation (%)'] = summary_df['Allocation (%)'].apply(lambda x: f"{x:.2f}%")
    else:
        summary_df['Allocation (%)'] = "0.00%"

    summary_df.rename(columns={
        'Risk Level': 'Risk Level Category',
        'Total_Market_Value': 'Total Market Value',
        'Asset_Count': 'Number of Assets'
    }, inplace=True)

    summary_df['Total Market Value'] = summary_df['Total Market Value'].apply(lambda x: f"${x:,.2f}")
    summary_df = summary_df.sort_values(by='Risk Level Category')
    cols_ordered = ['Risk Level Category', 'Total Market Value', 'Allocation (%)', 'Number of Assets']
    summary_df = summary_df[cols_ordered]

    html_content = create_html_table(df_display=summary_df, table_title="Summary by Asset Risk Level",
                                     table_id="summaryByRiskTable")
    return {"type": "table", "html": html_content}


def build_top_movers_section(
        original_portfolio_input_df: pd.DataFrame,
        rebuild_cache: bool = False,
        interactive_fallback: bool = True
) -> dict:
    logging.info(f"Building top movers section (rebuild_cache={rebuild_cache}, interactive={interactive_fallback})...")
    mover_periods = {"1-Day Movers": (1, "5d"), "5-Day Movers": (5, "10d"), "20-Day Movers": (20, "2mo"),
                     "3-Month Movers": (63, "4mo"), "6-Month Movers": (126, "7mo")}
    all_movers_data = []
    holdings = original_portfolio_input_df.groupby('Symbol')['Amount'].sum().reset_index()
    non_cash_holdings = holdings.copy()
    if non_cash_holdings.empty: return {"type": "html_content", "html": "<p>No assets to analyze for top movers.</p>"}
    symbols_for_history = non_cash_holdings['Symbol'].unique().tolist()
    current_processing_date_for_movers = datetime.now(timezone.utc).replace(hour=0, minute=0,
                                                                            second=0, microsecond=0)

    for period_label, (days_ago_offset, yf_fetch_period) in mover_periods.items():
        logging.debug(
            f"Calculating movers for {period_label} (offset: {days_ago_offset}, fetch period: {yf_fetch_period})")
        period_asset_changes = []
        historical_prices = get_historical_prices(symbols_for_history, period=yf_fetch_period,
                                                  end_date_dt=current_processing_date_for_movers,
                                                  rebuild_cache=rebuild_cache,
                                                  interactive_fallback=interactive_fallback)
        if historical_prices.empty: all_movers_data.append(
            {"period_label": period_label, "error": "Historical data unavailable."}); continue
        for _, holding_row in non_cash_holdings.iterrows():
            symbol_orig, amount = holding_row['Symbol'], holding_row['Amount']
            eff_sym, _, is_cash, was_ignored = get_effective_symbol_info(symbol_orig, f"movers for {period_label}",
                                                                         interactive_fallback)
            if is_cash or was_ignored: continue
            price_col_to_use = eff_sym if eff_sym in historical_prices.columns else symbol_orig
            if price_col_to_use not in historical_prices.columns: continue
            current_value = get_asset_value_at_offset(price_col_to_use, amount, historical_prices, 0, False)
            prev_value = get_asset_value_at_offset(price_col_to_use, amount, historical_prices, days_ago_offset, False)
            if current_value is not None and prev_value is not None:
                dollar_change = current_value - prev_value
                percent_change = (dollar_change / prev_value) * 100 if prev_value != 0 else (
                    0.0 if dollar_change == 0 else np.inf * np.sign(dollar_change))
                period_asset_changes.append(
                    {"symbol": symbol_orig, "dollar_change": dollar_change, "percent_change": percent_change})
        if not period_asset_changes: all_movers_data.append(
            {"period_label": period_label, "error": "No asset changes."}); continue
        df_changes = pd.DataFrame(period_asset_changes).dropna(subset=['dollar_change', 'percent_change'])
        if df_changes.empty: all_movers_data.append(
            {"period_label": period_label, "error": "No valid changes."}); continue
        df_changes.replace([np.inf, -np.inf], np.nan, inplace=True)
        top_dollar_gain = df_changes.nlargest(1, 'dollar_change').iloc[0] if not df_changes[
            df_changes['dollar_change'] > 0].empty else None
        top_dollar_loss = df_changes.nsmallest(1, 'dollar_change').iloc[0] if not df_changes[
            df_changes['dollar_change'] < 0].empty else None
        df_valid_pct = df_changes.dropna(subset=['percent_change'])
        top_pct_gain = df_valid_pct.nlargest(1, 'percent_change').iloc[0] if not df_valid_pct[
            df_valid_pct['percent_change'] > 0].empty else None
        top_pct_loss = df_valid_pct.nsmallest(1, 'percent_change').iloc[0] if not df_valid_pct[
            df_valid_pct['percent_change'] < 0].empty else None
        all_movers_data.append({"period_label": period_label,
                                "top_dollar_gain": top_dollar_gain.to_dict() if top_dollar_gain is not None else None,
                                "top_dollar_loss": top_dollar_loss.to_dict() if top_dollar_loss is not None else None,
                                "top_pct_gain": top_pct_gain.to_dict() if top_pct_gain is not None else None,
                                "top_pct_loss": top_pct_loss.to_dict() if top_pct_loss is not None else None})
    if not all_movers_data: return {"type": "html_content", "html": "<p>Could not determine any top movers.</p>"}
    return {"type": "top_movers", "data": all_movers_data}


# --- Delete the entire build_asset_allocation_section function ---

# --- Add this new function in its place ---
def build_summary_by_asset_class_section(processed_df: pd.DataFrame) -> dict:
    """Builds the Summary by Asset Class table."""
    logging.info("Building summary by asset class section...")
    if processed_df.empty or 'Asset Class' not in processed_df.columns or 'Market Value' not in processed_df.columns:
        return {"type": "html_content", "html": "<p>No data available for asset class summary.</p>"}

    summary_df = processed_df.groupby('Asset Class').agg(
        Total_Market_Value=('Market Value', 'sum'),
        Asset_Count=('Symbol', 'nunique')
    ).reset_index()

    total_portfolio_value = processed_df['Market Value'].sum()
    if total_portfolio_value > 0:
        summary_df['Allocation (%)'] = (summary_df['Total_Market_Value'] / total_portfolio_value) * 100
        summary_df['Allocation (%)'] = summary_df['Allocation (%)'].apply(lambda x: f"{x:.2f}%")
    else:
        summary_df['Allocation (%)'] = "0.00%"

    summary_df.rename(columns={
        'Asset Class': 'Asset Class',
        'Total_Market_Value': 'Total Market Value',
        'Asset_Count': 'Number of Assets'
    }, inplace=True)

    summary_df['Total Market Value'] = summary_df['Total Market Value'].apply(lambda x: f"${x:,.2f}")
    summary_df = summary_df.sort_values(by='Total Market Value', ascending=False)
    cols_ordered = ['Asset Class', 'Total Market Value', 'Allocation (%)', 'Number of Assets']
    summary_df = summary_df[cols_ordered]

    html_content = create_html_table(
        df_display=summary_df,
        table_title="Summary by Asset Class",
        table_id="summaryByAssetClassTable"
    )
    return {"type": "table", "html": html_content}


# --- Replace the existing build_holdings_summary_section function ---
def build_holdings_summary_section(processed_df: pd.DataFrame) -> dict:
    logging.info("Building holdings details section...")
    df_investments = processed_df[processed_df['type'] != 'Cash'].copy()
    df_cash = processed_df[processed_df['type'] == 'Cash'].copy()
    final_html = ""

    def format_gain_loss_dollar(val):
        if pd.isnull(val): return "-"
        klass = "positive" if val > 0 else "negative" if val < 0 else "neutral"
        return f'<span class="{klass}">{"+" if val >= 0 else ""}${val:,.2f}</span>'

    def format_gain_loss_percent(val):
        if pd.isnull(val) or not np.isfinite(val): return "-"
        klass = "positive" if val > 0 else "negative" if val < 0 else "neutral"
        return f'<span class="{klass}">{"+" if val >= 0 else ""}{val:.2f}%</span>'

    if not df_investments.empty:
        df_investments['Market Value'] = pd.to_numeric(df_investments['Market Value'], errors='coerce').fillna(0.0)
        df_investments['Cost'] = pd.to_numeric(df_investments['Cost'], errors='coerce').fillna(0.0)
        df_investments['Gain/Loss $'] = df_investments['Market Value'] - df_investments['Cost']
        df_investments['Gain/Loss %'] = (df_investments['Gain/Loss $'] / df_investments['Cost'].replace(0,
                                                                                                        np.nan)) * 100

        # MODIFIED: Added 'Asset Class' to the display columns
        columns_to_display = [
            "Account", "Symbol", "name", "Amount", "price", "Market Value", "Cost",
            "Gain/Loss $", "Gain/Loss %", "type", "sector", "Risk Level", "Asset Class"
        ]
        df_investments_display = df_investments.reindex(columns=columns_to_display).copy()

        currency_cols = ['price', 'Market Value', 'Cost']
        for col in currency_cols:
            df_investments_display[col] = pd.to_numeric(df_investments_display[col], errors='coerce').apply(
                lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"
            )
        df_investments_display['Amount'] = pd.to_numeric(df_investments_display['Amount'], errors='coerce').apply(
            lambda x: (f"{x:,.4f}".rstrip('0').rstrip('.') if pd.notnull(x) and isinstance(x, (float, int)) and float(
                x) % 1 != 0
                       else f"{x:,.0f}" if pd.notnull(x) else "N/A")
        )
        df_investments_display['Gain/Loss $'] = df_investments['Gain/Loss $'].apply(format_gain_loss_dollar)
        df_investments_display['Gain/Loss %'] = df_investments['Gain/Loss %'].apply(format_gain_loss_percent)

        total_market_value = df_investments["Market Value"].sum()
        total_cost_basis = df_investments["Cost"].sum()
        total_gain_loss_abs = total_market_value - total_cost_basis
        total_gain_loss_pct_str = "-"
        if total_cost_basis != 0:
            total_gain_loss_pct = (total_gain_loss_abs / total_cost_basis) * 100
            total_gain_loss_pct_str = format_gain_loss_percent(total_gain_loss_pct)

        total_row_dict = {col: '' for col in columns_to_display}
        total_row_dict["Account"] = "TOTAL"
        total_row_dict["Market Value"] = f"${total_market_value:,.2f}"
        total_row_dict["Cost"] = f"${total_cost_basis:,.2f}"
        total_row_dict["Gain/Loss $"] = format_gain_loss_dollar(total_gain_loss_abs)
        total_row_dict["Gain/Loss %"] = total_gain_loss_pct_str
        total_row_df = pd.DataFrame([total_row_dict])
        df_investments_display = pd.concat([df_investments_display, total_row_df], ignore_index=True)

        investment_html = create_html_table(
            df_display=df_investments_display,
            table_title="Investment Holdings",
            table_id="investmentHoldingsTable"
        )
        final_html += investment_html

    if not df_cash.empty:
        # MODIFIED: Added 'Asset Class' to cash holdings as well for consistency
        df_cash['Asset Class'] = 'Cash & Equivalents'
        cash_columns_to_display = ["Account", "Symbol", "name", "Market Value", "Asset Class"]
        df_cash_display = df_cash.reindex(columns=cash_columns_to_display).copy()

        df_cash_display['Market Value'] = pd.to_numeric(df_cash_display['Market Value'], errors='coerce').apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A"
        )

        total_cash = df_cash["Market Value"].sum()
        cash_total_row = {col: '' for col in cash_columns_to_display}
        cash_total_row["Account"] = "TOTAL"
        cash_total_row["Market Value"] = f"${total_cash:,.2f}"
        cash_total_df = pd.DataFrame([cash_total_row])
        df_cash_display = pd.concat([df_cash_display, cash_total_df], ignore_index=True)

        cash_html = create_html_table(
            df_display=df_cash_display,
            table_title="Cash & Cash Equivalents",
            table_id="cashHoldingsTable"
        )
        final_html += cash_html

    if not final_html:
        final_html = "<p>No holdings data available to display.</p>"

    return {"type": "html_content", "html": final_html}
