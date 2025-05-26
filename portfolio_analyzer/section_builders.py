# portfolio_analyzer/section_builders.py
import pandas as pd
import numpy as np
import logging
import re  # For parsing yf_period_str in build_key_metrics_section
import plotly.graph_objects as go
from .report_components import create_pie_chart_figure, create_html_table
# create_line_chart_figure is not directly used if fig built within historical section
from .data_processing import get_historical_prices, \
    get_effective_symbol_info  # Added get_effective_symbol_info for movers

# Configure logging for this module if not configured globally
logger = logging.getLogger(__name__)  # Keep this commented unless you need module-specific logger


def build_asset_allocation_section(processed_df: pd.DataFrame) -> dict:
    """Builds the Asset Allocation Overview section dictionary."""
    logging.info("Building asset allocation section...")
    charts_html_list = []

    alloc_type_df = processed_df.groupby("type")["Market Value"].sum().reset_index()
    alloc_type_df = alloc_type_df[alloc_type_df["Market Value"] > 0]
    if not alloc_type_df.empty:
        fig_alloc_type = create_pie_chart_figure(
            df_grouped=alloc_type_df,
            values_col="Market Value",
            names_col="type",
            title="Asset Allocation by Type",
            legend_title="Type"
        )
        charts_html_list.append(fig_alloc_type.to_html(
            full_html=False, include_plotlyjs=False, div_id="chart-alloc-type", config={'responsive': True}
        ))

    sector_allocation_df_source = processed_df[
        ~processed_df["type"].isin(["Cash", "Mutual Fund", "Other", "Unknown"]) &
        (processed_df["sector"].notna()) &
        (processed_df["sector"] != "N/A") &
        (processed_df["sector"] != "N/A ETF Category")
        ]
    if not sector_allocation_df_source.empty:
        alloc_sector_df = sector_allocation_df_source.groupby("sector")["Market Value"].sum().reset_index()
        alloc_sector_df = alloc_sector_df[alloc_sector_df["Market Value"] > 0]
        if not alloc_sector_df.empty:
            fig_alloc_sector = create_pie_chart_figure(
                df_grouped=alloc_sector_df,
                values_col="Market Value",
                names_col="sector",
                title="Stock & ETF Allocation by Sector",
                legend_title="Sector"
            )
            charts_html_list.append(fig_alloc_sector.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-alloc-sector", config={'responsive': True}
            ))

    return {
        "type": "charts_vertical",
        "charts": charts_html_list
    }


def build_holdings_summary_section(processed_df: pd.DataFrame) -> dict:
    """Builds the Holdings Details section dictionary."""
    logging.info("Building holdings details section...")

    columns_to_display = ["Account", "Symbol", "name", "Amount", "price", "Market Value", "type", "sector",
                          "Risk Level", "Cost"]
    df_for_table = processed_df.copy()

    for col in columns_to_display:
        if col not in df_for_table.columns:
            df_for_table[col] = "N/A" if col not in ["Amount", "price", "Market Value", "Cost"] else 0.0

    df_for_table_display = df_for_table[columns_to_display].copy()

    currency_cols = ['price', 'Market Value', 'Cost']
    for col in currency_cols:
        if col in df_for_table_display.columns:
            df_for_table_display[col] = pd.to_numeric(df_for_table_display[col], errors='coerce')
            df_for_table_display[col] = df_for_table_display[col].apply(
                lambda x: f"${x:,.2f}" if pd.notnull(x) and isinstance(x, (int, float)) else (
                    "N/A" if pd.isnull(x) else str(x))
            )

    if 'Amount' in df_for_table_display.columns:
        df_for_table_display['Amount'] = pd.to_numeric(df_for_table_display['Amount'], errors='coerce')
        df_for_table_display['Amount'] = df_for_table_display['Amount'].apply(
            lambda x: f"{x:,.4f}".rstrip('0').rstrip('.') if pd.notnull(x) and isinstance(x,
                                                                                          float) and x % 1 != 0 else (
                f"{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isnull(x) else str(x)))
        )

    if "Market Value" in processed_df.columns and not processed_df.empty:
        numeric_market_value = pd.to_numeric(processed_df["Market Value"], errors='coerce').fillna(0.0)
        numeric_cost = pd.to_numeric(processed_df["Cost"], errors='coerce').fillna(
            0.0) if "Cost" in processed_df.columns else pd.Series([0.0] * len(processed_df))

        total_market_value = numeric_market_value.sum()
        total_cost_basis = numeric_cost.sum()

        total_row_dict = {col: '' for col in columns_to_display}
        total_row_dict[columns_to_display[0]] = "TOTAL"
        total_row_dict["Market Value"] = f"${total_market_value:,.2f}"
        total_row_dict["Cost"] = f"${total_cost_basis:,.2f}"

        total_row_df = pd.DataFrame([total_row_dict])
        df_for_table_display = pd.concat([df_for_table_display, total_row_df], ignore_index=True)

    holdings_summary_html = create_html_table(
        df_display=df_for_table_display,
        table_title="Detailed Holdings",
        table_id="holdingsTable"
    )
    return {"type": "table", "html": holdings_summary_html}


def calculate_net_worth_and_spy_benchmark(
        portfolio_df: pd.DataFrame,
        period: str,
        rebuild_cache: bool = False,
        interactive_fallback: bool = True
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    holdings = portfolio_df.groupby('Symbol')['Amount'].sum().reset_index()
    symbols_to_fetch = holdings[holdings['Symbol'] != 'CASH']['Symbol'].tolist()
    cash_holdings = holdings[holdings['Symbol'] == 'CASH']
    cash_amount = cash_holdings['Amount'].sum() if not cash_holdings.empty else 0.0

    yf_period = "max" if period.lower() == "all" else period

    portfolio_historical_prices_df = pd.DataFrame()
    if symbols_to_fetch:
        portfolio_historical_prices_df = get_historical_prices(
            symbols_to_fetch, period=yf_period, rebuild_cache=rebuild_cache, interactive_fallback=interactive_fallback
        )
        if portfolio_historical_prices_df.empty and cash_amount == 0:
            return None, None

    date_index = None
    if not portfolio_historical_prices_df.empty:
        date_index = portfolio_historical_prices_df.index
    elif cash_amount > 0:
        today = pd.Timestamp('today').normalize()
        num_periods_map = {"max": 252 * 5, "5y": 252 * 5, "1y": 252, "6mo": 126, "3mo": 63, "1mo": 21, "ytd": None}
        effective_period_key = period.lower() if period.lower() in num_periods_map else "1y"
        num_periods = num_periods_map.get(effective_period_key)
        if effective_period_key == "ytd":
            start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
            date_index = pd.bdate_range(start_of_year, today)
        elif num_periods:
            date_index = pd.date_range(end=today, periods=num_periods, freq='B')
        else:
            date_index = pd.date_range(end=today, periods=num_periods_map["1y"], freq='B')
        if date_index.empty: date_index = pd.DatetimeIndex([today])
    else:
        return None, None

    historical_values_dict = {}
    for _, row in holdings.iterrows():
        symbol, amount = row['Symbol'], row['Amount']
        if symbol == 'CASH': continue
        temp_series = pd.Series(0.0, index=date_index, name=symbol)
        if not portfolio_historical_prices_df.empty and symbol in portfolio_historical_prices_df.columns:
            current_symbol_prices = portfolio_historical_prices_df[symbol]
            if isinstance(current_symbol_prices.index, pd.MultiIndex):
                if isinstance(current_symbol_prices.index.get_level_values(0), pd.DatetimeIndex):
                    temp_prices = current_symbol_prices.groupby(level=0).first()
                    asset_prices = temp_prices.reindex(date_index).ffill().bfill()
                else:
                    asset_prices = pd.Series(np.nan, index=date_index)
            else:
                asset_prices = current_symbol_prices.reindex(date_index).ffill().bfill()
            if not asset_prices.isnull().all():
                temp_series = asset_prices * amount
            else:
                logging.warning(f"Historical prices for {symbol} were all NaN after alignment. Using 0 value.")
        else:
            logging.warning(f"No historical price data for {symbol} in fetched df. Its historical value set to 0.")
        historical_values_dict[symbol] = temp_series
    historical_values_dict['CASH_TOTAL'] = pd.Series(cash_amount, index=date_index)
    if not historical_values_dict:
        all_historical_values = pd.DataFrame(index=date_index)
    else:
        all_historical_values = pd.DataFrame(historical_values_dict)
    all_historical_values = all_historical_values.fillna(0.0)

    portfolio_net_worth_series = all_historical_values.sum(axis=1)
    if portfolio_net_worth_series.empty: return None, None
    is_constant = portfolio_net_worth_series.nunique() <= 1
    is_zero = portfolio_net_worth_series.iloc[0] == 0.0 if not portfolio_net_worth_series.empty else True
    if is_constant and is_zero and not (cash_amount > 0 and not symbols_to_fetch): return None, None
    portfolio_net_worth_df = portfolio_net_worth_series.reset_index()
    portfolio_net_worth_df.columns = ['Date', 'Net Worth']
    portfolio_net_worth_df['Net Worth'] = pd.to_numeric(portfolio_net_worth_df['Net Worth'], errors='coerce').fillna(
        0.0)

    spy_prices_df = get_historical_prices(["SPY"], period=yf_period, rebuild_cache=rebuild_cache,
                                          interactive_fallback=interactive_fallback)
    if spy_prices_df.empty or 'SPY' not in spy_prices_df.columns: return portfolio_net_worth_df, None
    spy_prices_aligned = spy_prices_df['SPY'].reindex(date_index).ffill().bfill()
    if spy_prices_aligned.isnull().all(): return portfolio_net_worth_df, None
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
    for display_name in display_periods:
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
        fig.to_html(full_html=False, include_plotlyjs=True, div_id="chart-net-worth", config={'responsive': True})
    ]}


def build_key_metrics_section(
        current_portfolio_processed_df: pd.DataFrame,
        original_portfolio_input_df: pd.DataFrame,
        rebuild_cache: bool = False,
        interactive_fallback: bool = True
) -> dict:
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
    base_historical_nw_df, _ = calculate_net_worth_and_spy_benchmark(original_portfolio_input_df,
                                                                     period=longest_yf_period_for_changes,
                                                                     rebuild_cache=rebuild_cache,
                                                                     interactive_fallback=interactive_fallback)
    last_close_net_worth_for_changes = None
    if base_historical_nw_df is not None and not base_historical_nw_df.empty:
        sorted_historical = base_historical_nw_df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        if not sorted_historical.empty: last_close_net_worth_for_changes = sorted_historical['Net Worth'].iloc[
            0]; logging.info(
            f"Using last historical close for change calculations: ${last_close_net_worth_for_changes:,.2f}")
    else:
        logging.warning("Cannot get recent historical data for change calculations; change metrics will be N/A.")

    for label, (days_offset, _) in change_definitions.items():
        prev_net_worth = None
        if base_historical_nw_df is not None and not base_historical_nw_df.empty and last_close_net_worth_for_changes is not None:
            sorted_historical_for_offset = base_historical_nw_df.sort_values(by='Date', ascending=False).reset_index(
                drop=True)
            if len(sorted_historical_for_offset) > days_offset: prev_net_worth = \
                sorted_historical_for_offset['Net Worth'].iloc[days_offset]
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


def get_asset_value_at_offset(
        symbol: str,
        amount: float,
        base_historical_prices_df: pd.DataFrame,
        days_offset: int,
        is_cash_equivalent: bool  # This param was missing in provided signature
) -> float | None:
    if is_cash_equivalent:
        return amount * 1.0

    if base_historical_prices_df is None or base_historical_prices_df.empty or symbol not in base_historical_prices_df.columns:
        return None

    # Assuming base_historical_prices_df.index is DatetimeIndex and sorted ascending (standard from yf)
    # For "days_ago" logic from a descending sort, we sort here or ensure caller does.
    # Let's assume the DataFrame is sorted ascending by date (standard yf output)
    # To get value 'days_offset' business days ago from the *last available date*:

    valid_prices = base_historical_prices_df[symbol].dropna()
    if valid_prices.empty:
        return None

    # If days_offset is 0, we want the last available price
    # If days_offset is 1, we want the price from one business day before the last available price
    if len(valid_prices) > days_offset:
        # iloc[-1] is last, iloc[-2] is second to last.
        # So, iloc[-(days_offset + 1)]
        try:
            price_at_offset = valid_prices.iloc[-(days_offset + 1)]
            return amount * price_at_offset
        except IndexError:
            logging.debug(
                f"Index out of bounds for {symbol} with offset {days_offset} from end. Available: {len(valid_prices)}")
            return None
    return None


def build_summary_by_symbol_section(processed_df: pd.DataFrame) -> dict:
    logging.info("Building summary by symbol section...")
    if processed_df.empty or 'Market Value' not in processed_df.columns:
        return {"type": "html_content", "html": "<p>No data available for symbol summary.</p>"}

    summary_df = processed_df.groupby('Symbol').agg(
        Name=('name', 'first'),
        Total_Amount=('Amount', 'sum'),
        Avg_Price=('price', 'mean'),  # This is a simple mean, not weighted by amount of each lot
        Total_Market_Value=('Market Value', 'sum'),
        Asset_Type=('type', 'first'),
        Sector=('sector', 'first'),
        Risk_Level=('Risk Level', 'first'),
        Total_Cost=('Cost', 'sum')
    ).reset_index()

    summary_df.rename(columns={
        'Symbol': 'Symbol', 'Name': 'Asset Name', 'Total_Amount': 'Total Amount',
        'Avg_Price': 'Avg. Price', 'Total_Market_Value': 'Total Market Value',
        'Asset_Type': 'Type', 'Sector': 'Sector', 'Risk_Level': 'Risk Level',
        'Total_Cost': 'Total Cost'
    }, inplace=True)

    summary_df['Total Market Value'] = summary_df['Total Market Value'].apply(lambda x: f"${x:,.2f}")
    summary_df['Total Cost'] = summary_df['Total Cost'].apply(lambda x: f"${x:,.2f}")
    summary_df['Avg. Price'] = summary_df['Avg. Price'].apply(lambda x: f"${x:,.2f}")
    summary_df['Total Amount'] = summary_df['Total Amount'].apply(
        lambda x: f"{x:,.4f}".rstrip('0').rstrip('.') if pd.notnull(x) and isinstance(x, float) and x % 1 != 0 else (
            f"{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else ("N/A" if pd.isnull(x) else x))
    )

    cols_ordered = ['Symbol', 'Asset Name', 'Total Amount', 'Avg. Price', 'Total Market Value', 'Total Cost', 'Type',
                    'Sector', 'Risk Level']
    summary_df = summary_df[cols_ordered]

    html_content = create_html_table(df_display=summary_df, table_title="Summary by Symbol",
                                     table_id="summaryBySymbolTable")
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

    mover_periods = {
        "1-Day Movers": (1, "5d"), "5-Day Movers": (5, "10d"),
        "20-Day Movers": (20, "2mo"), "3-Month Movers": (63, "4mo"),
        "6-Month Movers": (126, "7mo")}
    all_movers_data = []
    holdings = original_portfolio_input_df.groupby('Symbol')['Amount'].sum().reset_index()
    non_cash_holdings = holdings.copy()  # Start with all, then filter within loop if identified as cash by mapping

    if non_cash_holdings.empty:
        return {"type": "html_content", "html": "<p>No assets to analyze for top movers.</p>"}

    symbols_for_history = non_cash_holdings['Symbol'].unique().tolist()

    for period_label, (days_ago_offset, yf_fetch_period) in mover_periods.items():
        logging.debug(
            f"Calculating movers for {period_label} (offset: {days_ago_offset}, fetch period: {yf_fetch_period})")
        period_asset_changes = []

        historical_prices = get_historical_prices(symbols_for_history, period=yf_fetch_period,
                                                  rebuild_cache=rebuild_cache,
                                                  interactive_fallback=interactive_fallback)
        if historical_prices.empty:
            all_movers_data.append({"period_label": period_label, "error": "Historical data unavailable."});
            continue

        for _, holding_row in non_cash_holdings.iterrows():
            symbol_orig = holding_row['Symbol']  # This is already uppercase from load_portfolio_data
            amount = holding_row['Amount']

            eff_sym, _, is_cash, was_ignored = get_effective_symbol_info(symbol_orig, f"movers for {period_label}",
                                                                         interactive_fallback)
            if is_cash or was_ignored: continue

            price_col_to_use = eff_sym if eff_sym in historical_prices.columns else symbol_orig
            if price_col_to_use not in historical_prices.columns: continue

            current_value = get_asset_value_at_offset(price_col_to_use, amount, historical_prices, 0, False)
            prev_value = get_asset_value_at_offset(price_col_to_use, amount, historical_prices, days_ago_offset, False)

            if current_value is not None and prev_value is not None:
                dollar_change = current_value - prev_value
                percent_change = (
                                         dollar_change / prev_value) * 100 if prev_value != 0 else 0.0 if dollar_change == 0 else np.inf * np.sign(
                    dollar_change)
                period_asset_changes.append(
                    {"symbol": symbol_orig, "dollar_change": dollar_change, "percent_change": percent_change})

        if not period_asset_changes: all_movers_data.append(
            {"period_label": period_label, "error": "No asset changes."}); continue
        df_changes = pd.DataFrame(period_asset_changes).dropna(subset=['dollar_change', 'percent_change'])
        if df_changes.empty: all_movers_data.append(
            {"period_label": period_label, "error": "No valid changes."}); continue

        # Replace inf with a large number for sorting, or handle display separately
        df_changes.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remove inf before idxmax/min for pct change

        top_dollar_gain = df_changes.nlargest(1, 'dollar_change').iloc[0] if not df_changes[
            df_changes['dollar_change'] > 0].empty else None
        top_dollar_loss = df_changes.nsmallest(1, 'dollar_change').iloc[0] if not df_changes[
            df_changes['dollar_change'] < 0].empty else None

        df_valid_pct = df_changes.dropna(subset=['percent_change'])  # For percent changes, only consider non-NaN
        top_pct_gain = df_valid_pct.nlargest(1, 'percent_change').iloc[0] if not df_valid_pct[
            df_valid_pct['percent_change'] > 0].empty else None
        top_pct_loss = df_valid_pct.nsmallest(1, 'percent_change').iloc[0] if not df_valid_pct[
            df_valid_pct['percent_change'] < 0].empty else None

        all_movers_data.append({
            "period_label": period_label,
            "top_dollar_gain": top_dollar_gain.to_dict() if top_dollar_gain is not None else None,
            "top_dollar_loss": top_dollar_loss.to_dict() if top_dollar_loss is not None else None,
            "top_pct_gain": top_pct_gain.to_dict() if top_pct_gain is not None else None,
            "top_pct_loss": top_pct_loss.to_dict() if top_pct_loss is not None else None,
        })

    if not all_movers_data:
        return {"type": "html_content", "html": "<p>Could not determine any top movers.</p>"}
    return {"type": "top_movers", "data": all_movers_data}
