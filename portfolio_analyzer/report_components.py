# portfolio_analyzer/report_components.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_pie_chart_figure(df_grouped: pd.DataFrame, values_col: str, names_col: str, title: str,
                            legend_title: str) -> go.Figure:
    """
    Creates a generic interactive pie chart Plotly figure.
    Assumes df_grouped is already grouped and has the necessary columns.
    """
    fig = px.pie(
        df_grouped,
        values=values_col,
        names=names_col,
        hole=0.3,
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        textfont_size=10,
        insidetextorientation='radial'
    )
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(t=50, b=40, l=40, r=40, pad=4),
        legend_title_text=legend_title,
        showlegend=True,
        autosize=True,
        width=None,
        height=None,
    )
    return fig


def create_html_table(df_display: pd.DataFrame, table_title: str, table_id: str = "genericTable") -> str:
    table_html_content = df_display.to_html(
        index=False,
        escape=False,
        classes='styled-table display compact datatable-enhance',  # ADDED 'datatable-enhance'
    )

    # Inject the table ID
    # Ensure this replacement is robust enough for your pandas version's output
    table_html_content = table_html_content.replace(
        'class="dataframe styled-table display compact datatable-enhance"',
        f'id="{table_id}" class="dataframe styled-table display compact datatable-enhance"', 1)
    # Fallback if the above was too specific
    if f'id="{table_id}"' not in table_html_content:
        table_html_content = table_html_content.replace(
            '<table border="1"',
            f'<table border="1" id="{table_id}"', 1)

    # ... (rest of header and total row styling) ...
    table_html_content = table_html_content.replace('<thead>\n  <tr style="text-align: right;">',
                                                    '<thead>\n  <tr class="header-row" style="text-align: left;">', 1)
    table_html_content = table_html_content.replace('<thead>\n  <tr>',
                                                    '<thead>\n  <tr class="header-row">', 1)
    if "TOTAL" in df_display.iloc[-1, 0] if not df_display.empty and df_display.shape[0] > 0 and isinstance(
            df_display.iloc[-1, 0], str) else False:
        parts = table_html_content.rsplit('<tr>', 1)
        if len(parts) > 1:
            table_html_content = parts[0] + '<tr class="total-row">' + parts[1]

    return f"""
    <h3>{table_title}</h3>
    <div class="table-responsive-wrapper">
        {table_html_content}
    </div>
    """


def create_line_chart_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                             y_axis_title: str = None) -> go.Figure:
    df_plot = df.copy()
    df_plot[y_col] = df_plot[y_col].astype(float)
    fig = px.line(df_plot, x=x_col, y=y_col, title=title)
    fig.update_layout(
        title_x=0.5,
        xaxis_title=None,
        yaxis_title=y_axis_title if y_axis_title else y_col,
        margin=dict(t=50, b=20, l=20, r=20),
        autosize=True,
        width=None,
        height=None,
        yaxis=dict(
            autorange=True,
            type='linear'
        )
    )
    fig.update_traces(hovertemplate='%{x|%Y-%m-%d}: $%{y:,.2f}')
    return fig
