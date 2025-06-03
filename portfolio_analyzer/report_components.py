# portfolio_analyzer/report_components.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_pie_chart_figure(df_grouped: pd.DataFrame, values_col: str, names_col: str, title: str,
                            legend_title: str) -> go.Figure:
  fig = px.pie(df_grouped, values=values_col, names=names_col, hole=0.3,
               color_discrete_sequence=px.colors.qualitative.Pastel)
  fig.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+percent+value',
                    textfont_size=10, insidetextorientation='radial')
  fig.update_layout(
    title_text=title, title_x=0.5, title_font_size=18,
    margin=dict(t=60, b=40, l=20, r=20, pad=4),
    legend_title_text=legend_title, showlegend=True,
    autosize=True, width=None, height=None,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif")
  )
  return fig


def create_line_chart_figure(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                             y_axis_title: str = None) -> go.Figure:
  df_plot = df.copy()
  df_plot[y_col] = df_plot[y_col].astype(float)
  fig = px.line(df_plot, x=x_col, y=y_col, title=title)
  fig.update_layout(
    title_x=0.5, title_font_size=18,
    xaxis_title=None,
    yaxis_title=y_axis_title if y_axis_title else y_col,
    margin=dict(t=60, b=40, l=40, r=40),  # Adjusted margins
    autosize=True, width=None, height=None,
    yaxis=dict(autorange=True, type='linear'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif")
  )
  fig.update_traces(hovertemplate='%{x|%Y-%m-%d}: $%{y:,.2f}<extra></extra>', line=dict(width=2.5))
  fig.update_xaxes(showgrid=True, gridwidth=1, showline=True, linewidth=1, zeroline=False)
  fig.update_yaxes(showgrid=True, gridwidth=1, showline=True, linewidth=1, zeroline=False)
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


def create_bar_chart_figure(
    df_grouped: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_axis_title: str | None = None,
    y_axis_title: str | None = None,
    orientation: str = 'v',
    color_discrete_sequence=px.colors.qualitative.Alphabet  # A nice qualitative color sequence
) -> go.Figure:
  # For formatting text on bars (e.g., 1.11M, 967.6k)
  def format_bar_text(val):
    if val >= 1e6:
      return f"{val / 1e6:.3f}M"
    elif val >= 1e3:
      return f"{val / 1e3:.3f}k"
    return f"{val:,.2f}"

  # Apply formatting to the value column if it's numeric for text display
  df_display = df_grouped.copy()
  if pd.api.types.is_numeric_dtype(df_display[y_col]):
    # Create a text column for formatted values
    df_display['text_on_bar'] = df_display[y_col].apply(format_bar_text)
    text_col_name = 'text_on_bar'
  else:
    text_col_name = y_col  # Use original y_col if not numeric (less likely for values)

  if orientation == 'h':
    fig = px.bar(df_display,
                 y=x_col,
                 x=y_col,
                 title=title,
                 orientation='h',
                 text=text_col_name,  # Use the formatted text column
                 color=x_col,  # Color bars by category
                 color_discrete_sequence=color_discrete_sequence)
    fig.update_layout(yaxis_title=x_axis_title if x_axis_title else x_col,
                      xaxis_title=y_axis_title if y_axis_title else y_col)
    fig.update_traces(textposition='outside')  # Place text outside horizontal bars
  else:  # Vertical
    fig = px.bar(df_display,
                 x=x_col,
                 y=y_col,
                 title=title,
                 orientation='v',
                 text=text_col_name,
                 color=x_col,
                 color_discrete_sequence=color_discrete_sequence)
    fig.update_layout(xaxis_title=x_axis_title if x_axis_title else x_col,
                      yaxis_title=y_axis_title if y_axis_title else y_col)
    fig.update_traces(textposition='outside')  # Place text above vertical bars

  fig.update_layout(
    title_font_size=18,
    title_x=0.5,
    margin=dict(t=60, b=40, l=40, r=40),  # Adjusted default margins
    autosize=True,
    width=None,
    height=None,
    showlegend=False,  # Usually not needed if bars are colored by category and labeled
    # Initial theme-agnostic background for better JS theme switching
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(
      family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
      # color will be set by JS theme toggle
    ),
    bargap=0.2,  # Gap between bars of different categories
    # bargroupgap=0.1 # Gap between bars within the same category (if grouped bar chart)
  )

  # Customize hovertemplate
  if orientation == 'h':
    fig.update_traces(hovertemplate=f"<b>{x_col}</b>: %{{y}}<br><b>{y_col}</b>: %{{x:,.2f}}<extra></extra>")
  else:
    fig.update_traces(hovertemplate=f"<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y:,.2f}}<extra></extra>")

  # Bar appearance
  fig.update_traces(
    marker_line_width=1,
    # marker_line_color='rgba(0,0,0,0.7)', # Softer outline
    # width=0.7 # Use bargap for spacing control instead of fixed width
  )

  # Axis styling (will be further themed by JS)
  fig.update_xaxes(
    showgrid=True, gridwidth=1,  # gridcolor will be set by JS
    showline=True, linewidth=1,  # linecolor will be set by JS
    zeroline=False
  )
  fig.update_yaxes(
    showgrid=True, gridwidth=1,  # gridcolor will be set by JS
    showline=True, linewidth=1,  # linecolor will be set by JS
    zeroline=False
  )

  return fig
