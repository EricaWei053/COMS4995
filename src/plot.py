"""
plot.py to view plots.
"""
import sys
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import flask
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
application = flask.Flask(__name__)

# dash object
dash_app = dash.Dash(__name__, server=application, external_stylesheets=[dbc.themes.BOOTSTRAP],
                     url_base_pathname='/dash_plots/')
dash_app.validation_layout = True
dash_app.layout = html.Div()
# global variables for update dash dynamically depending on different user
OPTION_LIST = []
PNL_PATHS = []
TOTAL_CAPITAL = 10 ** 4


@application.route('/', methods=['GET'])
def user_results():
    """
    Redirect to dosh route for visualization.
    :return:
    """
    return flask.redirect('/dash_plots')


@application.route('/dash_plots/', methods=['GET'])
def render_reports():
    """
    Redirect flask endpoint to dash server endpoint.
    :return:
    """
    return flask.redirect('/dash_plots')

# dash part
def fig_update(file_path):
    """
    Given the file path, return an updated fig graph to display.
    :param file_path: string, to get csv file.
    :return: figs, the styled graph.
    """

    cr_fig, sr_rolling, pnl_hist, pnl_df, fig_3d = None, None, None, None, None
    if file_path is not None:

        pnl_df = pd.read_csv(file_path)

        pnl_df['cusum'] = pnl_df['pnl'].cumsum()
        cr_fig = px.line(pnl_df, x='date', y='cusum')

        cr_fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        # 3d plot
        another_df = pd.DataFrame()
        another_df['normalized_pnl'] = pnl_df['pnl'].div(TOTAL_CAPITAL)
        another_df['date'] = pnl_df['date']
        another_df['rolling_risk'] = pnl_df['pnl'].iloc[1:].div(TOTAL_CAPITAL).rolling(3).std()
        another_df = another_df[another_df['rolling_risk'] > 0]
        risk_list = another_df['rolling_risk'].tolist()

        another_df['color'] = np.asarray([int(num) for num in risk_list])
        fig_3d = px.scatter_3d(another_df, x='date', y='rolling_risk', z='normalized_pnl',
                               color = 'color', width=800, height=800, opacity=0.7)

        # Rolling sharpe ratio plot
        pnl_df['rolling_SR'] = pnl_df.pnl.rolling(60).apply(
            lambda x: (x.mean() - 0.02) / x.std(), raw=True)

        pnl_df.fillna(0, inplace=True)
        sr_df = pnl_df[pnl_df['rolling_SR'] > 0]
        sr_rolling = go.Figure([go.Scatter(x=sr_df['date'], y=sr_df['rolling_SR'],
                                           line=dict(color="DarkOrange"), mode='lines+markers')])

        sr_rolling.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        mean = np.mean(sr_df['rolling_SR'])
        avg_title = "Average value={:.3f}".format(mean)
        sr_rolling.add_hline(y=mean, line_width=3, line_dash="dash", line_color="green",
                             annotation_text=avg_title,
                             annotation_position="bottom right")

        # pnl histogram plot
        pnl_hist = go.Figure()
        profit = pnl_df[pnl_df['pnl'] > 0]
        loss = pnl_df[pnl_df['pnl'] < 0]

        pnl_hist.add_trace(go.Bar(x=profit['date'], y=loss['pnl'],
                                  marker_color='crimson',
                                  name='loss'))
        pnl_hist.add_trace(go.Bar(x=loss['date'], y=profit['pnl'],
                                  marker_color='lightslategrey',
                                  name='profit'))

    return cr_fig, sr_rolling, pnl_hist, pnl_df, fig_3d


def new_plot():
    """
    Gives structure of plots.
    :return: html div
    """

    content_style = {
        "margin-left": "32rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    # global option_list

    contents = html.Div([
        html.Div(dcc.Dropdown(
            id='backtest_result',
            options=OPTION_LIST,
            placeholder="Select Backtest Result        ",
            style=dict(
                width='200%',
                verticalAlign="left"),
            className="dash-bootstrap"
        ), style={"width": "200%"}, ),

        html.Div(
            [
                html.H2('Cumulative Return',
                        style={'textAlign': 'center', 'font-family': 'Georgia'}),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='pnl_fig'),
                                width={"size": 10, "offset": 2}),
                    ]
                )

            ],
            style=content_style
        ),

        html.Div(
            [
                html.H2('3D View of Daily Change',
                        style={'textAlign': 'center', 'font-family': 'Georgia'}),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='fig_3d'),
                                width={"size": 10, "offset": 2}),
                    ] )

            ],
            style=content_style
        ),

        html.Div(
            [
                html.H2('Rolling Sharpe Ratio (1-month)',
                        style={'textAlign': 'center', 'font-family': 'Georgia'}),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='sr_rolling'),
                                width={"size": 10, "offset": 2}),
                    ]
                )
            ],
            style=content_style
        ),

        html.Div(
            [
                html.H2('Profit and Loss histogram',
                        style={'textAlign': 'center', 'font-family': 'Georgia'}),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id='pnl_hist'),
                                width={"size": 10, "offset": 2}),
                    ]
                )

            ],
            style=content_style
        ),
        html.Div(id='table')
    ])
    return contents


@dash_app.callback(
    Output('pnl_fig', 'figure'),
    Output('fig_3d', 'figure'),
    Output('sr_rolling', 'figure'),
    Output('pnl_hist', 'figure'),
    Output('table', 'children'),
    Input('backtest_result', 'value'))
def update_graph(backtest_fp):
    """
    Dash callback function for update graphs depending
    on the chosen backtest file from dropdown bar.
    :param backtest_fp:
    :return:
    """
    table_style = {
        "position": "fixed",
        "top": 60,
        "left": 5,
        "bottom": 5,
        "width": "30rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }
    if backtest_fp is not None:

        pnl_fig, sr_rolling, pnl_hist, pnl_df, fig_3d = fig_update(backtest_fp)
        table_df = pnl_summary(pnl_df)
        table_comp = html.Div(
            [
                html.H2('Statistic Table',
                        style={'textAlign': 'center', 'font-family': 'Georgia',
                               'front_size': '30'}),
                html.Hr(),
                dt.DataTable(
                    data=table_df.to_dict('records'),
                    columns=[{'id': c, 'name': c} for c in table_df.columns],

                    style_cell={'front_size': '30px'},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Backtest'},
                            'textAlign': 'left'
                        },

                        {
                            'if': {'column_id': 'Category'},
                            'textAlign': 'left'
                        },

                    ],
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                            'if': {'column_id': 'Category'},
                            'fontWeight': 'bold'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                ),
            ], style=table_style
        )
        return pnl_fig, fig_3d, sr_rolling, pnl_hist, table_comp

    raise PreventUpdate


def get_plot(file_paths):
    """
    Get two dictionaries, one mapping from strategy id to strategy name,
    another is mapping from strategy id to strategy location.
    Update the global variables for OptionList and pnl_paths.
    :param file_paths: a list of file paths.
    :return: None
    """

    file_names = {}
    id_paths = {}

    for idx, file_path in enumerate(file_paths):
        file_name = file_path.split('/')[-1]
        file_name = file_name.split('.')[0]

        file_names[idx] = file_name
        id_paths[idx] = file_path
        global OPTION_LIST
        global PNL_PATHS
        OPTION_LIST = [{'label': v, 'value': id_paths[k]} for k, v in file_names.items()]
        PNL_PATHS = [id_paths[k] for k in file_names.keys()]


def pnl_summary(data):
    """
    Statistic analysis of backtest result.
    :param data: A dataframe including the backtest results, contains date and pnl two columns.
    :return: A dataframe for making the table, it contains two columns,
    category name and corresponding values.
    """
    data['cumulative'] = data['pnl'].cumsum()
    result = {'Category': [], 'Value': []}
    total_date = data.shape[0]
    return_value = (data['cumulative'].iloc[-1]) / TOTAL_CAPITAL

    num_format = "{:,}".format

    # Annual return
    annual_return = round(return_value / (total_date / 365) * 100, 2)
    result['Category'].append('Annual Return')
    result['Value'].append(num_format(annual_return) + '%')

    # Cumulative return
    cumulative_return = round(return_value * 100, 2)
    result['Category'].append('Cumulative Return')
    result['Value'].append(num_format(cumulative_return) + '%')

    # Annual volatility
    daily_change = data['pnl'].iloc[1:].div(TOTAL_CAPITAL)
    annual_volatility = round(daily_change.std() * np.sqrt(365), 2)
    result['Category'].append('Annual Volatility')
    result['Value'].append(num_format(annual_volatility))

    # Sharpe ratio
    ratio_value = data['pnl'].div(TOTAL_CAPITAL)
    sharpe_ratio = round(ratio_value.mean() / ratio_value.std() * np.sqrt(365), 2)
    result['Category'].append('Sharpe Ratio')
    result['Value'].append(num_format(sharpe_ratio))

    # Max Dropdown
    max_drop = round((np.max(data['pnl']) - np.min(data['pnl'])) / np.max(data['pnl']), 2)
    result['Category'].append('Max Dropdown')
    result['Value'].append(num_format(max_drop))

    # Skew
    skew = round(data['pnl'].skew(), 2)
    result['Category'].append('Skew')
    result['Value'].append(num_format(skew))

    # Kurtosis
    kurtosis = round(data['pnl'].kurtosis(), 2)
    result['Category'].append('Kurtosis')
    result['Value'].append(num_format(kurtosis))

    return pd.DataFrame(result)


if __name__ == "__main__":

    dash_app.layout = new_plot
    dash_app.title = 'View'
    app_embeds = DispatcherMiddleware(application, {
        '/dash_plot': dash_app.server
    })
    files = sys.argv[1:]

    get_plot(files)
    run_simple('localhost', 5000, app_embeds, use_reloader=True, use_debugger=True)
