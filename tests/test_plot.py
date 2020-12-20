"""
This file contains all the tests for the dash part in application.py
"""

import numpy as np
import pandas as pd
import src.plot as app
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import datetime


TOTAL_CAPITAL = 10 ** 4
num_format = "{:,}".format

def test_annual_return():
    """
    Test for the annual return calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-03 12:09:07', '2020-10-04 16:10:05', '2020-10-05 12:03:00'],
                         'pnl': [1000, 1050, -787]})

    result = app.pnl_summary(data)
    ann_return = num_format(round ( (1000+1050-787) / TOTAL_CAPITAL / (3 / 365) * 100, 2)) + '%'
    assert ann_return == result['Value'].iloc[0]


def test_cumulative_return():
    """
    Test for the cumulative return calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-04 12:09:07', '2020-10-05 16:10:05',
                                  '2020-10-06 12:01:00', '2020-10-07 17:00:00'],
                         'pnl': [100000, 90000, 75000, -150000]})

    result = app.pnl_summary(data)
    ann_return = num_format(round(((100000+90000+75000-150000) / TOTAL_CAPITAL) * 100, 2)) + '%'
    assert ann_return == result['Value'].iloc[1]


def test_annual_volatility():
    """
    Test for the annual return calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-04 12:09:07', '2020-10-04 12:09:07', '2020-10-05 16:10:05',
                                  '2020-10-06 12:01:00', '2020-10-07 17:00:00'],
                         'pnl': [0, 1000, 900, 750, -1500]})

    std = np.std([1000/TOTAL_CAPITAL, 900/TOTAL_CAPITAL, 750/TOTAL_CAPITAL, -1500/TOTAL_CAPITAL], ddof=1)
    result = app.pnl_summary(data)
    volatility = num_format(round(std*np.sqrt(365), 2))
    assert volatility == result['Value'].iloc[2]


def test_sharpe_ratio():
    """
    Test for the Sharpe Ratio calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-04 12:09:07', '2020-10-05 16:10:05',
                                  '2020-10-06 12:01:00', '2020-10-07 17:00:00'],
                         'pnl': [1000, -800, 950, 600]})

    cum_shift = data['pnl']
    std = np.std(cum_shift, ddof=1)
    mean = cum_shift.mean()
    result = app.pnl_summary(data)
    # round to 2 decimals to be consistent
    sr = num_format(round(mean / std * np.sqrt(365), 2))
    assert sr == result['Value'].iloc[3]


def test_max_dropdown():
    """
    Test for the Max Dropdown calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-04 12:09:07', '2020-10-05 16:10:05',
                                  '2020-10-06 12:01:00', '2020-10-07 17:00:00'],
                         'pnl': [1000, 900, 750, -1500]})

    result = app.pnl_summary(data)
    md = num_format((1000 - (-1500)) / np.max(data['pnl']))
    # 4th value is dropdown
    assert md == result['Value'].iloc[4]


def test_skew():
    """
    Test for the Skew calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-04 12:09:07', '2020-10-05 16:10:05',
                                  '2020-10-06 12:01:00', '2020-10-07 17:00:00'],
                         'pnl': [1000, 900, 750, -1500]})

    result = app.pnl_summary(data)
    sk = num_format(round(data['pnl'].skew(), 2))
    # 5th value is skew
    assert sk == result['Value'].iloc[5]


def test_kurtosis():
    """
    Test for the kurtosis calculation.
    :return:
    """
    data = pd.DataFrame({'date': ['2020-10-04 12:09:07', '2020-10-05 16:10:05',
                                  '2020-10-06 12:01:00', '2020-10-07 17:00:00'],
                         'pnl': [1000, 900, 750, -1500]})

    result = app.pnl_summary(data)
    kui = num_format(round(data['pnl'].kurtosis(), 2))
    # 6th value is kurtosis
    assert kui == result['Value'].iloc[6]


def test_layout():
    """
    Test if new_plot() gives a valid layout for dash code.
    :return:
    """
    layout = app.new_plot()
    assert type(layout) == type(html.Div())

