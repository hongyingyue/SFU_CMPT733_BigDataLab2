"""
Financial Stock Data Processing Script

This script is designed to process and analyze financial stock data using the JQData SDK. It provides functionalities to identify stock industries, retrieve various financial factors, and preprocess these factors for financial analysis. It's structured to work with the Chinese stock market but can be adapted for other markets if the JQData SDK supports them.

Requirements:
- Python 3.x
- JQDataSDK
- pandas
- numpy
- statsmodels
- A valid JQData account for authentication

Features:
- Identification of stock industries based on the SW (Shenwan) classification.
- Retrieval of fundamental financial factors such as PE ratio, PB ratio, and market capitalization.
- Preprocessing of factors including winsorization, standardization, and neutralization against industry and market cap effects.

Usage:
The script accepts command-line arguments to specify the action, stock index, and date range for the analysis.

Actions:
- identify_stock_industry: Identifies the industries of stocks within the specified index and date range.
- get_all_factors_dict: Retrieves and saves fundamental and technical factors for stocks in the specified index and date range.
- preprocess_factors: Preprocesses the retrieved factors by applying winsorization, standardization, and neutralization.

Options:
- --action: Specify the action to perform. Choices are 'identify_stock_industry', 'get_all_factors_dict', 'preprocess_factors'.
- --start_trade_day: The start date for the trade data in 'YYYY-MM-DD' format. Default is '2016-01-01'.
- --end_trade_day: The end date for the trade data in 'YYYY-MM-DD' format. Default is '2023-12-31'.
- --index: The stock index to analyze. Default is '000905.XSHG' (representing the CSI 500).

Example Command:
python ProcessData.py --action identify_stock_industry
For specifying the start_trade_day and end_trade_day or stock_index:
python ProcessData.py --action identify_stock_industry --start_trade_day 2016-01-01 --end_trade_day 2023-12-31 --index 000905.XSHG

Note:
- Ensure you have installed all required libraries before running the script.
- You must replace the 'username' and 'password' variables in the script with your own JQData account credentials for authentication.
- The script can take several hours to complete depending on the specified action and the amount of data being processed.
"""

import os
import datetime
import argparse
import warnings
import jqdatasdk
import pandas as pd
import math
import numpy as np
import statsmodels.api as sm
from time import time
from jqdatasdk import *
from jqdatasdk.technical_analysis import *

# Disable warnings for cleaner output.
warnings.filterwarnings('ignore')

# Authentication with the jqdatasdk service to access financial data.
print('Start the authorization')
username = "18622991028"
password = "SFUsha3152024"
jqdatasdk.auth(username, password)

# Define the query for retrieving fundamental financial data from the jqdatasdk database.
query_data = query(valuation.code,
                   valuation.pe_ratio,
                   valuation.pb_ratio,
                   valuation.ps_ratio,
                   valuation.pcf_ratio,
                   balance.total_assets,
                   balance.total_liability,
                   balance.total_non_current_liability,
                   balance.total_current_liability,
                   cash_flow.cash_and_equivalents_at_end,
                   balance.total_current_assets,
                   indicator.gross_profit_margin,
                   indicator.net_profit_margin,
                   indicator.adjusted_profit_to_profit,
                   indicator.ocf_to_operating_profit,
                   indicator.inc_total_revenue_year_on_year,
                   indicator.inc_operation_profit_year_on_year,
                   indicator.inc_net_profit_year_on_year,
                   valuation.circulating_market_cap,
                   valuation.market_cap
                )

# Function to get the list of trading days between a start and end date, aggregated monthly.
def get_trade_days_monthly(start_trade_day, end_trade_day):
    all_days = get_trade_days(start_date=start_trade_day, end_date=end_trade_day)
    df = pd.DataFrame(all_days, index=all_days)
    df.index = pd.to_datetime(df.index)
    return list(df.resample('m').last().iloc[:,0])

# Function to retrieve the list of stocks within a specified index for the given trade days.
def get_all_stock_list(stock_index, trade_days):
    all_stock_list = {}
    for trade_day in trade_days:
        stocklist = get_index_stocks(stock_index, date=trade_day)
        all_stock_list = set(all_stock_list) | set(stocklist)
    sorted_stock_list = sorted(list(all_stock_list))
    return sorted_stock_list

# Function to get the industry code of a stock on a given date.
def get_industry_code_from_security(stock, date=None):
    industry_index=get_industries(name='sw_l1').index
    for i in range(0,len(industry_index)):
        try:
            index = get_industry_stocks(industry_index[i], date=date).index(stock)
            return industry_index[i]
        except:
            continue
    return u'Not Exist'

# Function to generate a DataFrame mapping stocks to their respective industries.
def get_industry_exposure(stock_list):
    df = pd.DataFrame(index=get_industries(name='sw_l1').index, columns=stock_list)
    for stock in stock_list:
        try:
            df[stock][get_industry_code_from_security(stock)] = 1
        except:
            continue
    return df.fillna(0)

# Function to identify and save the industry information for stocks.
def identify_stock_industry(trade_days, stock_index):
    print('Start identifying the stock industry data, this process probabilty costs 2 hours')
    stocklist = get_all_stock_list(stock_index, trade_days)
    df_industry = get_industry_exposure(stocklist)
    df_industry.to_csv(f'stock_industry.csv', index=True)
    print('Identify the stock industry data finished!')

# Helper functions to filter stocks based on criteria like being newly listed, being ST (Special Treatment), or paused.
def remove_new(stocks, begin_date, num_days_to_new):
    stocklist=[]
    if isinstance(begin_date,str):
        begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    for stock in stocks:
        start_date = get_security_info(stock).start_date
        if start_date < (begin_date - datetime.timedelta(days=num_days_to_new)):
            stocklist.append(stock)
    return stocklist

def remove_st(stocks, begin_date):
    is_st = get_extras('is_st', stocks, end_date=begin_date, count=1)
    return [stock for stock in stocks if not is_st[stock][0]]

def remove_paused(stocks, begin_date):
    is_paused = get_price(stocks, end_date=begin_date, count=1, fields='paused', panel=False)
    return list(is_paused[is_paused['paused'] != 1]['code'])

# Wrapper function to apply all filters and get the final list of stocks for analysis.
def get_stocks_filtered(begin_date, num_days_to_new, stock_index):
    stocklist = get_index_stocks(stock_index, date=begin_date)
    stocklist = remove_new(stocklist, begin_date, num_days_to_new)
    stocklist = remove_st(stocklist, begin_date)
    stocklist = remove_paused(stocklist, begin_date)
    return stocklist

# Function to retrieve and preprocess the financial factors for the stocks.
def get_newfactors_df(stocklist, df, date):
    df['net_assets'] = df['total_assets'] - df['total_liability']
    df_new = pd.DataFrame(index=stocklist)

    df_new['pe_ratio'] = df['pe_ratio']#.apply(lambda x: 1 / x)
    df_new['pb_ratio'] = df['pb_ratio']#.apply(lambda x: 1 / x)
    df_new['ps_ratio'] = df['ps_ratio']#.apply(lambda x: 1 / x)
    df_new['pcf_ratio'] = df['pcf_ratio']#.apply(lambda x: 1 / x)

    df_new['financial_leverage_ratio'] = df['total_assets'] / df['net_assets']
    df_new['debte_to_quity_ratio'] = df['total_non_current_liability'] / df['net_assets']
    df_new['cash_ratio'] = df['total_current_liability'] / df['cash_and_equivalents_at_end']
    df_new['current_ratio'] = df['total_current_liability'] / df['total_current_assets']

    df_new['gross_profit_margin'] = df['gross_profit_margin']
    df_new['net_profit_margin'] = df['net_profit_margin']
    df_new['adjusted_profit_to_profit'] = df['adjusted_profit_to_profit']
    df_new['ocf_to_operating_profit'] = df['ocf_to_operating_profit']

    df_new['inc_total_revenue_year_on_year'] = df['inc_total_revenue_year_on_year']
    df_new['inc_operation_profit_year_on_year'] = df['inc_operation_profit_year_on_year']
    df_new['inc_net_profit_year_on_year'] = df['inc_net_profit_year_on_year']

    df_new['RSI'] = pd.Series(RSI(stocklist, date, N1=20))
    df_new['BIAS'] = pd.Series(BIAS(stocklist, date, N1=20)[0])
    df_new['PSY'] = pd.Series(PSY(stocklist, date, timeperiod=20))
    dif, dea, macd = MACD(stocklist, date, SHORT=10, LONG=30, MID=15)
    df_new['DIF'] = pd.Series(dif)
    df_new['DEA'] = pd.Series(dea)
    df_new['MACD'] = pd.Series(macd)
    df_new['ATR14'] = pd.Series(ATR(stocklist, date, timeperiod=14)[0])
    df_new['HSL'] = pd.Series(HSL(stocklist, date, N=5)[0])

    df_new['circulating_market_cap'] = df['circulating_market_cap']
    df_new['market_cap'] = df['market_cap']
    return df_new

# Main function to loop through trade days, fetch stock factors, and save them.
def get_all_factors_dict(trade_days, query_data, stock_index):
    print('Start getting the stock factors data, this process probabilty costs 40 mins')
    for date in trade_days:
        stocklist = get_stocks_filtered(date, 90, stock_index)
        q_new = query_data.filter(valuation.code.in_(stocklist))
        q_factor = get_fundamentals(q_new, date=date)
        q_factor.set_index('code', inplace=True)
        data_file_name = f'{stock_index}/{date}.csv'
        q_factor_new = get_newfactors_df(stocklist, q_factor, date)
        q_factor_new.to_csv(data_file_name, index=True)
        print(f'Date: {date}, #stocks: {len(q_factor_new.index)} saved in {data_file_name}.')
    print('Get the stock factors data finished!')

# Function to apply winsorization to factor data to reduce the impact of extreme values.
def winsorize(factor, std=3, have_negative = True):
    # print('Winsorizing the factors data...')
    r = factor.copy()
    if have_negative == False:
        r = r[r >= 0]
    else:
        pass
    edge_up = r.mean() + std * r.std()
    edge_low = r.mean() - std * r.std()
    r[r > edge_up] = edge_up
    r[r < edge_low] = edge_low
    return r

# Function to standardize factor data.
def standardize(s, ty=2):
    # print('Standarizing the factors data...')
    data = s.dropna().copy()
    if int(ty) == 1:
        re = (data - data.min()) / (data.max() - data.min())
    elif ty==2:
        re = (data - data.mean()) / data.std()
    elif ty==3:
        re = data / 10 ** np.ceil(np.log10(data.abs().max()))
    return re

# Function to neutralize factor data against industry and market cap effects.
def neutralization(factor, mkt_cap = False, industry = True):
    # print('Neutralizating the factors data...')
    y = factor
    if type(mkt_cap) == pd.Series:
        LnMktCap = mkt_cap.apply(lambda x: math.log(x))
        if industry:
            read_dummy_industry = pd.read_csv(f'stock_industry.csv', index_col=0)
            common_index = factor.index.intersection(read_dummy_industry.T.index)
            x_concat = pd.concat([LnMktCap, read_dummy_industry.T], axis=1)
            x = x_concat.loc[common_index]
        else:
            x = LnMktCap
    elif industry:
        read_dummy_industry = pd.read_csv(f'stock_industry.csv', index_col=0)
        common_index = factor.index.intersection(read_dummy_industry.T.index)
        dummy_industry = read_dummy_industry.T.loc[common_index]
        x = dummy_industry
    result = sm.OLS(y.astype(float), x.astype(float)).fit()
    return result.resid

# Function to fill missing factor data with the industry mean.
def fillwith_industry_mean(df_factors):
    # print('Filling the empty factors data with industry mean...')
    df_industry_matrix = pd.read_csv(f'stock_industry.csv', index_col=0)
    industry_codes = df_industry_matrix.idxmax(axis=0)
    common_index = df_factors.index.intersection(industry_codes.index)
    common_industry_codes = industry_codes[common_index]
    common_industry_codes_reindex = common_industry_codes.reindex(df_factors.index)
    df_factors_new = pd.DataFrame({
        'factor_values': df_factors,
        'industry': common_industry_codes_reindex
    })
    industry_factor_means = df_factors_new.groupby('industry').mean()
    for factor in df_factors_new.columns[:-1]:
        df_factors_new[factor] = df_factors_new.apply(
            lambda row: industry_factor_means.loc[row['industry'], factor] if pd.isnull(row[factor]) else row[factor],
            axis=1
        )
    return pd.Series(list(df_factors_new['factor_values']), index=list(df_factors_new.index))

# Helper function to apply winsorization, standardization, and neutralization to factors.
def compute_win_stand_neutra(factor, trade_day, stock_index):
    data = pd.read_csv(f'{stock_index}/{trade_day}.csv', index_col=0)
    stocks_pb_se = pd.Series(list(data[f'{factor}']), index=list(data.index))
    stocks_mktcap_se = pd.Series(list(data['market_cap']), index=list(data.index))
    df_winsorize = winsorize(stocks_pb_se)
    df_fillwith_industry_mean = fillwith_industry_mean(df_winsorize)
    stocks_neutra_se = neutralization(df_fillwith_industry_mean, stocks_mktcap_se)
    stocks_pb_win_standse = standardize(stocks_neutra_se)
    return stocks_pb_win_standse

# Main function to preprocess factors for all trade days.
def preprocess_factors(trade_days, stock_index):
    print('Start preprocessing the stock factors data, this process probabilty costs 5 mins')
    factors_all = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
                   'financial_leverage_ratio', 'debte_to_quity_ratio', 'cash_ratio', 'current_ratio',
                   'gross_profit_margin', 'net_profit_margin', 'adjusted_profit_to_profit', 'ocf_to_operating_profit',
                   'inc_total_revenue_year_on_year', 'inc_operation_profit_year_on_year', 'inc_net_profit_year_on_year',
                   'RSI', 'BIAS', 'PSY', 'DIF', 'DEA', 'MACD', 'ATR14', 'HSL',
                   'circulating_market_cap', 'market_cap']
    for trade_day in trade_days:
        neutra_factor_datas = pd.DataFrame()
        for factor in factors_all:
            data = compute_win_stand_neutra(f'{factor}', trade_days[0], stock_index)
            neutra_factor_datas[f'{factor}'] = data
        neutra_factor_datas.to_csv(f'preprocess_{stock_index}/{trade_day}.csv', index=True)
        print(f'Preprocess preprocess_{stock_index}/{trade_day}.csv succeed!')
    print('Preprocess the stock factors data finished!')

# Entry point for script execution with argument parsing for flexible operation.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Stock Data')
    parser.add_argument('--action', type=str, help='Action to process')
    parser.add_argument('--start_trade_day', type=str, help='The start trade day YYYY-MM-DD', default='2016-01-01')
    parser.add_argument('--end_trade_day', type=str, help='The end trade day YYYY-MM-DD', default='2023-12-31')
    parser.add_argument('--index', type=str, help='stock index', default='000905.XSHG')
    args = parser.parse_args()

    start_trade_day = args.start_trade_day
    end_trade_day = args.end_trade_day
    trade_days = get_trade_days_monthly(start_trade_day, end_trade_day)

    stock_index = args.index
    if not os.path.exists(f'{stock_index}'):
        os.makedirs(f'{stock_index}')
    if not os.path.exists(f'preprocess_{stock_index}'):
        os.makedirs(f'preprocess_{stock_index}')

    if args.action == 'identify_stock_industry':
        start = time()
        identify_stock_industry(trade_days, stock_index)
        end = time()
        time_cost = datetime.datetime.fromtimestamp(end - start).strftime('%M:%S:%f')
        print(f'Identified the stock\'s industries in {time_cost}!')
    elif args.action == 'get_all_factors_dict':
        start = time()
        get_all_factors_dict(trade_days, query_data, stock_index)
        end = time()
        time_cost = datetime.datetime.fromtimestamp(end - start).strftime('%M:%S:%f')
        print(f'Organized all data files in {time_cost}!')
    elif args.action == 'preprocess_factors':
        start = time()
        preprocess_factors(trade_days, stock_index)
        end = time()
        time_cost = datetime.datetime.fromtimestamp(end - start).strftime('%M:%S:%f')
        print(f'Preprocessed all data files in {time_cost}!')
    else:
        print('The supported actions are "identify_stock_industry", "get_all_factors_dict", and "preprocess_factors"')
