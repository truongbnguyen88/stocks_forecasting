from prophet import Prophet
import xgboost as xgb

import sys, os, yaml, ta
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

import yfinance as yf
from dateutil.relativedelta import relativedelta


def import_stock_data(stock_sticker, start_dt):
    # Download stock price data
    two_month_before_start_dt = pd.to_datetime(start_dt) - relativedelta(months=2)
    today_date = datetime.today().date()

    stock_data = yf.download(stock_sticker, start=two_month_before_start_dt, end=today_date.strftime('%Y-%m-%d'))
    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    stock_data = stock_data.reset_index()

    # calculate RSI
    stock_data['RSI'] = ta.momentum.RSIIndicator(close=stock_data[f'Close_{stock_sticker}'], window=14).rsi()
    # calculate MACD
    stock_data['MACD'] = ta.trend.MACD(close=stock_data[f'Close_{stock_sticker}'], window_slow=26, window_fast=12, window_sign=9).macd()
    # calculate EMA
    stock_data['EMA_9_of_MACD'] = ta.trend.EMAIndicator(close=stock_data['MACD'], window=9).ema_indicator()
    # calculate Bollinger Bands
    stock_data['BB_High'] = ta.volatility.BollingerBands(close=stock_data[f'Close_{stock_sticker}'], window=20, window_dev=2).bollinger_hband()
    stock_data['BB_Low'] = ta.volatility.BollingerBands(close=stock_data[f'Close_{stock_sticker}'], window=20, window_dev=2).bollinger_lband()
    # calculate SMA
    stock_data['SMA_20'] = ta.trend.SMAIndicator(close=stock_data[f'Close_{stock_sticker}'], window=20).sma_indicator()
    # calculate 30 days moving average
    stock_data['MA_30'] = stock_data[f'Close_{stock_sticker}'].rolling(window=30).mean()

    stocks_indicator_features = ['RSI', 'MACD', 'BB_High', 'BB_Low', 'SMA_20', 'EMA_9_of_MACD', 'MA_30']

    # just keep data after start_dt
    stock_data = stock_data[stock_data['Date'] >= start_dt]
    stock_data = stock_data.reset_index(drop=True)

    return stock_data, stocks_indicator_features


def preprocess_data(df_in, stocks_additional_features=None, sticker=None):
    df = df_in.copy(deep=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # column rename into Prophet format
    df.rename(columns={'Date': 'ds', 
                     f'Close_{sticker}': 'y',
                     f'High_{sticker}': 'high_value',
                     f'Low_{sticker}': 'low_value',
                     f'Open_{sticker}': 'open_value',
                     f'Volume_{sticker}': 'volume_value'}, inplace=True)
    
    # normalize volume values
    df['volume_value'] = df['volume_value'] / 1e6 # use millions for better readability
    exo_vars = ['high_value', 'low_value', 'open_value', 'volume_value'] + stocks_additional_features
    for col in exo_vars:
        df[col] = df[col].astype(float)
    
    return df, exo_vars


def clean_market_data(df_sp500_in, df_nasdaq_in, df_vix_in):
    df_sp500 = df_sp500_in.copy(deep=True)
    df_nasdaq = df_nasdaq_in.copy(deep=True)
    df_vix = df_vix_in.copy(deep=True)

    df_sp500.columns = ['_'.join(col).strip() for col in df_sp500.columns.values]
    df_nasdaq.columns = ['_'.join(col).strip() for col in df_nasdaq.columns.values]
    df_vix.columns = ['_'.join(col).strip() for col in df_vix.columns.values]

    # Convert index to datetime
    df_sp500.index = pd.to_datetime(df_sp500.index)
    df_nasdaq.index = pd.to_datetime(df_nasdaq.index)
    df_vix.index = pd.to_datetime(df_vix.index)

    # Resample to daily frequency and forward fill missing values
    df_sp500 = df_sp500.resample('D').ffill()
    df_nasdaq = df_nasdaq.resample('D').ffill()
    df_vix = df_vix.resample('D').ffill()

    # Reset index to have 'ds' column
    df_sp500.reset_index(inplace=True)
    df_nasdaq.reset_index(inplace=True)
    df_vix.reset_index(inplace=True)

    # Rename columns for consistency
    df_sp500.rename(columns={'Date': 'ds', 'Close_^GSPC': 'sp500_close', 'Volume_^GSPC': 'sp500_volume'}, inplace=True)
    df_nasdaq.rename(columns={'Date': 'ds', 'Close_^IXIC': 'nasdaq_close', 'Volume_^IXIC': 'nasdaq_volume'}, inplace=True)
    df_vix.rename(columns={'Date': 'ds', 'Close_^VIX': 'vix_close'}, inplace=True)

    df_market = df_sp500[['ds','sp500_close','sp500_volume']].merge(df_nasdaq[['ds','nasdaq_close','nasdaq_volume']], on='ds', how='left')
    df_market = df_market.merge(df_vix[['ds','vix_close']], on='ds', how='left')
    df_market['ds'] = pd.to_datetime(df_market['ds'])
    df_market = df_market.sort_values(by='ds').reset_index(drop=True)

    df_market['sp500_volume']  = df_market['sp500_volume'] / 1e6 # use millions for better readability
    df_market['nasdaq_volume'] = df_market['nasdaq_volume'] / 1e6 # use millions for better readability

    # perform log-transf for volume features
    for c in ['sp500_volume', 'nasdaq_volume']:
        df_market[c] = np.log(df_market[c])

    return df_market


def clean_tech_stocks_data(df_aapl_in, df_nvda_in, df_msft_in):
    df_aapl = df_aapl_in.copy(deep=True)
    df_nvda = df_nvda_in.copy(deep=True)
    df_msft = df_msft_in.copy(deep=True)
    df_aapl.columns = ['_'.join(col).strip() for col in df_aapl.columns.values]
    df_nvda.columns = ['_'.join(col).strip() for col in df_nvda.columns.values]
    df_msft.columns = ['_'.join(col).strip() for col in df_msft.columns.values]
    # Convert index to datetime
    df_aapl.index = pd.to_datetime(df_aapl.index)
    df_nvda.index = pd.to_datetime(df_nvda.index)
    df_msft.index = pd.to_datetime(df_msft.index)
    # Resample to daily frequency and forward fill missing values
    df_aapl = df_aapl.resample('D').ffill()
    df_nvda = df_nvda.resample('D').ffill()
    df_msft = df_msft.resample('D').ffill()
    # Reset index to have 'ds' column
    df_aapl.reset_index(inplace=True)
    df_nvda.reset_index(inplace=True)
    df_msft.reset_index(inplace=True)

    # Rename columns for consistency
    df_aapl.rename(columns={'Date': 'ds', 'Close_AAPL': 'aapl_close', 'Volume_AAPL': 'aapl_volume'}, inplace=True)
    df_nvda.rename(columns={'Date': 'ds', 'Close_NVDA': 'nvda_close', 'Volume_NVDA': 'nvda_volume'}, inplace=True)
    df_msft.rename(columns={'Date': 'ds', 'Close_MSFT': 'msft_close', 'Volume_MSFT': 'msft_volume'}, inplace=True)

    df_tech_stocks = df_aapl[['ds','aapl_close','aapl_volume']].merge(df_nvda[['ds','nvda_close','nvda_volume']], on='ds', how='left')
    df_tech_stocks = df_tech_stocks.merge(df_msft[['ds','msft_close','msft_volume']], on='ds', how='left')
    df_tech_stocks['ds'] = pd.to_datetime(df_tech_stocks['ds'])
    df_tech_stocks = df_tech_stocks.sort_values(by='ds').reset_index(drop=True)
    df_tech_stocks['aapl_volume']  = df_tech_stocks['aapl_volume'] / 1e6 # use millions for better readability
    df_tech_stocks['nvda_volume'] = df_tech_stocks['nvda_volume'] / 1e6 # use millions for better readability
    df_tech_stocks['msft_volume'] = df_tech_stocks['msft_volume'] / 1e6 # use millions for better readability

    # perform log-transf for volume features
    for c in ['aapl_volume', 'nvda_volume', 'msft_volume']:
        df_tech_stocks[c] = np.log(df_tech_stocks[c])
    return df_tech_stocks


def get_earnings_dates(sticker=None):
    # Select the stock
    ticker = yf.Ticker(sticker)  # Example: Apple
    # Get earnings dates
    earnings = ticker.get_earnings_dates(limit=40).reset_index()
    earnings = earnings.rename(columns={'Earnings Date': 'ds', 
                                        'EPS Estimate': 'eps_estimate', 
                                        'Reported EPS': 'reported_eps',
                                        'Surprise(%)': 'surprise_pct'})
    earnings['ds'] = earnings['ds'].dt.strftime('%Y-%m-%d')
    earnings['ds'] = pd.to_datetime(earnings['ds'])
    eps_vars = ['eps_estimate', 'reported_eps', 'surprise_pct']

    return earnings, eps_vars


def get_fred_inflation_data(exo_vars, start_date=None):
    from fredapi import Fred
    # Your FRED API key here
    fred = Fred(api_key='e7bc3e975c7ef55f5e3bd20671a88973')
    # Pull U.S. CPI (Consumer Price Index for All Urban Consumers)
    cpi = fred.get_series('CPIAUCSL')
    df_cpi = pd.DataFrame(cpi).reset_index()
    df_cpi.columns = ['ds', 'cpi']
    df_cpi['ds'] = pd.to_datetime(df_cpi['ds'])
    df_cpi = df_cpi[df_cpi['ds'] >= start_date]  # Restrict to the same period as stock data
    exo_vars.append('cpi')

    # Ensure the data is sorted by date
    df_cpi = df_cpi.sort_values(by='ds')
    # Set the date column as the index
    df_cpi.set_index('ds', inplace=True)
    # Create a new date range with daily frequency
    daily_index = pd.date_range(start=df_cpi.index.min(), end=df_cpi.index.max(), freq='D')
    # Reindex the DataFrame to include the daily dates
    df_cpi_daily = df_cpi.reindex(daily_index)
    # Interpolate the missing values
    df_cpi_daily = df_cpi_daily.interpolate(method='linear')
    # Reset the index and rename the columns
    df_cpi_daily.reset_index(inplace=True)
    df_cpi_daily.rename(columns={'index': 'ds'}, inplace=True)
    return df_cpi_daily, exo_vars


def get_unemployment_rate_data(exo_vars, start_date=None):
    from fredapi import Fred
    fred = Fred(api_key='e7bc3e975c7ef55f5e3bd20671a88973')
    # Pull U.S. Unemployment Rate data
    unemployment_rate = fred.get_series('UNRATE')
    df_unemployment = pd.DataFrame(unemployment_rate).reset_index()
    df_unemployment.columns = ['ds', 'unemployment_rate']
    if 'unemployment_rate' not in exo_vars:
        exo_vars.append('unemployment_rate')
    df_unemployment['ds'] = pd.to_datetime(df_unemployment['ds'])
    df_unemployment = df_unemployment[df_unemployment['ds'] >= start_date]  # Restrict to the same period as stock data

    # Ensure the data is sorted by date
    df_unemployment = df_unemployment.sort_values(by='ds')
    # Set the date column as the index
    df_unemployment.set_index('ds', inplace=True)
    # Create a new date range with daily frequency
    daily_index = pd.date_range(start=df_unemployment.index.min(), end=df_unemployment.index.max(), freq='D')
    # Reindex the DataFrame to include the daily dates
    df_unemployment_daily = df_unemployment.reindex(daily_index)
    # Interpolate the missing values
    df_unemployment_daily = df_unemployment_daily.interpolate(method='linear')
    # Reset the index and rename the columns
    df_unemployment_daily.reset_index(inplace=True)
    df_unemployment_daily.rename(columns={'index': 'ds'}, inplace=True)
    return df_unemployment_daily, exo_vars


def get_interest_rate_data(exo_vars, start_date=None):
    from fredapi import Fred
    fred = Fred(api_key='e7bc3e975c7ef55f5e3bd20671a88973')  # Replace with your FRED API key
    # Pull U.S. Federal Funds Rate data
    interest_rate = fred.get_series('FEDFUNDS')
    df_interest_rate = pd.DataFrame(interest_rate).reset_index()
    df_interest_rate.columns = ['ds', 'interest_rate']
    if 'interest_rate' not in exo_vars:
        exo_vars.append('interest_rate')
    df_interest_rate['ds'] = pd.to_datetime(df_interest_rate['ds'])
    df_interest_rate = df_interest_rate[df_interest_rate['ds'] >= start_date]  # Restrict to the same period as stock data

    # Ensure the data is sorted by date
    df_interest_rate = df_interest_rate.sort_values(by='ds')
    # Set the date column as the index
    df_interest_rate.set_index('ds', inplace=True)
    # Create a new date range with daily frequency
    daily_index = pd.date_range(start=df_interest_rate.index.min(), end=df_interest_rate.index.max(), freq='D')
    # Reindex the DataFrame to include the daily dates
    df_interest_rate_daily = df_interest_rate.reindex(daily_index)
    # Interpolate the missing values
    df_interest_rate_daily = df_interest_rate_daily.interpolate(method='linear')
    # Reset the index and rename the columns
    df_interest_rate_daily.reset_index(inplace=True)
    df_interest_rate_daily.rename(columns={'index': 'ds'}, inplace=True)
    return df_interest_rate_daily, exo_vars


def get_consumer_behavior_data(exo_vars, start_date=None):
    from fredapi import Fred
    fred = Fred(api_key='e7bc3e975c7ef55f5e3bd20671a88973')  # Replace with your FRED API key
    # Pull UMCSENT data
    umcsent = fred.get_series('UMCSENT')
    df_umcsent = pd.DataFrame(umcsent).reset_index()
    df_umcsent.columns = ['ds', 'umcsent']
    if 'umcsent' not in exo_vars:
        exo_vars.append('umcsent')
    df_umcsent['ds'] = pd.to_datetime(df_umcsent['ds'])
    df_umcsent = df_umcsent[df_umcsent['ds'] >= start_date]  # Restrict to the same period as stock data

    # Ensure the data is sorted by date
    df_umcsent = df_umcsent.sort_values(by='ds')
    # Set the date column as the index
    df_umcsent.set_index('ds', inplace=True)
    # Create a new date range with daily frequency
    daily_index = pd.date_range(start=df_umcsent.index.min(), end=df_umcsent.index.max(), freq='D')
    # Reindex the DataFrame to include the daily dates
    df_umcsent_daily = df_umcsent.reindex(daily_index)
    # Interpolate the missing values
    df_umcsent_daily = df_umcsent_daily.interpolate(method='linear')
    # Reset the index and rename the columns
    df_umcsent_daily.reset_index(inplace=True)
    df_umcsent_daily.rename(columns={'index': 'ds'}, inplace=True)
    return df_umcsent_daily, exo_vars


def get_gdp_growth_data(exo_vars, start_date=None):
    from fredapi import Fred
    # Initialize the FRED API with your API key
    fred = Fred(api_key='e7bc3e975c7ef55f5e3bd20671a88973')
    # Fetch the most recent GDP growth data (Real Gross Domestic Product, Percent Change from Preceding Period)
    gdp_growth = fred.get_series('A191RL1Q225SBEA')
    # Convert the data to a DataFrame
    gdp_growth_df = pd.DataFrame(gdp_growth).reset_index()
    gdp_growth_df.columns = ['ds', 'gdp_growth']
    if 'gdp_growth' not in exo_vars:
        exo_vars.append('gdp_growth')
    gdp_growth_df['ds'] = pd.to_datetime(gdp_growth_df['ds'])
    # compute 1 year earlier date than start_date
    one_year_earlier_date = pd.to_datetime(start_date) - relativedelta(years=1)
    gdp_growth_df = gdp_growth_df[gdp_growth_df['ds'] >= one_year_earlier_date]


    # Ensure the data is sorted by date
    gdp_growth_df = gdp_growth_df.sort_values(by='ds')
    # Set the date column as the index
    gdp_growth_df.set_index('ds', inplace=True)
    # Create a new date range with daily frequency
    daily_index = pd.date_range(start=gdp_growth_df.index.min(), end=gdp_growth_df.index.max(), freq='D')
    # Reindex the DataFrame to include the daily dates
    gdp_growth_df_daily = gdp_growth_df.reindex(daily_index)
    # Interpolate the missing values
    gdp_growth_df_daily = gdp_growth_df_daily.interpolate(method='linear')
    # Reset the index and rename the columns
    gdp_growth_df_daily.reset_index(inplace=True)
    gdp_growth_df_daily.rename(columns={'index': 'ds'}, inplace=True)
    return gdp_growth_df_daily, exo_vars


def get_mortgage_rate_data(exo_vars, start_date=None):
    from fredapi import Fred
    fred = Fred(api_key='e7bc3e975c7ef55f5e3bd20671a88973')  # Replace with your FRED API key
    # Pull UMCSENT data
    mortgage = fred.get_series('MORTGAGE30US')
    df_mortgage = pd.DataFrame(mortgage).reset_index()
    df_mortgage.columns = ['ds', 'mortgage_rate']
    if 'mortgage_rate' not in exo_vars:
        exo_vars.append('mortgage_rate')
    df_mortgage['ds'] = pd.to_datetime(df_mortgage['ds'])
    df_mortgage = df_mortgage[df_mortgage['ds'] >= start_date]  # Restrict to the same period as stock data

    # Ensure the data is sorted by date
    df_mortgage = df_mortgage.sort_values(by='ds')
    # Set the date column as the index
    df_mortgage.set_index('ds', inplace=True)
    # Create a new date range with daily frequency
    daily_index = pd.date_range(start=df_mortgage.index.min(), end=df_mortgage.index.max(), freq='D')
    # Reindex the DataFrame to include the daily dates
    df_df_mortgage_daily = df_mortgage.reindex(daily_index)
    # Interpolate the missing values
    df_df_mortgage_daily = df_df_mortgage_daily.interpolate(method='linear')
    # Reset the index and rename the columns
    df_df_mortgage_daily.reset_index(inplace=True)
    df_df_mortgage_daily.rename(columns={'index': 'ds'}, inplace=True)
    return df_df_mortgage_daily, exo_vars


# combine all functions for getting federal reserve data into one function
# Federal Reserve data includes: 
# - Consumer Price Index (CPI)
# - Unemployment Rate
# - Federal Funds Rate (Interest Rate)
# - Consumer Behavior (UMCSENT)
# - Mortgage Rate
def get_federal_reserve_data(df_in, exo_vars, start_date=None):
    df = df_in.copy(deep=True)
    # get CPI data
    df_cpi, exo_vars = get_fred_inflation_data(exo_vars, start_date=start_date)
    df = df.merge(df_cpi, on='ds', how='left')
    # get unemployment rate data
    df_unemployment, exo_vars = get_unemployment_rate_data(exo_vars, start_date=start_date)
    df = df.merge(df_unemployment, on='ds', how='left')
    # get interest rate data
    df_interest_rate, exo_vars = get_interest_rate_data(exo_vars, start_date=start_date)
    df = df.merge(df_interest_rate, on='ds', how='left')
    # get consumer behavior data
    df_umcsent, exo_vars = get_consumer_behavior_data(exo_vars, start_date=start_date)
    df = df.merge(df_umcsent, on='ds', how='left')
    # get GDP growth data
    df_gdp_growth, exo_vars = get_gdp_growth_data(exo_vars, start_date=start_date)
    df = df.merge(df_gdp_growth, on='ds', how='left')
    # get mortgage rate data
    df_mortgage, exo_vars = get_mortgage_rate_data(exo_vars, start_date=start_date)
    df = df.merge(df_mortgage, on='ds', how='left')
    return df, exo_vars


#### ------------------------------------------------------------------------------------------------------------------- ####
#### -------------------------------------Functions for model training/prediction--------------------------------------- ####
#### ------------------------------------------------------------------------------------------------------------------- ####

def create_train_test_sets(df_in, split_date=None, start_date=None):
    df = df_in.copy(deep=True)
    # restrict start date for training set
    df = df[df['ds'] >= start_date]
    # drop weekends
    df['dayofweek'] = df['ds'].dt.dayofweek
    df = df[df['ds'].dt.dayofweek < 5]  # Keep only weekdays (Monday to Friday)
    train_df = df[df['ds'] < split_date]
    test_df = df[df['ds'] >= split_date]
    return df, train_df, test_df


def train_prophet_model(df_train, exo_vars):
    model = Prophet(
                changepoint_prior_scale=0.1,
                changepoint_range=0.7,
                n_changepoints=10,
                interval_width=0.7,
                yearly_seasonality=True,
                weekly_seasonality=True,
                seasonality_mode='multiplicative',
                daily_seasonality=False,
                seasonality_prior_scale=0.9,
            )
    model.add_seasonality(name='monthly', period=30, fourier_order=3, prior_scale=2.0)
    model.add_seasonality(name='quarterly', period=91, fourier_order=3, prior_scale=2.0)

    # add lags
    df_train['y_lag1'] = df_train['y'].shift(1)
    df_train['y_lag2'] = df_train['y'].shift(2)
    for lag in ['y_lag1', 'y_lag2']:
        model.add_regressor(lag)
    df_train.dropna(subset=['y_lag1','y_lag2'], inplace=True)

    model.fit(df_train) 
    return model


def create_residuals(df_train, forecast_prophet, exo_vars, eps_cols):
    df = df_train.copy(deep=True)
    df = df.merge(forecast_prophet[['ds', 'yhat']], on='ds', how='left')
    df['residuals'] = df['y'] - df['yhat']
    return df[['ds', 'y', 'yhat', 'residuals']+exo_vars+eps_cols]


def train_xgb_model(df_residuals, exo_vars, eps_cols):
    X = df_residuals[exo_vars+eps_cols]
    y = df_residuals['residuals']
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=500,
        learning_rate=0.1,
        max_depth=5,
        reg_lambda=0.1,
        reg_alpha=0.1,
        gamma=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=420,
    )
    xgb_model.fit(X, y)
    return xgb_model


def make_prophet_forecast(df_in, prophet_model):
    df = df_in.copy(deep=True)
    # make forecast with prophet_model
    future = df[['ds']]

    # add lags to prediction dataframe
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    future = future.merge(df[['ds','y_lag1','y_lag2']], on='ds', how='left')
    future = future.dropna(subset=['y_lag1','y_lag2'])

    forecast = prophet_model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def make_predictions(df_in, df_all, df_test, prophet_model, xgb_model, exo_vars, eps_cols, split_date=None):
    df = df_in.copy(deep=True)
    df = df.sort_values(by=['ds'])
    # make forecast with prophet_model
    future = df_test[['ds']]

    # add lags to prediction dataframe
    df1 = df_all.copy(deep=True)
    df1['y_lag1'] = df1['y'].shift(1)
    df1['y_lag2'] = df1['y'].shift(2)
    future = future.merge(df1[['ds','y_lag1','y_lag2']], on='ds', how='left')
    future = future.dropna(subset=['y_lag1','y_lag2'])

    forecast = prophet_model.predict(future)
    forecast = forecast[forecast['ds'] >= split_date]

    # make residuals predictions with xgb_model
    X_test = df_test[df_test['ds'] >= split_date][exo_vars+eps_cols]
    forecast['residuals'] = xgb_model.predict(X_test)
    # combine predictions
    forecast['preds'] = forecast['yhat'] + forecast['residuals']
    return forecast


# In this function, we assume we do not have the actual values for the test set
# Therefore, we need to predict future values of exogenous variables i.e., for future ds
def predict_future_exo_vars(df_train, df_test, exo_vars, lag_vars=None):
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_exo_vars_forecast = df[['ds']].copy(deep=True)
    print(exo_vars)
    # making raw forecasts for exogenous variables
    for var in exo_vars:
        print(var)
        # m = Prophet(
        #         changepoint_prior_scale=0.2,
        #         changepoint_range=0.8,
        #         n_changepoints=10,
        #         interval_width=0.7,
        #         yearly_seasonality=True,
        #         weekly_seasonality=False,
        #         seasonality_mode='multiplicative',
        #         daily_seasonality=False,
        #         seasonality_prior_scale=0.5,
        #     )
        m = Prophet()
        # m.add_seasonality(name='monthly', period=30, fourier_order=7, prior_scale=0.5)
        # m.add_seasonality(name='quarterly', period=91, fourier_order=7, prior_scale=0.5)
        df_train_var = df_train[['ds', var]].rename(columns={var: 'y'})
        m.fit(df_train_var)
        future = df[['ds']].copy(deep=True)
        forecast = m.predict(future)
        df_exo_vars_forecast = df_exo_vars_forecast.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        df_exo_vars_forecast.rename(columns={'yhat': var}, inplace=True)
    # making raw forecasts for lag-vars
    for var in lag_vars:
        print(var)
        m = Prophet()
        df_train_var = df_train[['ds', var]].rename(columns={var: 'y'})
        m.fit(df_train_var)
        future = df[['ds']].copy(deep=True)
        forecast = m.predict(future)
        df_exo_vars_forecast = df_exo_vars_forecast.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        df_exo_vars_forecast.rename(columns={'yhat': var}, inplace=True)
    # for eps variables we use the earnings dates
    with open('inputs.yml', 'r') as file:
        inputs = yaml.safe_load(file)
    earnings, _ = get_earnings_dates(sticker=inputs['params']['stock_sticker'])
    df_exo_vars_forecast = df_exo_vars_forecast.merge(earnings[['ds', 'eps_estimate', 'reported_eps', 'surprise_pct']], on='ds', how='left')
    return df_exo_vars_forecast


def predict_with_unk_future_exo_vars(df_in, df_train, df_test, prophet_model, xgb_model, exo_vars, eps_cols, lag_vars=None, split_date=None):
    df = df_in.copy(deep=True)
    df_test.reset_index(drop=True, inplace=True)
    # future = model.make_future_dataframe(periods=len(df_test), freq='D')
    future = df[['ds']]
    future = future.merge(df[['ds']+lag_vars], on='ds', how='left') # add lag vars
    forecast = prophet_model.predict(future)
    forecast = forecast[forecast['ds'] >= split_date]
    # making predictions with xgb_model for residuals
    X_test = df[df['ds'] >= split_date][exo_vars+eps_cols]

    print("Using the following exogeneous variables: ",exo_vars)

    forecast['residuals'] = xgb_model.predict(X_test)
    # combine predictions
    forecast['preds'] = forecast['yhat'] + forecast['residuals']
    forecast = forecast[['ds', 'yhat', 'residuals', 'preds']].copy(deep=True)
    
    return forecast


def calculate_performance_scores(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return mape, rmse, mae



def predict_fed_data(df_in, feature_name=None):
    # function for simple predicting Federal data for missing values
    # this has to be done since the Federal data is not available until end of month 
    # so there might be a gap of time between now and latest available date for Federal data
    df = df_in.copy(deep=True)

    df[feature_name] = df[feature_name].interpolate(method='linear')

    # Create a date range starting from the last date in the DataFrame to today_date
    future_dates = pd.date_range(start=df['ds'].max() + pd.Timedelta(days=1), end=datetime.today().date(), freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    # Assume a simple trend for CPI (e.g., using the last known rate of change)
    last_known_data = df[feature_name].iloc[-1]
    data_rate_of_change = df[feature_name].pct_change().mean()  # Average percentage change
    future_df[feature_name] = last_known_data * (1 + data_rate_of_change) ** (future_df.index + 1)
    df = pd.concat([df, future_df], axis=0, ignore_index=True)
    return df

# Update get_federal_reserve_data function to use predict_fed_data
# This function will now predict the Federal Reserve data for the missing values and up-to present date
#           since federal data updates are not available until end of month
def get_federal_reserve_data(df_in, exo_vars, start_date=None):
    df = df_in.copy(deep=True)

    # import CPI data
    df_cpi, exo_vars = get_fred_inflation_data(exo_vars, start_date=start_date)
    df_cpi = predict_fed_data(df_cpi, feature_name='cpi')
    df = df.merge(df_cpi, on='ds', how='left')

    # import unemployment rate data
    df_unemployment, exo_vars = get_unemployment_rate_data(exo_vars, start_date=start_date)
    df_unemployment = predict_fed_data(df_unemployment, feature_name='unemployment_rate')
    df = df.merge(df_unemployment, on='ds', how='left')

    # import interest rate data
    df_interest_rate, exo_vars = get_interest_rate_data(exo_vars, start_date=start_date)
    df_interest_rate = predict_fed_data(df_interest_rate, feature_name='interest_rate')
    df = df.merge(df_interest_rate, on='ds', how='left')

    # import consumer behavior data (umcsent)
    df_umcsent, exo_vars = get_consumer_behavior_data(exo_vars, start_date=start_date)
    df_umcsent = predict_fed_data(df_umcsent, feature_name='umcsent')
    df = df.merge(df_umcsent, on='ds', how='left')

    # import GDP growth data
    df_gdp_growth, exo_vars = get_gdp_growth_data(exo_vars, start_date=start_date)
    df_gdp_growth = predict_fed_data(df_gdp_growth, feature_name='gdp_growth')
    df = df.merge(df_gdp_growth, on='ds', how='left')

    # import mortgage rate data
    df_mortgage, exo_vars = get_mortgage_rate_data(exo_vars, start_date=start_date)
    df_mortgage = predict_fed_data(df_mortgage, feature_name='mortgage_rate')
    df = df.merge(df_mortgage, on='ds', how='left')
    return df, exo_vars


def generate_future_exogeneous_vars_forecasts(df_in, exo_vars, lag_vars=None, start_fc_date=None, end_fc_date=None):
    df = df_in.copy(deep=True)

    start_fc_date = pd.to_datetime(start_fc_date)
    end_fc_date   = pd.to_datetime(end_fc_date)
    num_days = (end_fc_date - start_fc_date).days
    df_exo_vars_forecast = pd.DataFrame({'ds': pd.date_range(start=start_fc_date, end=end_fc_date, freq='D')})
    df_exo_vars_forecast = df_exo_vars_forecast[df_exo_vars_forecast['ds'].dt.dayofweek < 5]  # Keep only weekdays (Monday to Friday)

    for var in exo_vars:
        print(var)
        # m = Prophet(
        #         changepoint_prior_scale=0.2,
        #         changepoint_range=0.8,
        #         n_changepoints=10,
        #         interval_width=0.7,
        #         yearly_seasonality=True,
        #         weekly_seasonality=False,
        #         seasonality_mode='multiplicative',
        #         daily_seasonality=False,
        #         seasonality_prior_scale=0.3,
        #     )
        m = Prophet()
        # m.add_seasonality(name='monthly', period=30, fourier_order=2, prior_scale=1.0)
        # m.add_seasonality(name='quarterly', period=91, fourier_order=2, prior_scale=1.0)
        df_train_var = df[['ds', var]].rename(columns={var: 'y'})
        m.fit(df_train_var)

        future = m.make_future_dataframe(periods=num_days, freq='D')
        future = future[future['ds'].dt.dayofweek < 5]  # Keep only weekdays (Monday to Friday)
        forecast = m.predict(future)
        df_exo_vars_forecast = df_exo_vars_forecast.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        df_exo_vars_forecast.rename(columns={'yhat': var}, inplace=True)

    # create lags then forecast into future
    df['y_lag1'] = df['y'].shift(1)
    df['y_lag2'] = df['y'].shift(2)
    df.dropna(subset=lag_vars, inplace=True)
    for var in lag_vars:
        print(var)
        m = Prophet()
        df_train_var = df[['ds', var]].rename(columns={var: 'y'})
        m.fit(df_train_var)

        future = m.make_future_dataframe(periods=num_days, freq='D')
        future = future[future['ds'].dt.dayofweek < 5]  # Keep only weekdays (Monday to Friday)
        forecast = m.predict(future)
        df_exo_vars_forecast = df_exo_vars_forecast.merge(forecast[['ds', 'yhat']], on='ds', how='left')
        df_exo_vars_forecast.rename(columns={'yhat': var}, inplace=True)
    return df_exo_vars_forecast


def predict_with_forecasted_exo_vars(df_exo_vars_forecast_in, 
                                     prophet_model, xgb_model, 
                                     exo_vars, eps_cols, lag_vars=None,
                                     start_fc_dt=None,
                                     end_fc_dt=None,
                                     ):
    df = df_exo_vars_forecast_in.copy(deep=True)

    start_fc_dt = pd.to_datetime(start_fc_dt)
    end_fc_dt   = pd.to_datetime(end_fc_dt)
    num_days = (end_fc_dt - start_fc_dt).days + 1
    # future = model.make_future_dataframe(periods=len(df_test), freq='D')
    future = prophet_model.make_future_dataframe(periods=num_days, freq='D')
    # incorporate predicted lag_vars into future df to make Prophet forecast
    future = future.merge(df[['ds']+lag_vars], on='ds', how='left')
    print(future.shape)
    future.dropna(subset=lag_vars, inplace=True)
    future = future[future['ds'].dt.dayofweek < 5]  # Keep only weekdays (Monday to Friday)
    forecast = prophet_model.predict(future)
    forecast_historical = forecast[forecast['ds'] < start_fc_dt]
    forecast = forecast[forecast['ds'] >= start_fc_dt]

    # making predictions with xgb_model for residuals
    X_test = df[df['ds'] >= start_fc_dt][exo_vars+eps_cols]

    print(forecast.shape)
    print(X_test.shape)

    print(exo_vars)

    forecast['residuals'] = xgb_model.predict(X_test)
    # combine predictions
    forecast['preds']       = forecast['yhat'] + forecast['residuals']
    forecast['preds_upper'] = forecast['yhat_upper'] + forecast['residuals']
    forecast['preds_lower'] = forecast['yhat_lower'] + forecast['residuals']
    forecast = forecast[['ds', 'yhat', 'residuals', 'preds', 'preds_upper', 'preds_lower']].copy(deep=True)
    
    return forecast, forecast_historical[['ds', 'yhat']]





#### ------------------------------------------------------------------------------------------------------------------- ####
#### ---------------------------Functions for getting NEWS data and NEWS sentiment analysis----------------------------- ####
#### ------------------------------------------------------------------------------------------------------------------- ####

def extract_historical_news_sentiment(exo_vars, folder_path=None):
    import os

    # Read all CSV files in the folder
    all_dataframes = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            all_dataframes.append(df)

    # Combine all dataframes into one
    df = pd.concat(all_dataframes, ignore_index=True)
    df.reset_index(drop=True)
    df.rename(columns={"DATE":"ds"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["ds"])
    if "SentimentScore" not in exo_vars:
        exo_vars.append("SentimentScore")
    return df, exo_vars







#### ------------------------------------------------------------------------------------------------------------------- ####
#### -------------------------------------Functions for analysis stocks features---------------------------------------- ####
#### ------------------------------------------------------------------------------------------------------------------- ####

def stock_analysis_plot(df_in, stock_sticker=None, zoom_start_date=None, zoom_end_date=None):
    df = df_in.copy(deep=True)
    df.rename(columns={'Date': 'ds', f'Close_{stock_sticker}': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])

    # Plot MACD, EMA-9 of MACD, and Close Price
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(15, 15))
    ax1.plot(df['ds'], df['MACD'],          label=f'{stock_sticker} MACD', color='blue', ls='-')
    ax1.plot(df['ds'], df['EMA_9_of_MACD'], label=f'{stock_sticker} EMA9(MACD)', color='red', ls='--')
    ax1.axhline(y=0, color='k', linestyle='--', label='y=0')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MACD vs. EMA-9 of MACD')
    ax1.legend()

    ax2.plot(df['ds'], df['MACD']-df['EMA_9_of_MACD'], label='EMA_9-MACD', color='purple', ls='-')
    ax2.axhline(y=0, color='k', linestyle='--', label='y=0')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('EMA-9 of MACD - MACD')
    ax2.legend()

    ax3.plot(df['ds'], df['y'],       label=f'{stock_sticker} Close Price', color='green', ls='-')
    ax3.plot(df['ds'], df['BB_High'], label='Bollinger High', color='red', ls='--')
    ax3.plot(df['ds'], df['BB_Low'],  label='Bollinger Low', color='blue', ls='--')
    ax3.set_xlabel('Date')
    ax3.set_ylabel(f'{stock_sticker} Close Price')
    ax3.legend()

    # generate same plot as ax3 but with zoom in over a time interval
    fig, ax = plt.subplots(1,1,figsize=(12, 4))
    zoom_df = df[(df['ds'] >= zoom_start_date) & (df['ds'] <= zoom_end_date)]
    ax.plot(zoom_df['ds'], zoom_df['y'],       label=f'{stock_sticker} Close Price (Zoomed)', color='green', ls='-')
    ax.plot(zoom_df['ds'], zoom_df['BB_High'], label='Bollinger High (Zoomed)', color='red', ls='--')
    ax.plot(zoom_df['ds'], zoom_df['BB_Low'],  label='Bollinger Low (Zoomed)', color='blue', ls='--')
    ax.set_xlim(pd.to_datetime(zoom_start_date), pd.to_datetime(zoom_end_date))
    ax.set_ylabel(f'{stock_sticker} Close Price (Zoomed)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot RSI
    fig, ax = plt.subplots(1,1,figsize=(12, 4))
    ax.plot(df['ds'], df['RSI'], label=f'{stock_sticker} RSI', color='orange', ls='-')
    ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
    ax.axhline(y=30, color='blue', linestyle='--', label='Oversold (30)')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()
    plt.show()