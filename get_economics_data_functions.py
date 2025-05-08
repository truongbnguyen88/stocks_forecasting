from fredapi import Fred
import pandas as pd

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