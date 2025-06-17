"""
Functions for fetching and processing economic data from the Federal Reserve Economic Data (FRED) API.

This module provides utilities to fetch key economic indicators from FRED and save them
to CSV files for further analysis.
"""

import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Define economic indicators to fetch
ECONOMIC_INDICATORS = {
    'GDP': 'GDPC1',              # Real Gross Domestic Product
    'UNEMPLOYMENT': 'UNRATE',    # Unemployment Rate
    'CPI': 'CPIAUCSL',           # Consumer Price Index for All Urban Consumers
    'FED_FUNDS': 'FEDFUNDS',     # Federal Funds Effective Rate
    'YIELD_CURVE': 'T10Y2Y',     # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    'INITIAL_CLAIMS': 'ICSA',    # Initial Claims
    'INDUSTRIAL_PROD': 'INDPRO', # Industrial Production Index
    'RETAIL_SALES': 'RSAFS',     # Advance Retail Sales: Retail and Food Services
    'HOUSING_STARTS': 'HOUST',   # Housing Starts: Total: New Privately Owned Housing Units Started
    'CONSUMER_SENTIMENT': 'UMCSENT' # University of Michigan: Consumer Sentiment
}

# Configure logging
logger = logging.getLogger(__name__)


def fetch_fred_data(api_key, start_date='1970-01-01', end_date=None):
    """
    Fetch economic indicators from FRED API.
    
    Parameters
    ----------
    api_key : str
        FRED API key
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format, defaults to current date
        
    Returns
    -------
    dict
        Dictionary of pandas DataFrames with economic indicators
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    fred = Fred(api_key=api_key)
    data = {}

    for name, series_id in ECONOMIC_INDICATORS.items():
        try:
            logger.info(f"Fetching {name} (Series ID: {series_id})")
            series = fred.get_series(series_id, start_date, end_date)
            data[name] = pd.DataFrame({name: series})
            logger.info(f"Successfully fetched {name} with {len(series)} observations")
        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")

    return data


def save_data(data, output_dir='data/fred'):
    """
    Save fetched data to CSV files.
    
    Parameters
    ----------
    data : dict
        Dictionary of pandas DataFrames with economic indicators
    output_dir : str
        Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save individual indicators
    for name, df in data.items():
        output_path = os.path.join(output_dir, f"{name.lower()}.csv")
        df.to_csv(output_path)
        logger.info(f"Saved {name} data to {output_path}")

    # Create a merged dataset with all indicators
    merged_data = None
    for name, df in data.items():
        if merged_data is None:
            merged_data = df.copy()
        else:
            merged_data = merged_data.join(df, how='outer')

    if merged_data is not None:
        merged_path = os.path.join(output_dir, "all_indicators.csv")
        merged_data.to_csv(merged_path)
        logger.info(f"Saved merged data to {merged_path}")
        
    return merged_data


def get_fred_data(api_key=None, start_date='1970-01-01', end_date=None, output_dir='data/fred'):
    """
    Convenience function to fetch and save FRED data in one step.
    
    Parameters
    ----------
    api_key : str, optional
        FRED API key, if None will try to get from environment variable
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format, defaults to current date
    output_dir : str
        Directory to save the CSV files
        
    Returns
    -------
    pandas.DataFrame
        Merged dataset with all indicators
    """
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            logger.error("FRED API key not found. Set the FRED_API_KEY environment variable.")
            return None

    # Fetch and save data
    data = fetch_fred_data(api_key, start_date, end_date)
    merged_data = save_data(data, output_dir)
    
    return merged_data
