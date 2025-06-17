"""
University of Michigan Consumer Sentiment Data Module

This module provides functions to fetch and process University of Michigan
Consumer Sentiment data from the FRED API.
"""

import os
import pandas as pd
import logging
from datetime import datetime
from fredapi import Fred

# Configure logging
logger = logging.getLogger(__name__)

# Define UMich sentiment indicators available from FRED
UMICH_INDICATORS = {
    'SENTIMENT': 'UMCSENT',     # University of Michigan: Consumer Sentiment
    'INFLATION_EXPECTATION': 'MICH'  # University of Michigan: Inflation Expectation
}


def fetch_umich_data(api_key, start_date='1970-01-01', end_date=None):
    """
    Fetch University of Michigan Consumer Sentiment data from FRED API.
    
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
        Dictionary of pandas DataFrames with UMich sentiment indicators
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    fred = Fred(api_key=api_key)
    data = {}

    for name, series_id in UMICH_INDICATORS.items():
        try:
            logger.info(f"Fetching {name} (Series ID: {series_id})")
            series = fred.get_series(series_id, start_date, end_date)
            data[name] = pd.DataFrame({name: series})
            logger.info(f"Successfully fetched {name} with {len(series)} observations")
        except Exception as e:
            logger.error(f"Error fetching {name}: {e}")

    return data


def save_data(data, output_dir='data/umich'):
    """
    Save UMich sentiment data to CSV files.
    
    Parameters
    ----------
    data : dict
        Dictionary of pandas DataFrames with UMich sentiment indicators
    output_dir : str
        Directory to save the CSV files
        
    Returns
    -------
    pandas.DataFrame
        Merged dataset with all indicators
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual series
    data_frames = []
    for name, df in data.items():
        # Save individual file
        output_path = os.path.join(output_dir, f"{name.lower()}.csv")
        df.to_csv(output_path)
        logger.info(f"Saved {name} to {output_path}")
        data_frames.append(df)
    
    # Merge all DataFrames
    if data_frames:
        merged_data = pd.concat(data_frames, axis=1)
        
        # Save merged data
        merged_path = os.path.join(output_dir, "all_sentiment.csv")
        merged_data.to_csv(merged_path)
        logger.info(f"Saved merged UMich data to {merged_path}")
        
        return merged_data
    else:
        logger.warning("No UMich data to save")
        return pd.DataFrame()


def get_umich_data(api_key=None, start_date='1970-01-01', end_date=None, output_dir='data/umich'):
    """
    Convenience function to fetch and save UMich sentiment data in one step.
    
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
        Merged dataset with all UMich sentiment indicators
    """
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            logger.error("FRED API key not found. Set the FRED_API_KEY environment variable.")
            return None
    
    # Fetch UMich sentiment data
    data = fetch_umich_data(api_key, start_date, end_date)
    
    if data:
        # Save data and return merged dataset
        merged_data = save_data(data, output_dir)
        
        logger.info("UMich sentiment data processing completed successfully")
        return merged_data
    else:
        logger.error("Failed to fetch UMich sentiment data")
        return None
