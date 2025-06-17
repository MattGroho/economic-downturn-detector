"""
Functions for fetching and processing NBER recession data.

This module provides utilities to fetch recession dates from the National Bureau of Economic
Research (NBER) and create recession indicator time series.
"""

import os
import pandas as pd
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Historical recession dates (peak to trough)
RECESSION_DATES = [
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]


def fetch_nber_data():
    """
    Create a DataFrame with NBER recession dates.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing recession start and end dates
    """
    try:
        logger.info("Creating NBER recession data")
        
        df = pd.DataFrame(RECESSION_DATES, columns=['peak', 'trough'])
        df['peak'] = pd.to_datetime(df['peak'])
        df['trough'] = pd.to_datetime(df['trough'])
        
        logger.info(f"Successfully created data for {len(df)} recessions")
        return df
        
    except Exception as e:
        logger.error(f"Error creating NBER recession data: {e}")
        return pd.DataFrame()


def create_recession_indicator(recession_df, start_date='1970-01-01', end_date=None, freq='M'):
    """
    Create a time series with recession indicators (1 during recession, 0 otherwise).
    
    Parameters
    ----------
    recession_df : pandas.DataFrame
        DataFrame containing recession start and end dates
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format, defaults to current date
    freq : str
        Frequency of the time series ('M' for monthly, 'Q' for quarterly)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with recession indicator time series
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    recession_indicator = pd.DataFrame(index=date_range)
    recession_indicator['recession'] = 0
    
    # Mark recession periods
    for _, row in recession_df.iterrows():
        mask = (recession_indicator.index >= row['peak']) & (recession_indicator.index <= row['trough'])
        recession_indicator.loc[mask, 'recession'] = 1
    
    return recession_indicator


def save_data(recession_df, recession_indicator, output_dir='data/nber'):
    """
    Save NBER recession data to CSV files.
    
    Parameters
    ----------
    recession_df : pandas.DataFrame
        DataFrame containing recession start and end dates
    recession_indicator : pandas.DataFrame
        DataFrame with recession indicator time series
    output_dir : str
        Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save recession dates
    recession_dates_path = os.path.join(output_dir, "recession_dates.csv")
    recession_df.to_csv(recession_dates_path, index=False)
    logger.info(f"Saved recession dates to {recession_dates_path}")
    
    # Save recession indicator
    recession_indicator_path = os.path.join(output_dir, "recession_indicator.csv")
    recession_indicator.to_csv(recession_indicator_path)
    logger.info(f"Saved recession indicator to {recession_indicator_path}")
    
    return recession_indicator


def get_nber_data(start_date='1970-01-01', end_date=None, output_dir='data/nber'):
    """
    Convenience function to fetch and save NBER data in one step.
    
    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format, defaults to current date
    output_dir : str
        Directory to save the CSV files
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with recession indicator time series
    """
    # Fetch recession data
    recession_df = fetch_nber_data()
    
    if not recession_df.empty:
        # Create recession indicator
        recession_indicator = create_recession_indicator(
            recession_df, start_date, end_date
        )
        
        # Save data
        save_data(recession_df, recession_indicator, output_dir)
        
        logger.info("NBER data processing completed successfully")
        return recession_indicator
    else:
        logger.error("Failed to create NBER recession data")
        return None
