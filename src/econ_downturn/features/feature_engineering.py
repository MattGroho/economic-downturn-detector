"""
Functions for feature engineering on economic data.

This module provides utilities to create lag variables, calculate rates of change,
and perform other feature engineering tasks on economic data.
"""

import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)


def handle_missing_values(data, method='ffill_bfill'):
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    method : str
        Method to handle missing values:
        - 'ffill_bfill': Forward fill followed by backward fill
        - 'mean': Replace with mean
        - 'median': Replace with median
        - 'drop': Drop rows with missing values
        
    Returns
    -------
    pandas.DataFrame
        Dataset with handled missing values
    """
    data_filled = data.copy()
    
    # Separate the recession indicator before filling
    recession_col = None
    if 'recession' in data_filled.columns:
        recession_col = data_filled['recession'].copy()
        data_filled = data_filled.drop(columns=['recession'])
    
    # Handle missing values based on the specified method
    if method == 'ffill_bfill':
        data_filled = data_filled.fillna(method='ffill').fillna(method='bfill')
    elif method == 'mean':
        data_filled = data_filled.fillna(data_filled.mean())
    elif method == 'median':
        data_filled = data_filled.fillna(data_filled.median())
    elif method == 'drop':
        data_filled = data_filled.dropna()
    else:
        logger.warning(f"Unknown method '{method}', using 'ffill_bfill' instead")
        data_filled = data_filled.fillna(method='ffill').fillna(method='bfill')
    
    # Add back the recession indicator
    if recession_col is not None:
        data_filled['recession'] = recession_col
    
    # Check for any remaining missing values
    missing_after = data_filled.isnull().sum()
    if missing_after.sum() > 0:
        logger.warning(f"There are still {missing_after.sum()} missing values after handling")
    
    return data_filled


def resample_data(data, freq='M'):
    """
    Resample data to a specified frequency.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    freq : str
        Frequency to resample to:
        - 'M': Monthly
        - 'Q': Quarterly
        - 'Y': Yearly
        
    Returns
    -------
    pandas.DataFrame
        Resampled dataset
    """
    data_resampled = data.resample(freq).last()
    logger.info(f"Resampled data to {freq} frequency, new shape: {data_resampled.shape}")
    
    return data_resampled


def create_lag_variables(data, lag_periods=[1, 3, 6, 12]):
    """
    Create lag variables for each feature in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    lag_periods : list
        List of lag periods to create
        
    Returns
    -------
    pandas.DataFrame
        Dataset with lag variables
    """
    # Separate the recession indicator before creating lags
    recession_col = None
    if 'recession' in data.columns:
        recession_col = data['recession'].copy()
        data_for_lags = data.drop(columns=['recession'])
    else:
        data_for_lags = data.copy()
    
    # Create lag variables for each feature
    data_with_lags = data_for_lags.copy()
    
    for col in data_for_lags.columns:
        for lag in lag_periods:
            data_with_lags[f"{col}_lag{lag}"] = data_for_lags[col].shift(lag)
    
    # Add back the recession indicator
    if recession_col is not None:
        data_with_lags['recession'] = recession_col
    
    logger.info(f"Created lag variables with periods {lag_periods}, new shape: {data_with_lags.shape}")
    
    return data_with_lags


def calculate_rate_of_change(data, periods=[1, 3, 12]):
    """
    Calculate rate of change for each feature in the dataset.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    periods : list
        List of periods to calculate rate of change for
        
    Returns
    -------
    pandas.DataFrame
        Dataset with rate of change variables
    """
    # Separate the recession indicator before calculating rates of change
    recession_col = None
    if 'recession' in data.columns:
        recession_col = data['recession'].copy()
        data_for_roc = data.drop(columns=['recession'])
    else:
        data_for_roc = data.copy()
    
    # Calculate rate of change for original features (not lag variables)
    data_with_roc = data_for_roc.copy()
    
    original_cols = [col for col in data_for_roc.columns if '_lag' not in col]
    
    for col in original_cols:
        for period in periods:
            data_with_roc[f"{col}_pct_change_{period}"] = data_for_roc[col].pct_change(periods=period)
    
    # Add back the recession indicator
    if recession_col is not None:
        data_with_roc['recession'] = recession_col
    
    # Replace infinite values with NaN
    data_with_roc = data_with_roc.replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Calculated rate of change with periods {periods}, new shape: {data_with_roc.shape}")
    
    return data_with_roc


def drop_missing_values(data):
    """
    Drop rows with missing values.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
        
    Returns
    -------
    pandas.DataFrame
        Dataset without missing values
    """
    data_dropped = data.dropna()
    logger.info(f"Dropped rows with missing values, new shape: {data_dropped.shape}")
    
    return data_dropped


def engineer_features(data, lag_periods=[1, 3, 6, 12], roc_periods=[1, 3, 12], freq='M'):
    """
    Perform complete feature engineering pipeline.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    lag_periods : list
        List of lag periods to create
    roc_periods : list
        List of periods to calculate rate of change for
    freq : str
        Frequency to resample to
        
    Returns
    -------
    pandas.DataFrame
        Dataset with engineered features
    """
    # Handle missing values
    data_filled = handle_missing_values(data)
    
    # Resample to consistent frequency
    data_resampled = resample_data(data_filled, freq)
    
    # Create lag variables
    data_with_lags = create_lag_variables(data_resampled, lag_periods)
    
    # Calculate rate of change
    data_with_roc = calculate_rate_of_change(data_with_lags, roc_periods)
    
    # Drop rows with missing values
    data_engineered = drop_missing_values(data_with_roc)
    
    logger.info(f"Completed feature engineering, final shape: {data_engineered.shape}")
    
    return data_engineered
