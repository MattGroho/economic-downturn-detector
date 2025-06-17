"""
Functions for normalizing and transforming features.

This module provides utilities to normalize features, apply dimensionality reduction,
and prepare data for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

# Configure logging
logger = logging.getLogger(__name__)


def normalize_data(data, method='standard', target_col='recession'):
    """
    Normalize the features in the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    method : str
        Normalization method:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (range [0, 1])
    target_col : str
        Name of the target column to exclude from normalization

    Returns
    -------
    pandas.DataFrame
        Dataset with normalized features
    """
    # Separate features and target
    target = None
    if target_col in data.columns:
        target = data[target_col].copy()
        features = data.drop(columns=[target_col])
    else:
        features = data.copy()

    # Check for and handle infinite values
    inf_mask = np.isinf(features).any(axis=1)
    if inf_mask.any():
        logger.warning(f"Found {inf_mask.sum()} rows with infinite values. Replacing with NaN.")
        features = features.replace([np.inf, -np.inf], np.nan)

    # Check for and handle NaN values
    nan_mask = features.isnull().any(axis=1)
    if nan_mask.any():
        logger.warning(f"Found {nan_mask.sum()} rows with NaN values. Forward filling and backward filling.")
        features = features.fillna(method='ffill').fillna(method='bfill')

        # If there are still NaN values (e.g., entire columns are NaN), fill with 0
        remaining_nan = features.isnull().any(axis=1)
        if remaining_nan.any():
            logger.warning(f"Still have {remaining_nan.sum()} rows with NaN after filling. Filling with 0.")
            features = features.fillna(0)

    # Apply normalization
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        logger.warning(f"Unknown method '{method}', using 'standard' instead")
        scaler = StandardScaler()

    # Fit and transform the features
    features_normalized = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    # Add back the target
    if target is not None:
        features_normalized[target_col] = target

    logger.info(f"Normalized data using {method} method, shape: {features_normalized.shape}")

    return features_normalized, scaler


def apply_pca(data, n_components=0.95, target_col='recession'):
    """
    Apply Principal Component Analysis (PCA) for dimensionality reduction.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    n_components : float or int
        Number of components to keep:
        - If float between 0 and 1: Percentage of variance to retain
        - If int > 1: Number of components to keep
    target_col : str
        Name of the target column to exclude from PCA
        
    Returns
    -------
    pandas.DataFrame
        Dataset with PCA components
    sklearn.decomposition.PCA
        Fitted PCA model
    """
    # Separate features and target
    target = None
    if target_col in data.columns:
        target = data[target_col].copy()
        features = data.drop(columns=[target_col])
    else:
        features = data.copy()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    # Create a DataFrame with PCA components
    pca_cols = [f'PC{i+1}' for i in range(features_pca.shape[1])]
    features_pca_df = pd.DataFrame(
        features_pca,
        columns=pca_cols,
        index=features.index
    )
    
    # Add back the target
    if target is not None:
        features_pca_df[target_col] = target
    
    # Log PCA results
    if isinstance(n_components, float):
        logger.info(f"Applied PCA retaining {n_components*100:.1f}% of variance")
    else:
        logger.info(f"Applied PCA with {n_components} components")
    
    logger.info(f"Number of PCA components: {features_pca.shape[1]}")
    logger.info(f"Cumulative explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return features_pca_df, pca


def prepare_data_for_modeling(data, normalize_method='standard', apply_pca_flag=True, 
                             pca_components=0.95, target_col='recession'):
    """
    Prepare data for modeling by normalizing and optionally applying PCA.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    normalize_method : str
        Normalization method:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (range [0, 1])
    apply_pca_flag : bool
        Whether to apply PCA
    pca_components : float or int
        Number of components to keep:
        - If float between 0 and 1: Percentage of variance to retain
        - If int > 1: Number of components to keep
    target_col : str
        Name of the target column
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'normalized': Normalized dataset
        - 'pca': PCA dataset (if apply_pca_flag is True)
        - 'scaler': Fitted scaler
        - 'pca_model': Fitted PCA model (if apply_pca_flag is True)
    """
    results = {}
    
    # Normalize the data
    normalized_data, scaler = normalize_data(data, method=normalize_method, target_col=target_col)
    results['normalized'] = normalized_data
    results['scaler'] = scaler
    
    # Apply PCA if requested
    if apply_pca_flag:
        pca_data, pca_model = apply_pca(normalized_data, n_components=pca_components, target_col=target_col)
        results['pca'] = pca_data
        results['pca_model'] = pca_model
    
    return results
