"""
Advanced feature engineering functions.

This module contains advanced feature engineering functions for the economic downturn detector,
including custom lag creation, interaction terms, and sentiment transformations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

def engineer_features_with_custom_lags(data, sentiment_lags=[1, 3, 6, 12, 18, 24], other_lags=[1, 3, 6, 12]):
    """
    Engineer features with custom lag periods for sentiment and other indicators.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    sentiment_lags : list
        List of lag periods for sentiment indicators
    other_lags : list
        List of lag periods for other economic indicators
        
    Returns
    -------
    pandas.DataFrame
        Dataset with engineered features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Handle missing values intelligently
    df = _handle_missing_values_intelligently(df)
    
    # Resample to monthly frequency if needed
    if df.index.freq != 'M':
        df = df.resample('M').last()
        print(f"Resampled data to M frequency, new shape: {df.shape}")
    
    # Identify sentiment columns
    sentiment_cols = [col for col in df.columns if 'SENTIMENT' in col]
    
    # Create lag variables for sentiment columns with custom lags
    for col in sentiment_cols:
        for lag in sentiment_lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Create lag variables for other columns
    other_cols = [col for col in df.columns if col not in sentiment_cols and col != 'recession']
    for col in other_cols:
        for lag in other_lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    print(f"Created lag variables, new shape: {df.shape}")
    
    # Calculate rate of change for all columns except recession
    for col in df.columns:
        if col != 'recession':
            # 1-month rate of change
            roc1 = df[col].pct_change(1)
            roc1 = roc1.replace([np.inf, -np.inf], np.nan)
            df[f"{col}_roc1"] = roc1

            # 3-month rate of change
            roc3 = df[col].pct_change(3)
            roc3 = roc3.replace([np.inf, -np.inf], np.nan)
            df[f"{col}_roc3"] = roc3

            # 12-month rate of change
            roc12 = df[col].pct_change(12)
            roc12 = roc12.replace([np.inf, -np.inf], np.nan)
            df[f"{col}_roc12"] = roc12
    
    print(f"Calculated rate of change, new shape: {df.shape}")

    # Replace any remaining infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # More intelligent handling of missing values after lag creation
    print(f"Data shape before handling missing values: {df.shape}")

    # Check which rows have missing values and why
    missing_mask = df.isnull().any(axis=1)
    print(f"Rows with missing values: {missing_mask.sum()}")

    # Special handling for recession periods - try to preserve them
    if 'recession' in df.columns:
        recession_rows = df[df['recession'] == 1]
        if len(recession_rows) > 0:
            recession_missing = recession_rows.isnull().any(axis=1)
            if recession_missing.any():
                print(f"WARNING: {recession_missing.sum()} recession period rows have missing values")
                print("Recession periods with missing data:")
                print(recession_rows[recession_missing].index.tolist())

    # Instead of dropping all rows with any missing values, be more selective
    # Only drop rows where lag variables are missing (expected at the beginning)
    # but preserve rows where only a few features are missing

    # Calculate missing percentage per row
    missing_percentage = df.isnull().sum(axis=1) / len(df.columns)

    # Drop rows with more than 50% missing values (these are likely from lag creation)
    rows_to_drop = missing_percentage > 0.5

    if rows_to_drop.any():
        print(f"Dropping {rows_to_drop.sum()} rows with >50% missing values")
        df = df[~rows_to_drop]

    # For remaining rows with some missing values, use forward fill
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Forward-filling {remaining_missing} remaining missing values")
        df = df.fillna(method='ffill').fillna(method='bfill')

        # If still missing, use interpolation
        if df.isnull().sum().sum() > 0:
            df = df.interpolate(method='linear')

    print(f"Data shape after intelligent missing value handling: {df.shape}")

    return df

def create_interaction_terms(data):
    """
    Create interaction terms between consumer sentiment and economic indicators.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset

    Returns
    -------
    pandas.DataFrame
        Dataset with interaction terms
    """
    # Make a copy of the data
    df = data.copy()

    # Identify sentiment columns (base columns, not lagged or transformed)
    sentiment_cols = [col for col in df.columns if col == 'SENTIMENT']

    # Identify key economic indicators (base columns, not lagged or transformed)
    key_indicators = ['GDP', 'UNEMPLOYMENT', 'FED_FUNDS', 'YIELD_CURVE']

    # Create interaction terms
    for sentiment_col in sentiment_cols:
        for indicator in key_indicators:
            if sentiment_col in df.columns and indicator in df.columns:
                # Create interaction term
                df[f"{sentiment_col}_x_{indicator}"] = df[sentiment_col] * df[indicator]

                # Create ratio term (safe division)
                denominator = df[indicator].replace(0, np.nan)
                ratio = df[sentiment_col] / denominator
                # Replace infinite values with NaN
                ratio = ratio.replace([np.inf, -np.inf], np.nan)
                df[f"{sentiment_col}_div_{indicator}"] = ratio

                # Create difference term
                df[f"{sentiment_col}_minus_{indicator}"] = df[sentiment_col] - df[indicator]

    # Fill NaN values that might have been created
    df = df.fillna(method='ffill').fillna(method='bfill')

    print(f"Created interaction terms, new shape: {df.shape}")

    return df

def apply_sentiment_transformations(data):
    """
    Apply different transformations to consumer sentiment data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset

    Returns
    -------
    pandas.DataFrame
        Dataset with transformed sentiment features
    """
    # Make a copy of the data
    df = data.copy()

    # Identify sentiment columns (base columns, not lagged or transformed)
    sentiment_cols = [col for col in df.columns if col == 'SENTIMENT']

    for col in sentiment_cols:
        if col in df.columns:
            # Log transformation (safe)
            log_values = np.log(df[col].replace(0, np.nan))
            log_values = log_values.replace([np.inf, -np.inf], np.nan)
            df[f"{col}_log"] = log_values

            # Square root transformation (safe for positive values)
            sqrt_values = np.sqrt(np.abs(df[col]))  # Use abs to handle negative values
            df[f"{col}_sqrt"] = sqrt_values

            # Square transformation
            df[f"{col}_squared"] = df[col] ** 2

            # Z-score normalization (safe)
            col_std = df[col].std()
            if col_std > 0:  # Avoid division by zero
                zscore_values = (df[col] - df[col].mean()) / col_std
                zscore_values = zscore_values.replace([np.inf, -np.inf], np.nan)
                df[f"{col}_zscore"] = zscore_values
            else:
                df[f"{col}_zscore"] = 0  # If std is 0, all values are the same

            # Min-max scaling (safe)
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:  # Avoid division by zero
                minmax_values = (df[col] - col_min) / (col_max - col_min)
                df[f"{col}_minmax"] = minmax_values
            else:
                df[f"{col}_minmax"] = 0  # If min equals max, all values are the same

            # Moving average (3-month)
            df[f"{col}_ma3"] = df[col].rolling(window=3).mean()

            # Moving average (6-month)
            df[f"{col}_ma6"] = df[col].rolling(window=6).mean()

            # Moving average (12-month)
            df[f"{col}_ma12"] = df[col].rolling(window=12).mean()

            # Exponential moving average (3-month)
            df[f"{col}_ema3"] = df[col].ewm(span=3).mean()

            # Exponential moving average (6-month)
            df[f"{col}_ema6"] = df[col].ewm(span=6).mean()

            # Exponential moving average (12-month)
            df[f"{col}_ema12"] = df[col].ewm(span=12).mean()

    # Replace any remaining infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values that might have been created
    df = df.fillna(method='ffill').fillna(method='bfill')

    print(f"Applied transformations to sentiment data, new shape: {df.shape}")

    return df

def select_features(X, y, method='anova', k=20):
    """
    Select the most predictive features using different methods.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    method : str
        Feature selection method ('anova', 'rfe')
    k : int
        Number of features to select
        
    Returns
    -------
    pandas.DataFrame
        Selected features
    list
        Names of selected features
    """
    if method == 'anova':
        # ANOVA F-value feature selection
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        # Create a DataFrame with scores and selection status
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_,
            'Selected': selector.get_support()
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False)

        # Show selected features
        selected_scores = feature_scores[feature_scores['Selected']].head(10)

        print(f"Selected {k} features using ANOVA F-value")
        print("Top 10 selected features:")
        print(selected_scores[['Feature', 'Score']])
        print(f"\nAll top 10 features (selected and non-selected):")
        print(feature_scores[['Feature', 'Score', 'Selected']].head(10))
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.support_].tolist()

        # Get feature importance from the final estimator
        feature_importance = np.abs(selector.estimator_.coef_[0])

        # Create a DataFrame with selected features and their importance
        selected_feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': feature_importance
        })
        selected_feature_importance = selected_feature_importance.sort_values('Importance', ascending=False)

        # Also create a DataFrame with all features and their rankings
        feature_rankings = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': selector.ranking_
        })
        feature_rankings = feature_rankings.sort_values('Ranking')

        print(f"Selected {k} features using Recursive Feature Elimination")
        print("Top 10 selected features by importance:")
        print(selected_feature_importance.head(10))
        print("\nRFE Rankings (1 = selected, >1 = eliminated in that step):")
        print(feature_rankings.head(10))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame with selected features
    X_selected_df = X[selected_features]
    
    return X_selected_df, selected_features


def _handle_missing_values_intelligently(df):
    """
    Handle missing values intelligently based on the nature of each economic indicator.

    This function specifically addresses the 2020 recession data issue by:
    1. Forward-filling GDP (quarterly data)
    2. Intelligently handling other economic indicators
    3. Preserving critical recession periods

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with economic indicators

    Returns
    -------
    pandas.DataFrame
        Dataset with intelligently handled missing values
    """
    print("Applying intelligent missing value handling...")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # GDP is reported quarterly, so forward-fill to carry values between quarters
    if 'GDP' in df.columns:
        print("Forward-filling GDP values (quarterly data)")
        df['GDP'] = df['GDP'].fillna(method='ffill')

    # Handle other economic indicators with different strategies based on their nature
    economic_indicators = [col for col in df.columns if col not in ['recession', 'GDP']]

    for col in economic_indicators:
        if col in df.columns:
            # Check how many missing values we have
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                print(f"Handling {missing_count} missing values in {col}")

                # For most economic indicators, forward-fill then backward-fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

                # If still missing (edge cases), use interpolation
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].interpolate(method='linear')

                # If still missing (very edge cases), use median
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

    # Special handling for recession indicator - preserve it as is
    # (it should not be forward/backward filled as it represents specific periods)

    # Check for any remaining missing values
    missing_after = df.isnull().sum()
    if missing_after.sum() > 0:
        print(f"Remaining missing values after intelligent handling: {missing_after[missing_after > 0].to_dict()}")
    else:
        print("All missing values successfully handled")

    return df
