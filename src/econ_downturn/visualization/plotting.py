"""
Functions for visualizing economic data and model results.

This module provides utilities to create various plots for economic indicators,
recession periods, and model results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Set default plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


def _preprocess_data_for_plotting(data):
    """
    Preprocess data to handle missing values and ensure consistent frequency for plotting.

    This function addresses common issues that cause empty or broken plots:
    - Mixed data frequencies (daily, monthly, etc.)
    - Missing values that break plot continuity
    - Data type inconsistencies

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data to preprocess

    Returns
    -------
    pandas.DataFrame
        Preprocessed data suitable for plotting
    """
    if data is None or data.empty:
        logger.warning("Empty data provided for preprocessing")
        return data

    # Convert to monthly frequency for consistency
    # First, ensure the index is datetime
    data_copy = data.copy()
    data_copy.index = pd.to_datetime(data_copy.index)

    # For each column, resample to monthly frequency
    monthly_data = {}

    for col in data_copy.columns:
        # Skip columns with all NaN values
        if data_copy[col].isna().all():
            continue

        # For economic indicators, use the last value of the month
        if col in ['GDP', 'UNEMPLOYMENT', 'CPI', 'FED_FUNDS', 'YIELD_CURVE',
                  'INDUSTRIAL_PROD', 'RETAIL_SALES', 'HOUSING_STARTS',
                  'CONSUMER_SENTIMENT', 'SENTIMENT', 'INFLATION_1Y']:
            # Resample to monthly frequency, using last value
            series = data_copy[col].resample('M').last()

        # For initial claims (weekly), use the mean of the month
        elif col == 'INITIAL_CLAIMS':
            series = data_copy[col].resample('M').mean()

        # For recession indicator, use max (if any day in month is recession, month is recession)
        elif col == 'recession':
            series = data_copy[col].resample('M').max()

        else:
            # Default to last value
            series = data_copy[col].resample('M').last()

        monthly_data[col] = series

    # Combine all series into a single DataFrame
    monthly_df = pd.DataFrame(monthly_data)

    # Forward fill missing values for economic indicators (carry forward last known value)
    for col in monthly_df.columns:
        if col != 'recession':  # Don't fill recession indicator
            monthly_df[col] = monthly_df[col].fillna(method='ffill')

    # Ensure recession column exists and is filled with 0 for NaN
    if 'recession' in monthly_df.columns:
        monthly_df['recession'] = monthly_df['recession'].fillna(0).astype(int)

    logger.debug(f"Preprocessed data from {data.shape} to {monthly_df.shape}")
    return monthly_df


def plot_indicator_with_recessions(data, indicator_name, title=None, ylabel=None,
                                  recession_col='recession', figsize=(14, 8),
                                  save_path=None, preprocess=True):
    """
    Plot an economic indicator with recession periods shaded.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing the indicator and recession column
    indicator_name : str
        Name of the indicator column to plot
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    recession_col : str
        Name of the recession indicator column
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    preprocess : bool, optional
        Whether to preprocess the data for consistent plotting (default: True)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Preprocess data if requested
    if preprocess:
        data = _preprocess_data_for_plotting(data)

    # Check if indicator exists in data
    if indicator_name not in data.columns:
        logger.error(f"Indicator '{indicator_name}' not found in data columns: {list(data.columns)}")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the indicator
    ax.plot(data.index, data[indicator_name], linewidth=2)

    # Shade recession periods if recession column exists
    if recession_col in data.columns:
        recession_periods = []
        recession_start = None

        for date, value in data[recession_col].items():
            if value == 1 and recession_start is None:
                recession_start = date
            elif value == 0 and recession_start is not None:
                recession_periods.append((recession_start, date))
                recession_start = None

        # Add the last recession period if it extends to the end of the data
        if recession_start is not None:
            recession_periods.append((recession_start, data.index[-1]))

        for start, end in recession_periods:
            ax.axvspan(start, end, alpha=0.2, color='gray')

    # Set title and labels
    ax.set_title(title or f"{indicator_name} Over Time", fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel or indicator_name, fontsize=12)
    ax.grid(True, alpha=0.3)

    # Format x-axis to show years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")

    return fig


def plot_correlation_matrix(data, figsize=(14, 12), annot=True, save_path=None, preprocess=True):
    """
    Plot correlation matrix of the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to calculate correlations for
    figsize : tuple
        Figure size (width, height)
    annot : bool
        Whether to annotate the heatmap with correlation values
    save_path : str, optional
        Path to save the plot
    preprocess : bool, optional
        Whether to preprocess the data for consistent plotting (default: True)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Preprocess data if requested
    if preprocess:
        data = _preprocess_data_for_plotting(data)

    # Calculate correlations
    correlation_matrix = data.corr()

    # Plot correlation heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Economic Indicators', fontsize=16)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved correlation matrix plot to {save_path}")

    return fig


def plot_recession_correlations(data, recession_col='recession', top_n=15, figsize=(12, 8), save_path=None, preprocess=True):
    """
    Plot correlations of features with the recession indicator.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing features and recession indicator
    recession_col : str
        Name of the recession indicator column
    top_n : int
        Number of top correlated features to show
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    preprocess : bool, optional
        Whether to preprocess the data for consistent plotting (default: True)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Preprocess data if requested
    if preprocess:
        data = _preprocess_data_for_plotting(data)

    if recession_col not in data.columns:
        logger.error(f"Recession column '{recession_col}' not found in the data")
        return None

    # Calculate correlations with recession
    recession_corr = data.corr()[recession_col].sort_values(ascending=False)

    # Remove the recession column itself from correlations
    recession_corr = recession_corr.drop(recession_col)

    # Remove any NaN correlations
    recession_corr = recession_corr.dropna()

    # If we have fewer features than requested, adjust top_n
    available_features = len(recession_corr)
    if available_features == 0:
        logger.error("No valid correlations found with recession indicator")
        return None

    # Adjust top_n if we don't have enough features
    actual_top_n = min(top_n, available_features // 2)

    # Select top positive and negative correlations without overlap
    top_positive = recession_corr.head(actual_top_n)
    top_negative = recession_corr.tail(actual_top_n)

    # Remove any overlap between positive and negative (shouldn't happen but be safe)
    top_negative = top_negative[~top_negative.index.isin(top_positive.index)]

    # Combine correlations, ensuring no duplicates
    combined_corr = pd.concat([top_positive, top_negative])

    # Sort by absolute correlation value for better visualization
    combined_corr = combined_corr.reindex(combined_corr.abs().sort_values(ascending=True).index)

    # Plot correlations
    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(x=combined_corr.values, y=combined_corr.index)

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Color positive and negative correlations differently
    # Use enumerate to ensure we don't go out of bounds
    for i, (_, corr) in enumerate(combined_corr.items()):
        if i < len(ax.patches):  # Safety check
            if corr > 0:
                ax.patches[i].set_facecolor('red')
            else:
                ax.patches[i].set_facecolor('blue')

    plt.title(f'Top Correlations with Recession', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Economic Indicators', fontsize=12)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved recession correlations plot to {save_path}")

    return fig


def plot_feature_importance(feature_importance, top_n=15, figsize=(12, 8), save_path=None):
    """
    Plot feature importances from a model.

    Parameters
    ----------
    feature_importance : pandas.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Select top features
    top_features = feature_importance.head(top_n)

    # Plot feature importances
    fig = plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importances', fontsize=16)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved feature importance plot to {save_path}")

    return fig


def plot_mda_projection(lda_results, figsize=(10, 6), save_path=None):
    """
    Plot MDA projection of the data.

    Parameters
    ----------
    lda_results : dict
        Dictionary with MDA results from apply_mda function
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Extract data from results
    lda = lda_results['model']
    X_train = lda_results['X_train']
    X_test = lda_results['X_test']
    y_train = lda_results['y_train']
    y_test = lda_results['y_test']

    # Transform the data
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    # Plot the transformed data
    fig = plt.figure(figsize=figsize)

    # Plot training data
    for label, color in zip([0, 1], ['blue', 'red']):
        mask = y_train == label
        plt.scatter(
            X_train_lda[mask], np.zeros_like(X_train_lda[mask]) + 0.1,
            color=color, alpha=0.5, label=f'Train: {"Recession" if label == 1 else "Non-Recession"}'
        )

    # Plot testing data
    for label, color in zip([0, 1], ['cyan', 'magenta']):
        mask = y_test == label
        plt.scatter(
            X_test_lda[mask], np.zeros_like(X_test_lda[mask]) - 0.1,
            color=color, alpha=0.5, label=f'Test: {"Recession" if label == 1 else "Non-Recession"}'
        )

    plt.axvline(x=0, color='black', linestyle='--')
    plt.title('MDA Projection', fontsize=16)
    plt.xlabel('Discriminant Function', fontsize=12)
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved MDA projection plot to {save_path}")

    return fig


def plot_discriminant_time_series(discriminant_df, figsize=(14, 8), save_path=None):
    """
    Plot discriminant function values over time.

    Parameters
    ----------
    discriminant_df : pandas.DataFrame
        DataFrame with discriminant function values and optionally recession indicator
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)

    # Plot the discriminant function
    plt.plot(discriminant_df.index, discriminant_df['Discriminant'], linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--')

    # Shade recession periods if available
    if 'Recession' in discriminant_df.columns:
        recession_periods = []
        recession_start = None

        for date, value in discriminant_df['Recession'].items():
            if value == 1 and recession_start is None:
                recession_start = date
            elif value == 0 and recession_start is not None:
                recession_periods.append((recession_start, date))
                recession_start = None

        # Add the last recession period if it extends to the end of the data
        if recession_start is not None:
            recession_periods.append((recession_start, discriminant_df.index[-1]))

        for start, end in recession_periods:
            plt.axvspan(start, end, alpha=0.2, color='red')

    plt.title('Discriminant Function Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Discriminant Function', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Format x-axis to show years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved discriminant time series plot to {save_path}")

    return fig


def plot_sentiment_vs_indicator(data, sentiment_col='SENTIMENT', indicator_col=None,
                               recession_col='recession', figsize=(14, 8), save_path=None, preprocess=True):
    """
    Plot consumer sentiment against an economic indicator with recession periods highlighted.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing sentiment, indicator, and recession columns
    sentiment_col : str
        Name of the consumer sentiment column
    indicator_col : str, optional
        Name of the economic indicator column to compare with sentiment
    recession_col : str
        Name of the recession indicator column
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot
    preprocess : bool, optional
        Whether to preprocess the data for consistent plotting (default: True)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Preprocess data if requested
    if preprocess:
        data = _preprocess_data_for_plotting(data)

    if sentiment_col not in data.columns:
        logger.error(f"Sentiment column '{sentiment_col}' not found in the data")
        return None

    if indicator_col and indicator_col not in data.columns:
        logger.error(f"Indicator column '{indicator_col}' not found in the data")
        return None

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot sentiment on primary y-axis
    ax1.plot(data.index, data[sentiment_col], 'b-', linewidth=2, label=sentiment_col)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel(sentiment_col, color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')

    # If indicator column is provided, plot on secondary y-axis
    if indicator_col:
        ax2 = ax1.twinx()
        ax2.plot(data.index, data[indicator_col], 'g-', linewidth=2, label=indicator_col)
        ax2.set_ylabel(indicator_col, color='g', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='g')

    # Shade recession periods if recession column exists
    if recession_col in data.columns:
        recession_periods = []
        recession_start = None

        for date, value in data[recession_col].items():
            if value == 1 and recession_start is None:
                recession_start = date
            elif value == 0 and recession_start is not None:
                recession_periods.append((recession_start, date))
                recession_start = None

        # Add the last recession period if it extends to the end of the data
        if recession_start is not None:
            recession_periods.append((recession_start, data.index[-1]))

        for start, end in recession_periods:
            ax1.axvspan(start, end, alpha=0.2, color='gray')

    # Set title
    if indicator_col:
        plt.title(f'{sentiment_col} vs {indicator_col} Over Time', fontsize=16)
    else:
        plt.title(f'{sentiment_col} Over Time', fontsize=16)

    # Format x-axis to show years
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    plt.xticks(rotation=45)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if indicator_col:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax1.legend(loc='best')

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved sentiment vs indicator plot to {save_path}")

    return fig


def plot_sentiment_correlation_matrix(data, sentiment_cols=None, top_n=10, figsize=(12, 8), save_path=None):
    """
    Plot correlation matrix between consumer sentiment and top correlated economic indicators.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset containing sentiment and economic indicators
    sentiment_cols : list, optional
        List of sentiment column names to include, if None will look for 'SENTIMENT'
    top_n : int
        Number of top correlated indicators to show
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # If sentiment_cols not provided, use default
    if sentiment_cols is None:
        sentiment_cols = ['SENTIMENT']

    # Check if sentiment columns exist in the data
    missing_cols = [col for col in sentiment_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Sentiment columns {missing_cols} not found in the data")
        return None

    # Calculate correlations
    correlation_matrix = data.corr()

    # Create a figure
    fig = plt.figure(figsize=figsize)

    # For each sentiment column, find top correlated indicators
    for i, sentiment_col in enumerate(sentiment_cols):
        # Get correlations with the sentiment column
        sentiment_corr = correlation_matrix[sentiment_col].drop(sentiment_cols)

        # Get top positive and negative correlations
        top_pos = sentiment_corr.nlargest(min(top_n, len(sentiment_corr)))
        top_neg = sentiment_corr.nsmallest(min(top_n, len(sentiment_corr)))

        # Combine top correlations
        top_corr = pd.concat([top_pos, top_neg])

        # Create subplot
        plt.subplot(len(sentiment_cols), 1, i+1)
        ax = sns.barplot(x=top_corr.values, y=top_corr.index)

        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Color positive and negative correlations differently
        for j, (_, corr) in enumerate(top_corr.items()):
            if j < len(ax.patches):  # Make sure we don't go out of bounds
                if corr > 0:
                    ax.patches[j].set_facecolor('green')
                else:
                    ax.patches[j].set_facecolor('red')

        plt.title(f'Top Correlations with {sentiment_col}', fontsize=14)
        plt.xlabel('Correlation Coefficient', fontsize=12)

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Saved sentiment correlation matrix plot to {save_path}")

    return fig
