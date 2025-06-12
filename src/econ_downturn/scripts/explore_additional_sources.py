#!/usr/bin/env python3
"""
Explore Additional Data Sources for Economic Downturn Detection

This script explores additional data sources that can enhance the economic downturn detector:
1. Conference Board Consumer Confidence Index
2. Business sentiment indicators (e.g., ISM Manufacturing PMI)
3. Social media sentiment analysis

The script demonstrates how to fetch, process, and integrate these data sources
with the existing economic indicators.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fredapi import Fred
from dotenv import load_dotenv

# Add the parent directory to the path to import the econ_downturn package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the econ_downturn package
from econ_downturn import (
    get_all_data, setup_logger, load_environment
)

# Set up logging
logger = setup_logger('explore_additional_sources')

# Load environment variables
load_dotenv()

# Get FRED API key
fred_api_key = os.getenv('FRED_API_KEY')
if not fred_api_key:
    logger.error("FRED API key not found. Please set the FRED_API_KEY environment variable.")
    sys.exit(1)

# Initialize FRED API client
fred = Fred(api_key=fred_api_key)

def fetch_conference_board_data():
    """
    Fetch Conference Board Consumer Confidence Index data from FRED.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with Conference Board Consumer Confidence Index
    """
    logger.info("Fetching Conference Board Consumer Confidence Index data...")
    
    # FRED series ID for Conference Board Consumer Confidence Index
    series_id = 'CSCICP03USM665S'
    
    try:
        # Fetch data from FRED
        data = fred.get_series(series_id)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['CONF_BOARD'])
        df.index.name = 'date'
        
        logger.info(f"Fetched Conference Board data with shape: {df.shape}")
        
        # Save to CSV
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/conf_board')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'consumer_confidence.csv')
        df.to_csv(output_path)
        logger.info(f"Saved Conference Board data to {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching Conference Board data: {e}")
        return pd.DataFrame()

def fetch_business_sentiment_data():
    """
    Fetch business sentiment indicators from FRED.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with business sentiment indicators
    """
    logger.info("Fetching business sentiment indicators...")
    
    # FRED series IDs for business sentiment indicators
    series_ids = {
        'ISM_PMI': 'MANEMP',           # ISM Manufacturing PMI
        'ISM_NONMFG': 'NMFBAI',        # ISM Non-Manufacturing Index
        'BUS_OPTIMISM': 'NFCIBUSOPX',  # NFIB Small Business Optimism Index
        'CEO_CONFIDENCE': 'CEOCONF',   # CEO Confidence Index
        'PHILLY_FED': 'USPHCI'         # Philadelphia Fed Business Outlook Survey
    }
    
    dfs = []
    
    try:
        for name, series_id in series_ids.items():
            # Fetch data from FRED
            data = fred.get_series(series_id)
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[name])
            df.index.name = 'date'
            
            dfs.append(df)
            
            logger.info(f"Fetched {name} data with shape: {df.shape}")
        
        # Merge all DataFrames
        if dfs:
            merged_df = pd.concat(dfs, axis=1)
            merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Merged business sentiment data with shape: {merged_df.shape}")
            
            # Save to CSV
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/business')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'business_sentiment.csv')
            merged_df.to_csv(output_path)
            logger.info(f"Saved business sentiment data to {output_path}")
            
            return merged_df
        else:
            logger.warning("No business sentiment data fetched")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error fetching business sentiment data: {e}")
        return pd.DataFrame()

def simulate_social_media_sentiment():
    """
    Simulate social media sentiment data for demonstration purposes.
    In a real implementation, this would connect to social media APIs or sentiment analysis services.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with simulated social media sentiment
    """
    logger.info("Simulating social media sentiment data...")
    
    # Create date range for the past 5 years
    end_date = datetime.now()
    start_date = datetime(end_date.year - 5, end_date.month, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Create DataFrame with random sentiment values
    np.random.seed(42)  # For reproducibility
    
    # Base sentiment follows a cyclical pattern
    t = np.arange(len(date_range))
    base_sentiment = 50 + 10 * np.sin(t / 12 * 2 * np.pi)
    
    # Add noise
    noise = np.random.normal(0, 5, len(date_range))
    
    # Add trend
    trend = -0.05 * t
    
    # Combine components
    sentiment = base_sentiment + noise + trend
    
    # Create DataFrame
    df = pd.DataFrame({
        'SOCIAL_SENTIMENT': sentiment,
        'SOCIAL_VOLUME': np.random.randint(1000, 10000, len(date_range))
    }, index=date_range)
    
    logger.info(f"Created simulated social media sentiment data with shape: {df.shape}")
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/social')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'social_sentiment.csv')
    df.to_csv(output_path)
    logger.info(f"Saved simulated social media sentiment data to {output_path}")
    
    return df

def integrate_additional_sources():
    """
    Integrate additional data sources with the existing economic indicators.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with all integrated data
    """
    logger.info("Integrating additional data sources...")
    
    # Load existing data
    existing_data = get_all_data()
    
    # Fetch additional data sources
    conf_board_data = fetch_conference_board_data()
    business_data = fetch_business_sentiment_data()
    social_data = simulate_social_media_sentiment()
    
    # Merge all data sources
    dfs = [existing_data]
    
    if not conf_board_data.empty:
        dfs.append(conf_board_data)
    
    if not business_data.empty:
        dfs.append(business_data)
    
    if not social_data.empty:
        dfs.append(social_data)
    
    # Merge all DataFrames
    integrated_data = pd.concat(dfs, axis=1)
    
    # Handle missing values
    integrated_data = integrated_data.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Integrated data with shape: {integrated_data.shape}")
    
    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'integrated_data.csv')
    integrated_data.to_csv(output_path)
    logger.info(f"Saved integrated data to {output_path}")
    
    return integrated_data

def visualize_additional_sources(data):
    """
    Create visualizations for the additional data sources.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Integrated dataset
    """
    logger.info("Creating visualizations for additional data sources...")
    
    # Create output directory for visualizations
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs/images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Conference Board Consumer Confidence Index
    if 'CONF_BOARD' in data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['CONF_BOARD'], 'b-', linewidth=2)
        plt.title('Conference Board Consumer Confidence Index', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Index', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'conf_board_index.png')
        plt.savefig(output_path)
        logger.info(f"Saved Conference Board visualization to {output_path}")
    
    # Plot business sentiment indicators
    business_cols = ['ISM_PMI', 'ISM_NONMFG', 'BUS_OPTIMISM', 'CEO_CONFIDENCE', 'PHILLY_FED']
    available_cols = [col for col in business_cols if col in data.columns]
    
    if available_cols:
        plt.figure(figsize=(12, 6))
        for col in available_cols:
            plt.plot(data.index, data[col], linewidth=2, label=col)
        
        plt.title('Business Sentiment Indicators', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Index', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'business_sentiment.png')
        plt.savefig(output_path)
        logger.info(f"Saved business sentiment visualization to {output_path}")
    
    # Plot social media sentiment
    if 'SOCIAL_SENTIMENT' in data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['SOCIAL_SENTIMENT'], 'g-', linewidth=2)
        plt.title('Social Media Sentiment', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sentiment Index', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'social_sentiment.png')
        plt.savefig(output_path)
        logger.info(f"Saved social media sentiment visualization to {output_path}")
    
    # Plot correlation matrix
    sentiment_cols = ['SENTIMENT', 'CONF_BOARD', 'SOCIAL_SENTIMENT']
    available_sentiment = [col for col in sentiment_cols if col in data.columns]
    
    if available_sentiment:
        plt.figure(figsize=(10, 8))
        
        # Select relevant columns for correlation
        cols_to_correlate = available_sentiment + ['GDP', 'UNEMPLOYMENT', 'recession']
        cols_to_correlate = [col for col in cols_to_correlate if col in data.columns]
        
        # Calculate correlation matrix
        corr_matrix = data[cols_to_correlate].corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix of Sentiment Indicators', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'sentiment_correlation_matrix.png')
        plt.savefig(output_path)
        logger.info(f"Saved sentiment correlation matrix to {output_path}")

def main():
    """Main function to explore additional data sources."""
    # Load environment variables
    load_environment()
    
    # Integrate additional data sources
    integrated_data = integrate_additional_sources()
    
    # Create visualizations
    visualize_additional_sources(integrated_data)
    
    logger.info("Exploration of additional data sources completed successfully")

if __name__ == "__main__":
    main()
