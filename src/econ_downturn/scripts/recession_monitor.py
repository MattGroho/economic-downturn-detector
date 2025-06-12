#!/usr/bin/env python3
"""
Real-time Economic Downturn Monitoring System

This script implements a monitoring system that tracks consumer sentiment alongside
other economic indicators and provides an early warning when indicators suggest
an increased recession risk.

The system:
1. Loads the latest economic data
2. Applies the trained MDA model to calculate recession probability
3. Generates a dashboard with current economic indicators
4. Sends alerts when recession risk exceeds a threshold
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime
import argparse
import warnings
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Add the parent directory to the path to import the econ_downturn package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the econ_downturn package
from econ_downturn import (
    get_all_data, engineer_features, create_discriminant_time_series,
    setup_logger, load_environment
)

# Set up logging
logger = setup_logger('recession_monitor')

# Suppress warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Economic Downturn Monitoring System')
    parser.add_argument('--model-path', type=str, default='models/mda_model.joblib',
                        help='Path to the trained MDA model')
    parser.add_argument('--scaler-path', type=str, default='models/scaler.joblib',
                        help='Path to the fitted scaler')
    parser.add_argument('--output-dir', type=str, default='dashboard',
                        help='Directory to save the dashboard')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Recession probability threshold for alerts')
    parser.add_argument('--lookback', type=int, default=60,
                        help='Number of months to look back for trends')
    parser.add_argument('--alert', action='store_true',
                        help='Enable alert system')
    
    return parser.parse_args()

def load_latest_data():
    """Load the latest economic data."""
    logger.info("Loading latest economic data...")

    # Load merged data from all sources
    merged_data = get_all_data()

    logger.info(f"Loaded data with shape: {merged_data.shape}")

    return merged_data

def prepare_data_for_prediction(data, scaler_path):
    """Prepare data for prediction using the trained model."""
    logger.info("Preparing data for prediction...")
    
    # Engineer features
    data_with_features = engineer_features(data)
    
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    # Apply the scaler to the features
    features = data_with_features.drop(columns=['recession'])
    features_scaled = pd.DataFrame(
        scaler.transform(features),
        columns=features.columns,
        index=features.index
    )
    
    # Add back the recession indicator
    features_scaled['recession'] = data_with_features['recession']
    
    logger.info(f"Prepared data with shape: {features_scaled.shape}")
    
    return features_scaled

def calculate_recession_probability(data, model_path):
    """Calculate recession probability using the trained model."""
    logger.info("Calculating recession probability...")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Separate features and target
    X = data.drop(columns=['recession'])
    y = data['recession']
    
    # Create discriminant time series
    discriminant_df = create_discriminant_time_series(model, X, y)
    
    # Calculate recession probability
    discriminant_df['Probability'] = 1 / (1 + np.exp(-discriminant_df['Discriminant']))
    
    logger.info(f"Calculated recession probability for {len(discriminant_df)} periods")
    
    return discriminant_df

def generate_dashboard(data, discriminant_df, output_dir, lookback=60):
    """Generate a dashboard with current economic indicators and recession probability."""
    logger.info("Generating dashboard...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the latest data point
    latest_date = discriminant_df.index[-1]
    latest_prob = discriminant_df['Probability'].iloc[-1]
    
    # Get data for the lookback period
    start_date = latest_date - pd.DateOffset(months=lookback)
    recent_data = data[data.index >= start_date]
    recent_discriminant = discriminant_df[discriminant_df.index >= start_date]
    
    # Create the dashboard figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Recession Probability
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(recent_discriminant.index, recent_discriminant['Probability'], 'b-', linewidth=2)
    ax1.axhline(y=0.5, color='r', linestyle='--')
    ax1.set_title('Recession Probability', fontsize=16)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add recession periods if available
    if 'recession' in data.columns:
        for i in range(len(recent_data)-1):
            if recent_data['recession'].iloc[i] == 1:
                ax1.axvspan(recent_data.index[i], recent_data.index[i+1], alpha=0.2, color='gray')
    
    # Plot 2: Consumer Sentiment
    ax2 = fig.add_subplot(gs[1, 0])
    if 'SENTIMENT' in recent_data.columns:
        ax2.plot(recent_data.index, recent_data['SENTIMENT'], 'g-', linewidth=2)
        ax2.set_title('Consumer Sentiment', fontsize=16)
        ax2.set_ylabel('Index', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Unemployment Rate
    ax3 = fig.add_subplot(gs[1, 1])
    if 'UNEMPLOYMENT' in recent_data.columns:
        ax3.plot(recent_data.index, recent_data['UNEMPLOYMENT'], 'r-', linewidth=2)
        ax3.set_title('Unemployment Rate', fontsize=16)
        ax3.set_ylabel('Percent', fontsize=12)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: GDP Growth
    ax4 = fig.add_subplot(gs[2, 0])
    if 'GDP' in recent_data.columns:
        ax4.plot(recent_data.index, recent_data['GDP'], 'b-', linewidth=2)
        ax4.set_title('GDP Growth', fontsize=16)
        ax4.set_ylabel('Percent Change', fontsize=12)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Yield Curve
    ax5 = fig.add_subplot(gs[2, 1])
    if 'YIELD_CURVE' in recent_data.columns:
        ax5.plot(recent_data.index, recent_data['YIELD_CURVE'], 'purple', linewidth=2)
        ax5.axhline(y=0, color='r', linestyle='--')
        ax5.set_title('Yield Curve (10Y-3M)', fontsize=16)
        ax5.set_ylabel('Percentage Points', fontsize=12)
        ax5.grid(True, alpha=0.3)
    
    # Format x-axis for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add dashboard title with latest date and probability
    fig.suptitle(f'Economic Downturn Monitor - {latest_date.strftime("%Y-%m-%d")}\n'
                 f'Current Recession Probability: {latest_prob:.2%}',
                 fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the dashboard
    dashboard_path = os.path.join(output_dir, f'dashboard_{latest_date.strftime("%Y%m%d")}.png')
    plt.savefig(dashboard_path, dpi=150)
    
    # Also save as latest.png for easy reference
    latest_path = os.path.join(output_dir, 'dashboard_latest.png')
    plt.savefig(latest_path, dpi=150)
    
    logger.info(f"Dashboard saved to {dashboard_path}")
    
    return dashboard_path

def check_alert_conditions(discriminant_df, threshold=0.7):
    """Check if alert conditions are met."""
    # Get the latest probability
    latest_prob = discriminant_df['Probability'].iloc[-1]
    
    # Check if probability exceeds threshold
    if latest_prob >= threshold:
        return True, latest_prob
    
    return False, latest_prob

def send_alert(probability, dashboard_path):
    """Send an alert when recession risk is high."""
    logger.warning(f"ALERT: Recession probability is {probability:.2%}, which exceeds the threshold!")
    logger.warning(f"Dashboard available at: {dashboard_path}")
    
    # In a real system, this would send an email, SMS, or other notification
    # For now, we just log the alert
    
    # Create an alert file
    alert_path = os.path.join(os.path.dirname(dashboard_path), 'RECESSION_ALERT.txt')
    with open(alert_path, 'w') as f:
        f.write(f"RECESSION ALERT - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Current recession probability: {probability:.2%}\n")
        f.write(f"Dashboard: {dashboard_path}\n")
    
    logger.warning(f"Alert saved to {alert_path}")

def main():
    """Main function to run the monitoring system."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load environment variables
    load_environment()
    
    # Get full paths
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.model_path)
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.scaler_path)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output_dir)
    
    # Load the latest data
    data = load_latest_data()
    
    # Prepare data for prediction
    prepared_data = prepare_data_for_prediction(data, scaler_path)
    
    # Calculate recession probability
    discriminant_df = calculate_recession_probability(prepared_data, model_path)
    
    # Generate dashboard
    dashboard_path = generate_dashboard(data, discriminant_df, output_dir, args.lookback)
    
    # Check alert conditions
    alert_triggered, probability = check_alert_conditions(discriminant_df, args.threshold)
    
    # Send alert if conditions are met and alerts are enabled
    if alert_triggered and args.alert:
        send_alert(probability, dashboard_path)
    
    logger.info("Monitoring completed successfully")
    logger.info(f"Latest recession probability: {probability:.2%}")

if __name__ == "__main__":
    main()
