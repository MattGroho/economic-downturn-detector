#!/usr/bin/env python3
"""
Train and Save MDA Model for Economic Downturn Detection

This script trains a Multiple Discriminant Analysis (MDA) model on the combined dataset
(including consumer sentiment data) and saves the model and scaler for later use
in the real-time monitoring system.
"""

import os
import sys
import pandas as pd
import joblib
import logging
import argparse

# Add the parent directory to the path to import the econ_downturn package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the econ_downturn package
from econ_downturn import (
    get_all_data, engineer_features, normalize_data,
    apply_mda, create_discriminant_time_series,
    plot_feature_importance, plot_discriminant_time_series,
    setup_logger, load_environment
)

# Set up logging
logger = setup_logger('train_model')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and Save MDA Model')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the processed data (if None, will load and process data)')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save the model and scaler')
    parser.add_argument('--include-sentiment', action='store_true', default=True,
                        help='Include sentiment features in the model')
    parser.add_argument('--custom-features', action='store_true', default=False,
                        help='Use custom feature engineering (from notebook 05)')

    return parser.parse_args()

def load_and_prepare_data(data_path=None, include_sentiment=True, custom_features=False):
    """
    Load and prepare data for model training.

    Parameters
    ----------
    data_path : str, optional
        Path to the processed data
    include_sentiment : bool
        Whether to include sentiment features
    custom_features : bool
        Whether to use custom feature engineering

    Returns
    -------
    tuple
        X (features), y (target), scaler
    """
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        logger.info("Loading and processing data from original sources")
        # Load data from different sources
        merged_data = get_all_data()

        # Engineer features
        if custom_features:
            # Import custom feature engineering functions
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
            try:
                from feature_engineering_utils import engineer_features_with_custom_lags, create_interaction_terms, apply_sentiment_transformations

                # Apply custom feature engineering
                data = engineer_features_with_custom_lags(merged_data)
                data = create_interaction_terms(data)
                data = apply_sentiment_transformations(data)

                logger.info("Applied custom feature engineering")
            except ImportError:
                logger.warning("Custom feature engineering module not found, using standard feature engineering")
                data = engineer_features(merged_data)
        else:
            # Use standard feature engineering
            data = engineer_features(merged_data)

    # Remove sentiment features if not including them
    if not include_sentiment:
        sentiment_cols = [col for col in data.columns if 'SENTIMENT' in col]
        data = data.drop(columns=sentiment_cols)
        logger.info(f"Removed {len(sentiment_cols)} sentiment-related features")

    # Normalize the data
    data_normalized, scaler = normalize_data(data)

    # Separate features and target
    X = data_normalized.drop(columns=['recession'])
    y = data_normalized['recession']

    logger.info(f"Prepared data with {X.shape[1]} features and {len(y)} samples")

    return X, y, scaler

def train_and_save_model(X, y, scaler, model_dir):
    """
    Train MDA model and save model and scaler.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    model_dir : str
        Directory to save the model and scaler

    Returns
    -------
    dict
        MDA results
    """
    # Apply MDA
    mda_results = apply_mda(X, y)

    # Create output directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(model_dir, 'mda_model.joblib')
    joblib.dump(mda_results['model'], model_path)
    logger.info(f"Saved MDA model to {model_path}")

    # Save the scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Save feature importance
    if mda_results['feature_importance'] is not None:
        feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': mda_results['feature_importance']
        }).sort_values('Importance', ascending=False)
        feature_importance_df.to_csv(feature_importance_path, index=False)
        logger.info(f"Saved feature importance to {feature_importance_path}")

    # Save model performance metrics
    metrics_path = os.path.join(model_dir, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {mda_results['accuracy']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{mda_results['conf_matrix']}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{mda_results['class_report']}\n\n")
        f.write("Cross-Validation Scores:\n")
        f.write(f"{mda_results['cv_scores']}\n")
        f.write(f"Mean CV Score: {mda_results['cv_scores'].mean():.4f}\n")

    logger.info(f"Saved model performance metrics to {metrics_path}")

    # Create discriminant time series
    discriminant_df = create_discriminant_time_series(
        mda_results['model'], X, y
    )

    # Save discriminant time series
    discriminant_path = os.path.join(model_dir, 'discriminant_time_series.csv')
    discriminant_df.to_csv(discriminant_path)
    logger.info(f"Saved discriminant time series to {discriminant_path}")

    # Create and save visualizations
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs/images')
    os.makedirs(output_dir, exist_ok=True)

    # Plot discriminant time series - use the main discriminant_time_series.png file
    plot_discriminant_time_series(
        discriminant_df,
        save_path=os.path.join(output_dir, 'discriminant_time_series.png')
    )

    # Plot feature importance - use the main feature_importance.png file
    if mda_results['feature_importance'] is not None:
        # Create a proper DataFrame for feature importance
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': mda_results['feature_importance']
        }).sort_values('Importance', ascending=False)

        plot_feature_importance(
            feature_importance_df,
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )

    return mda_results

def main():
    """Main function to train and save the MDA model."""
    # Parse command line arguments
    args = parse_arguments()

    # Load environment variables
    load_environment()

    # Get full paths
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.model_dir)

    # Load and prepare data
    X, y, scaler = load_and_prepare_data(
        data_path=args.data_path,
        include_sentiment=args.include_sentiment,
        custom_features=args.custom_features
    )

    # Train and save model
    mda_results = train_and_save_model(X, y, scaler, model_dir)

    logger.info("Model training and saving completed successfully")
    logger.info(f"Model accuracy: {mda_results['accuracy']:.4f}")
    logger.info(f"Mean CV score: {mda_results['cv_scores'].mean():.4f}")

if __name__ == "__main__":
    main()
