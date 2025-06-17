"""
Configuration utilities for the economic downturn detector.

This module provides functions to load and manage configuration settings.
"""

import os
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file."""
    # Load environment variables from .env file
    load_dotenv()

    # Get API keys
    fred_api_key = os.getenv('FRED_API_KEY')

    # Create config dictionary
    config = {
        'fred_api_key': fred_api_key,
        'data_dir': os.getenv('DATA_DIR', 'data'),
        'output_dir': os.getenv('OUTPUT_DIR', 'docs/images')
    }

    return config


def get_project_root():
    """Get the project root directory."""
    # Get the directory containing this file (src/econ_downturn/utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up three levels to get to project root: utils -> econ_downturn -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return project_root


def get_data_paths(base_dir='data'):
    """Get paths to data directories and files relative to project root."""
    # Get project root and make all paths relative to it
    project_root = get_project_root()
    base_path = os.path.join(project_root, base_dir)

    paths = {
        'fred_dir': os.path.join(base_path, 'fred'),
        'nber_dir': os.path.join(base_path, 'nber'),
        'processed_dir': os.path.join(base_path, 'processed'),
        'fred_all_indicators': os.path.join(base_path, 'fred', 'all_indicators.csv'),
        'nber_recession_indicator': os.path.join(base_path, 'nber', 'recession_indicator.csv'),
        'merged_data': os.path.join(base_path, 'processed', 'merged_data.csv'),
        'data_with_features': os.path.join(base_path, 'processed', 'data_with_features.csv'),
        'data_normalized': os.path.join(base_path, 'processed', 'data_normalized.csv'),
        'data_pca': os.path.join(base_path, 'processed', 'data_pca.csv')
    }

    return paths


def get_output_paths(base_dir='docs/images'):
    """Get paths to output directories and files relative to project root."""
    # Get project root and make all paths relative to it
    project_root = get_project_root()
    base_path = os.path.join(project_root, base_dir)

    paths = {
        'images_dir': base_path,
        'feature_importance': os.path.join(base_path, 'feature_importance.png'),
        'mda_projection': os.path.join(base_path, 'mda_projection.png'),
        'discriminant_time': os.path.join(base_path, 'discriminant_time.png'),
        'correlation_matrix': os.path.join(base_path, 'correlation_matrix.png'),
        'recession_correlations': os.path.join(base_path, 'recession_correlations.png'),
        'pca_explained_variance': os.path.join(base_path, 'pca_explained_variance.png')
    }

    return paths
