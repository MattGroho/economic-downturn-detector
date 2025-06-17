"""
Economic Downturn Detector

A package for predicting economic recessions using Multiple Discriminant Analysis (MDA)
on key economic indicators.
"""

# Import submodules
from . import data
from . import features
from . import models
from . import visualization
from . import utils

# Import key functions for easier access
from .data.fred import get_fred_data
from .data.nber import get_nber_data
from .data.umich import get_umich_data
from .data.data_loader import (
    get_all_data, load_merged_data, load_umich_data,
    load_fred_data, load_nber_data
)

from .features.feature_engineering import engineer_features
from .features.normalization import normalize_data, apply_pca, prepare_data_for_modeling
from .features.advanced_engineering import (
    engineer_features_with_custom_lags,
    create_interaction_terms,
    apply_sentiment_transformations,
    select_features
)

from .models.mda import apply_mda, create_discriminant_time_series

from .visualization.plotting import (
    plot_indicator_with_recessions,
    plot_correlation_matrix,
    plot_recession_correlations,
    plot_feature_importance,
    plot_mda_projection,
    plot_discriminant_time_series,
    plot_sentiment_vs_indicator,
    plot_sentiment_correlation_matrix
)

from .utils.logger import setup_logger, get_default_logger
from .utils.config import load_environment, get_data_paths, get_output_paths

# Set up package-level logger
logger = get_default_logger(__name__)

# Version
__version__ = '0.1.0'