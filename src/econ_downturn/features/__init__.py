"""
Features module for feature engineering and data preparation.
"""

from .feature_engineering import (
    handle_missing_values,
    resample_data,
    create_lag_variables,
    calculate_rate_of_change,
    drop_missing_values,
    engineer_features
)

from .normalization import (
    normalize_data,
    apply_pca,
    prepare_data_for_modeling
)