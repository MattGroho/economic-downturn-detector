"""
Utility functions for custom feature engineering.

This module now imports the advanced feature engineering functions from the
econ_downturn package to maintain DRY (Don't Repeat Yourself) principle.
"""

import os
import sys

# Add the parent directory to the path to import the econ_downturn package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from econ_downturn package
from econ_downturn.features.advanced_engineering import (
    engineer_features_with_custom_lags,
    create_interaction_terms,
    apply_sentiment_transformations,
    select_features
)

# These functions are now imported from the package:
# - engineer_features_with_custom_lags: Creates lag variables with custom lag periods
# - create_interaction_terms: Creates interaction terms between sentiment and economic indicators
# - apply_sentiment_transformations: Applies various transformations to sentiment data
# - select_features: Selects the most predictive features
