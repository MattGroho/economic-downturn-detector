"""
Models module for Multiple Discriminant Analysis (MDA) and other modeling techniques.
"""

from .mda import (
    load_data_for_mda,
    apply_mda,
    transform_data_with_mda,
    create_discriminant_time_series,
    predict_recession_probability
)