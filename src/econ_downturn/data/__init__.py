"""
Data module for fetching and loading economic data.
"""

from .fred import fetch_fred_data, save_data, get_fred_data
from .nber import fetch_nber_data, create_recession_indicator, get_nber_data
from .umich import fetch_umich_data, get_umich_data
from .data_loader import load_fred_data, load_nber_data, load_umich_data, merge_datasets, load_merged_data, get_all_data