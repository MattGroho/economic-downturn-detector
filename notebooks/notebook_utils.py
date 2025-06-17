"""
Utility functions for Jupyter notebooks.

This module provides a simple way to set up imports and common configurations
for the economic downturn detector notebooks.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings


def init_notebook():
    """Set up notebook environment and econ_downturn imports."""
    print("Initializing notebook environment...")

    # Step 1: Set up econ_downturn imports
    success = _setup_econ_downturn_imports()
    if not success:
        print("⚠ econ_downturn package setup failed - some functionality may be limited")

    # Step 2: Configure notebook environment
    _configure_notebook_environment()

    # Step 3: Load environment variables and show paths
    if success:
        _load_environment_and_paths()

    print("✓ Notebook initialization complete!")
    print("\nYou can now import econ_downturn functions directly:")
    print("  from econ_downturn import get_all_data, plot_indicator_with_recessions")

    return success


def _setup_econ_downturn_imports():
    """Add src directory to Python path so we can import econ_downturn."""
    try:
        # First, try to import the package directly
        import econ_downturn
        print("✓ econ_downturn package already available")
        return True
    except ImportError:
        pass

    # Find the src directory
    current_dir = os.path.abspath(os.getcwd())
    notebook_dir = os.path.dirname(os.path.abspath(__file__))

    # Try different possible locations for the src directory
    possible_src_paths = [
        os.path.join(notebook_dir, '..', 'src'),  # From notebooks directory
        os.path.join(current_dir, 'src'),         # From project root
        os.path.join(current_dir, '..', 'src'),   # From subdirectory
    ]

    src_path = None
    for path in possible_src_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, 'econ_downturn')):
            src_path = abs_path
            break

    if src_path is None:
        print("✗ Could not find econ_downturn package")
        return False

    # Add to sys.path if not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✓ Added {src_path} to Python path")

    # Verify the import works
    try:
        import econ_downturn
        print("✓ econ_downturn package imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import econ_downturn package: {e}")
        return False


def _configure_notebook_environment():
    """Set up plotting styles and pandas options."""
    try:
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('deep')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

        # Display all columns in pandas
        pd.set_option('display.max_columns', None)

        # Suppress warnings
        warnings.filterwarnings('ignore')

        print("✓ Notebook environment configured")

    except Exception as e:
        print(f"⚠ Could not configure environment: {e}")


def _load_environment_and_paths():
    """Load .env file and show data/output paths."""
    try:
        from econ_downturn import load_environment, get_data_paths, get_output_paths

        # Load environment variables
        load_environment()
        print("✓ Environment variables loaded")

        # Show available paths
        print("\nAvailable data paths:")
        for key, path in get_data_paths().items():
            print(f"  {key}: {path}")
        print("\nAvailable output paths:")
        for key, path in get_output_paths().items():
            print(f"  {key}: {path}")

    except Exception as e:
        print(f"⚠ Could not load environment/paths: {e}")


# Convenience functions for common notebook operations
def load_data(use_cached=True):
    """Load merged dataset. Uses cached version if available."""
    try:
        from econ_downturn import get_all_data, load_merged_data, get_data_paths

        # Get data paths
        data_paths = get_data_paths()

        # Check if merged data exists and use_cached is True
        merged_data_path = data_paths.get('merged_data')
        if use_cached and merged_data_path and os.path.exists(merged_data_path):
            print(f"Loading cached merged data from {merged_data_path}")
            merged_data = load_merged_data()
            print(f"Loaded merged data with shape: {merged_data.shape}")
            return merged_data

        # Otherwise, load and merge data from original sources
        print("Loading data from original sources...")
        merged_data = get_all_data()

        if merged_data is not None and not merged_data.empty:
            print(f"Loaded and merged data with shape: {merged_data.shape}")
            return merged_data
        else:
            print("Failed to load data.")
            return None

    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def display_data_info(data):
    """Show dataset summary, stats, and missing values."""
    if data is None or data.empty:
        print("No data to display.")
        return

    # Display basic information
    print("Dataset Information:")
    print(f"Time Range: {data.index.min()} to {data.index.max()}")
    print(f"Number of Observations: {len(data)}")
    print(f"Number of Features: {data.shape[1]}")

    # Display summary statistics
    print("\nSummary Statistics:")
    display(data.describe())

    # Check for missing values
    print("\nMissing Values:")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    display(missing_df.sort_values('Missing Values', ascending=False))


def save_figure(fig, filename, output_dir=None):
    """Save plot to docs/images (or specified directory)."""
    try:
        from econ_downturn import get_output_paths

        if output_dir is None:
            output_dir = get_output_paths()['images_dir']

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the figure
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    except Exception as e:
        print(f"Error saving figure: {e}")


# Legacy support functions for backward compatibility
def setup_notebook():
    """Legacy function - use init_notebook() instead."""
    print("⚠ setup_notebook() is deprecated - use init_notebook() instead")
    return init_notebook()


