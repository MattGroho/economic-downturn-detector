# Economic Downturn Detector Notebooks

This directory contains Jupyter notebooks for analyzing economic indicators and developing a recession prediction model using Multiple Discriminant Analysis (MDA).

## Notebook Structure

The notebooks follow a logical progression:

0. **00_data_collection.ipynb**: Comprehensive data collection from all sources including FRED, NBER, University of Michigan, Conference Board, and business sentiment indicators
1. **01_data_exploration.ipynb**: Explores the economic indicators data and their relationships with recession periods
2. **02_feature_engineering.ipynb**: Creates lag variables and other features to capture leading indicators
3. **03_multiple_discriminant_analysis.ipynb**: Applies MDA to identify the most significant predictors of recessions
4. **04_combined_data_analysis.ipynb**: Analyzes the combined dataset with consumer sentiment data
5. **05_feature_engineering_optimization.ipynb**: Optimizes feature engineering with advanced techniques

## Clean Import Setup

**For all notebooks, start with this simple setup:**

```python
from notebook_utils import init_notebook
init_notebook()

# Now you can import econ_downturn functions directly
from econ_downturn import get_all_data, plot_indicator_with_recessions
```

This replaces manual sys.path setup and provides clean, consistent access to all project functionality.

## DRY Structure

The notebooks follow the DRY (Don't Repeat Yourself) principle:

1. **notebook_utils.py**: Provides `init_notebook()` for setup and common utility functions
2. **feature_engineering_utils.py**: Advanced feature engineering functions
3. **econ_downturn package**: Core functionality organized in modules

## How to Use

1. **Start with data collection**: Run `00_data_collection.ipynb` first to fetch all required data sources
2. **Follow the sequence**: Run notebooks in order (00 → 05) as each builds on the previous ones
3. **Use init_notebook()**: Start each notebook with `from notebook_utils import init_notebook; init_notebook()`
4. **Import directly**: After setup, import econ_downturn functions directly

## No More Manual sys.path Setup!

❌ **Old way (don't do this):**
```python
import sys
import os
sys.path.append('../src')
```

✅ **New way (clean and simple):**
```python
from notebook_utils import init_notebook
init_notebook()
```

## Data Collection (Notebook 00)

The data collection notebook handles all data gathering tasks:

- **FRED Economic Indicators**: GDP, unemployment, CPI, federal funds rate, yield curve, and other key economic metrics
- **NBER Recession Data**: Official recession dates and indicators from the National Bureau of Economic Research
- **University of Michigan Sentiment**: Consumer sentiment, expectations, and inflation expectations
- **Conference Board Data**: Consumer confidence index
- **Business Sentiment**: ISM PMI, NFIB optimism, CEO confidence, Philadelphia Fed survey
- **Data Integration**: Combines all sources into a comprehensive dataset ready for analysis

**Data Cutoff**: May 2024 - All data collection is limited to this date to ensure consistency across all indicators.

## Utility Functions

The `notebook_utils.py` module provides several utility functions:

- `setup_notebook()`: Sets up the notebook environment with common configurations
- `load_data(use_cached=True)`: Loads all data sources and merges them
- `display_data_info(data)`: Displays information about the dataset
- `save_figure(fig, filename, output_dir=None)`: Saves a figure to the output directory

## Example Usage

```python
# Import notebook utilities
from notebook_utils import (
    setup_notebook, load_data, display_data_info, save_figure,
    plot_indicator_with_recessions, plot_correlation_matrix
)

# Set up the notebook environment
setup_notebook()

# Load data
merged_data = load_data()

# Display information about the dataset
display_data_info(merged_data)

# Plot an indicator with recession periods
fig = plot_indicator_with_recessions(
    merged_data, 
    'UNEMPLOYMENT',
    title='Unemployment Rate with Recession Periods'
)

# Save the figure
save_figure(fig, "unemployment_over_time.png")
```

## Advanced Feature Engineering

For advanced feature engineering, use the functions from the `econ_downturn.features.advanced_engineering` module:

- `engineer_features_with_custom_lags()`: Creates lag variables with custom lag periods
- `create_interaction_terms()`: Creates interaction terms between sentiment and economic indicators
- `apply_sentiment_transformations()`: Applies various transformations to sentiment data
- `select_features()`: Selects the most predictive features

These functions are imported through the `notebook_utils.py` module for convenience.
