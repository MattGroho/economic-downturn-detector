# Data Dictionary

This document provides descriptions of the economic indicators and datasets used in the Economic Downturn Detector project.

## Primary Dataset: Federal Reserve Economic Data (FRED)

| Variable | FRED Series ID | Description | Frequency | Units |
|----------|---------------|-------------|-----------|-------|
| GDP | GDPC1 | Real Gross Domestic Product | Quarterly | Billions of Chained 2012 Dollars |
| UNEMPLOYMENT | UNRATE | Unemployment Rate | Monthly | Percent |
| CPI | CPIAUCSL | Consumer Price Index for All Urban Consumers | Monthly | Index 1982-1984=100 |
| FED_FUNDS | FEDFUNDS | Federal Funds Effective Rate | Monthly | Percent |
| YIELD_CURVE | T10Y2Y | 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity | Daily | Percentage Points |
| INITIAL_CLAIMS | ICSA | Initial Claims | Weekly | Number |
| INDUSTRIAL_PROD | INDPRO | Industrial Production Index | Monthly | Index 2017=100 |
| RETAIL_SALES | RSAFS | Advance Retail Sales: Retail and Food Services | Monthly | Millions of Dollars |
| HOUSING_STARTS | HOUST | Housing Starts: Total: New Privately Owned Housing Units Started | Monthly | Thousands of Units |
| CONSUMER_SENTIMENT | UMCSENT | University of Michigan: Consumer Sentiment | Monthly | Index 1966:Q1=100 |

## Secondary Dataset: National Bureau of Economic Research (NBER)

| Variable | Description | Frequency | Units |
|----------|-------------|-----------|-------|
| peak | Start date of recession | Date | YYYY-MM-DD |
| trough | End date of recession | Date | YYYY-MM-DD |
| recession | Binary indicator of recession (1 = recession, 0 = no recession) | Monthly | Binary |

## Secondary Dataset: University of Michigan Consumer Sentiment

| Variable | FRED Series ID | Description | Frequency | Units |
|----------|---------------|-------------|-----------|-------|
| SENTIMENT | UMCSENT | University of Michigan: Consumer Sentiment | Monthly | Index 1966:Q1=100 |
| INFLATION_EXPECTATION | MICH | University of Michigan: Inflation Expectation | Monthly | Percent |

## Additional Sentiment Data: Conference Board Consumer Confidence

| Variable | FRED Series ID | Description | Frequency | Units |
|----------|---------------|-------------|-----------|-------|
| CONF_BOARD | CSCICP03USM665S | Conference Board Consumer Confidence Index | Monthly | Index 1985=100 |

## Business Sentiment Indicators

| Variable | FRED Series ID | Description | Frequency | Units |
|----------|---------------|-------------|-----------|-------|
| ISM_PMI | MANEMP | ISM Manufacturing PMI | Monthly | Index |
| ISM_NONMFG | NMFBAI | ISM Non-Manufacturing Index | Monthly | Index |
| BUS_OPTIMISM | NFCIBUSOPX | NFIB Small Business Optimism Index | Monthly | Index |
| CEO_CONFIDENCE | CEOCONF | CEO Confidence Index | Quarterly | Index |
| PHILLY_FED | USPHCI | Philadelphia Fed Business Outlook Survey | Monthly | Index |

## Derived Variables

The following variables are derived from the primary economic indicators:

### Lag Variables

Lag variables capture the delayed effects of economic indicators on recession onset.

| Variable Pattern | Description | Example |
|------------------|-------------|---------|
| {indicator}_lag{n} | Value of the indicator n months ago | GDP_lag3 = GDP value 3 months ago |

### Rate of Change Variables

Rate of change variables capture the trends in economic indicators.

| Variable Pattern | Description | Example |
|------------------|-------------|---------|
| {indicator}_pct_change_1m | 1-month percentage change in the indicator | GDP_pct_change_1m = (GDP - GDP_lag1) / GDP_lag1 |
| {indicator}_pct_change_3m | 3-month percentage change in the indicator | GDP_pct_change_3m = (GDP - GDP_lag3) / GDP_lag3 |
| {indicator}_pct_change_12m | 12-month percentage change in the indicator | GDP_pct_change_12m = (GDP - GDP_lag12) / GDP_lag12 |

## Processed Datasets

The following processed datasets are created during the analysis:

| Dataset | Description | Location |
|---------|-------------|----------|
| merged_data.csv | Combined dataset with all economic indicators | data/processed/merged_data.csv |
| data_with_features.csv | Dataset with lag variables and rate of change features | data/processed/data_with_features.csv |
| data_normalized.csv | Normalized dataset with standardized features | data/processed/data_normalized.csv |
| data_pca.csv | Dataset with principal components | data/processed/data_pca.csv |

## Notes on Data Frequency

- Economic indicators have different natural frequencies (daily, weekly, monthly, quarterly)
- All data is standardized to monthly frequency for analysis
- For indicators with higher frequency (daily, weekly), the last value of the month is used
- For indicators with lower frequency (quarterly), the value is repeated for each month in the quarter

## Data Coverage and Cutoff

**Data Period**: January 1970 to May 2024
**Data Cutoff Date**: May 2024

All data collection is limited to May 2024 to ensure consistency across all indicators and provide a clear temporal boundary for the analysis. This timeframe encompasses 8 recession periods and multiple economic cycles, providing sufficient historical data for robust model training.

## Data Sources

- Federal Reserve Economic Data (FRED): https://fred.stlouisfed.org/
- National Bureau of Economic Research (NBER): https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
- University of Michigan Surveys of Consumers: https://data.sca.isr.umich.edu/
- Conference Board: https://www.conference-board.org/data/consumerconfidence.cfm
