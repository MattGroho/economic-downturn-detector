# Project Proposal: Predicting Recession: A Multiple Discriminant Analysis of Economic Indicators

## 1. Team Members

- Avery Cloutier
- Matthew Grohotolski
- Sruthi Rayasam

## 2. Project Summary

### What is your proposed project and why are you proposing it?

Our project aims to develop a predictive model to identify potential recessions by analyzing key economic indicators using Multiple Discriminant Analysis (MDA). Recognizing the profound impact recessions have on societies and economies, early detection is crucial for policymakers, businesses, and individuals to make informed decisions.

### What are the question(s) you want to answer, or goal you want to achieve?

We propose to investigate the following questions:

- Which economic indicators are most significant in predicting recessions?
- How effectively can MDA classify periods as recessionary or non-recessionary based on these indicators?

By integrating diverse datasets and employing MDA, we seek to uncover patterns that precede recessions, offering a tool for early warning and decision-making support.

## 3. Datasets

### 3.1 Primary Dataset Description

- **Short description**: The Federal Reserve Economic Data (FRED) repository offers a comprehensive collection of U.S. economic indicators, including real GDP, unemployment rates, Consumer Price Index (CPI), federal funds rate, and consumer sentiment indices.
- **Estimated size**: ~50,000 records
- **Location**: [FRED - Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- **Format**: CSV (exportable)
- **Access method**: API and direct CSV download (via Python fredapi or manual download)

### 3.2 Secondary Dataset(s) Description

#### National Bureau of Economic Research (NBER) Business Cycle Data

- **Short Description**: The National Bureau of Economic Research (NBER) provides official dates for U.S. business cycle peaks and troughs, identifying periods of recession and expansion.
- **Estimated Size**: Approximately 100 records, covering monthly data since 1854.
- **Location**: [NBER Business Cycle Dating](https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions)
- **Format**: CSV
- **Access Method**: Direct download

#### University of Michigan Consumer Sentiment Index

- **Short Description**: The University of Michigan's Surveys of Consumers offer insights into consumer sentiment, capturing expectations about personal finances and the economy.
- **Estimated Size**: Around 1,000 records, with monthly data available since the 1950s.
- **Location**: [University of Michigan: Consumer Sentiment Index](https://data.sca.isr.umich.edu/)
- **Format**: CSV
- **Access Method**: Direct download or API access via FRED

#### Unemployment Rate Data

- **Short Description**: % of labor force unemployed and actively seeking work.
- **Estimated Size**: ~900 records (monthly from 1948–present)
- **Location**: [FRED Unemployment Rate](https://fred.stlouisfed.org/series/UNRATE)
- **Format**: CSV
- **Access Method**: Direct download from FRED

### 3.3 Affirm: Datasets are Public

YES

## 4. Cleaning and Manipulation

To prepare the datasets for analysis, we will undertake the following steps:

- **Time Alignment**: Standardize the time frames across datasets, converting all data to a consistent monthly frequency to facilitate accurate merging and comparison.
- **Handling Missing Data**: Employ interpolation methods or remove records with missing values, depending on the extent and significance of the gaps.
- **Feature Engineering**: Create lag variables to capture the delayed effects of economic indicators on recession onset.
- **Normalization**: Standardize variables to ensure comparability and to meet the assumptions of MDA.
- **Data Integration**: Merge datasets on the date field, ensuring that each record represents a comprehensive snapshot of the economic indicators for that month.

Anticipated challenges include resolving discrepancies in data frequency and addressing any structural breaks in the time series data due to changes in data collection methodologies over time.

## 5. Analysis

Our analytical approach will encompass:

- **Feature Importances**: Display the contribution of each economic indicator to the discriminant functions and dropout irrelevant features not contributing to final MDA results.
- **Principal Component Analysis (PCA)**: Apply PCA to reduce dimensionality and multicollinearity among predictors, enhancing the robustness of the MDA model.
- **Multiple Discriminant Analysis (MDA)**: Utilize MDA to identify linear combinations of economic indicators that best differentiate between recessionary and non-recessionary periods.
- **Model Evaluation**: Assess model performance using metrics such as classification accuracy, sensitivity, specificity, and confusion matrices.
- **Cross-Validation**: Implement k-fold cross-validation to ensure the model's robustness and generalizability.

Through this analysis, we aim to uncover the most influential economic indicators in predicting recessions and to understand their combined effects.

## 6. Visualizations

We plan to create the following visualizations:

- **Correlation Heatmaps**: Illustrate the relationships among economic indicators to identify potential multicollinearity issues.
- **Time Series Plots**: Display trends of key economic indicators (e.g., GDP, unemployment rate, initial claims) over time, highlighting recession periods for contextual understanding.
- **Discriminant Function Plots**: Visualize the separation achieved by the MDA model between recessionary and non-recessionary periods.

These visualizations will aid in interpreting the model's findings and in communicating insights effectively.

## 7. Ethical Considerations

While the data used is publicly available and aggregated at the national level, we acknowledge the importance of ethical considerations in our analysis:

- **Data Interpretation**: We will exercise caution in interpreting the results, avoiding overreliance on the model's predictions for policy decisions without considering broader economic contexts.
- **Transparency**: All methodologies and assumptions will be documented clearly to ensure transparency and reproducibility.
- **Limitations**: We will explicitly state the limitations of our model, including potential biases and the historical nature of the data, to prevent misapplication of the findings.

## 8. Contributions

- **Avery Cloutier**: Responsible for data cleaning, feature engineering, and visualization creation.
- **Matthew Grohotolski**: Lead on data acquisition and preprocessing, development of the analytical framework, and coordination of team activities.
- **Sruthi Rayasam**: Focused on model implementation, evaluation, and documentation of findings.

All team members will collaborate on the final report and presentation, ensuring a cohesive and comprehensive project outcome.
