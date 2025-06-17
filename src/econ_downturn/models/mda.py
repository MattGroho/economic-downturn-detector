"""
Functions for Multiple Discriminant Analysis (MDA) on economic data.

This module provides utilities to apply MDA to economic indicators data
to identify potential recessions.
"""

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import logging

# Configure logging
logger = logging.getLogger(__name__)


def load_data_for_mda(file_path, target_col='recession'):
    """
    Load and prepare data for MDA.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file with data
    target_col : str
        Name of the target column
        
    Returns
    -------
    tuple
        X (features) and y (target) for MDA
    """
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        if target_col not in data.columns:
            logger.error(f"Error: '{target_col}' column not found in the data.")
            return None, None
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None


def apply_mda(X, y, test_size=0.3, random_state=42):
    """
    Apply Multiple Discriminant Analysis (MDA) and evaluate the model.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Fitted MDA model
        - 'X_train': Training features
        - 'X_test': Testing features
        - 'y_train': Training target
        - 'y_test': Testing target
        - 'y_pred': Predictions on test set
        - 'accuracy': Accuracy score
        - 'conf_matrix': Confusion matrix
        - 'class_report': Classification report
        - 'cv_scores': Cross-validation scores
        - 'feature_importance': Feature importance DataFrame
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    # Apply MDA
    lda = LDA()
    lda.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lda.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{class_report}")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(lda, X, y, cv=cv)
    logger.info(f"Cross-Validation Scores: {cv_scores}")
    logger.info(f"Mean CV Score: {cv_scores.mean():.4f}")
    
    # Get feature importances (coefficients)
    feature_importance = None
    if hasattr(lda, 'coef_'):
        coef = lda.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(coef)
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"Top 5 Most Important Features:\n{feature_importance.head(5)}")
    
    # Return results
    results = {
        'model': lda,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }
    
    return results


def transform_data_with_mda(lda, X):
    """
    Transform data using the MDA model.
    
    Parameters
    ----------
    lda : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        Fitted MDA model
    X : pandas.DataFrame
        Features to transform
        
    Returns
    -------
    pandas.DataFrame
        Transformed data with discriminant function values
    """
    # Transform the data
    X_lda = lda.transform(X)
    
    # Create a DataFrame with the discriminant function values
    lda_df = pd.DataFrame({
        'Discriminant': X_lda.flatten()
    }, index=X.index)
    
    return lda_df


def create_discriminant_time_series(lda, X, y=None):
    """
    Create a time series of discriminant function values.
    
    Parameters
    ----------
    lda : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        Fitted MDA model
    X : pandas.DataFrame
        Features
    y : pandas.Series, optional
        Target (recession indicator)
        
    Returns
    -------
    pandas.DataFrame
        Time series of discriminant function values
    """
    # Transform the data
    X_lda = lda.transform(X)
    
    # Create a DataFrame with the discriminant function values
    lda_df = pd.DataFrame({
        'Discriminant': X_lda.flatten()
    }, index=X.index)
    
    # Add recession indicator if available
    if y is not None:
        lda_df['Recession'] = y
    
    return lda_df


def predict_recession_probability(lda, X):
    """
    Predict recession probabilities.
    
    Parameters
    ----------
    lda : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        Fitted MDA model
    X : pandas.DataFrame
        Features
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with recession probabilities
    """
    # Predict probabilities
    proba = lda.predict_proba(X)
    
    # Create a DataFrame with probabilities
    proba_df = pd.DataFrame({
        'No_Recession_Prob': proba[:, 0],
        'Recession_Prob': proba[:, 1]
    }, index=X.index)
    
    return proba_df
