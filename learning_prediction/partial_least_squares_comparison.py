# Partial least squares comparison across multiple variables

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, KFold
from scipy import stats

# Set your prefix here
prefix = '/home/ines/repositories/'
# prefix = '/Users/ineslaranjeira/Documents/Repositories/'

# Load your data (you'll need to run the data loading parts from your original file first)
# This assumes you have already loaded and processed:
# - mat (your feature matrix)
# - syllables_df (your dataframe with target variables)

def run_pls_comparison(mat, syllables_df, test_variables=None):
    """
    Run PLS regression for multiple target variables and compare R² scores
    
    Parameters:
    -----------
    mat : array-like
        Feature matrix (n_samples x n_features)
    syllables_df : DataFrame
        DataFrame containing target variables
    test_variables : list, optional
        List of variable names to test. If None, uses default set.
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame with comparison results
    """
    
    if test_variables is None:
        # Default variables to test
        test_variables = ['log_training_time', 'log_elongation', 'choice', 'correct', 'log_reaction', 'bias']
    
    # Prepare data once
    X = np.array(mat)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dictionary to store results
    results_dict = {
        'variable': [],
        'cv_r2': [],
        'in_sample_r2': []
    }
    
    # Loop through each variable
    for test_var in test_variables:
        print(f"\nTesting variable: {test_var}")
        
        # Get target variable, skip if not available
        if test_var not in syllables_df.columns:
            print(f"Variable {test_var} not found, skipping...")
            continue
            
        y = syllables_df[test_var].copy()
        
        # Convert to numeric, handling non-numeric values
        y = pd.to_numeric(y, errors='coerce')
        y = np.array(y)
        
        # Remove any NaN values
        valid_idx = ~np.isnan(y)
        X_clean = X_scaled[valid_idx]
        y_clean = y[valid_idx]
        
        if len(y_clean) == 0:
            print(f"No valid data for {test_var}, skipping...")
            continue
        
        # Fit PLS model
        pls = PLSRegression(n_components=min(10, X_clean.shape[1]))
        pls.fit(X_clean, y_clean)
        
        # Calculate in-sample R²
        y_pred = pls.predict(X_clean)
        r2_in = r2_score(y_clean, y_pred)
        
        # Cross-validated R²
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(pls, X_clean, y_clean, cv=cv)
        r2_cv = r2_score(y_clean, y_pred_cv)
        
        # Store results
        results_dict['variable'].append(test_var)
        results_dict['cv_r2'].append(r2_cv)
        results_dict['in_sample_r2'].append(r2_in)
        
        print(f"In-sample R²: {r2_in:.4f}")
        print(f"Cross-validated R²: {r2_cv:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_dict)
    return results_df

def plot_r2_comparison(results_df, plot_type='bar'):
    """
    Create comparison plots for R² scores
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from run_pls_comparison
    plot_type : str
        'bar' for bar plot, 'violin' for violin plot
    """
    
    # Melt the dataframe for plotting
    melted_df = pd.melt(results_df, 
                       id_vars=['variable'], 
                       value_vars=['cv_r2', 'in_sample_r2'],
                       var_name='metric', 
                       value_name='r2_score')
    
    # Create plots
    if plot_type == 'bar':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Cross-validated R²
        cv_data = results_df.sort_values('cv_r2', ascending=False)
        sns.barplot(data=cv_data, x='cv_r2', y='variable', ax=ax1, palette='viridis')
        ax1.set_title('Cross-Validated R²')
        ax1.set_xlabel('R² Score')
        
        # In-sample R²
        in_sample_data = results_df.sort_values('in_sample_r2', ascending=False)
        sns.barplot(data=in_sample_data, x='in_sample_r2', y='variable', ax=ax2, palette='plasma')
        ax2.set_title('In-Sample R²')
        ax2.set_xlabel('R² Score')
        
    elif plot_type == 'violin':
        # For violin plot, we'd need multiple runs to show distribution
        # This is a simplified version showing both metrics side by side
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sns.barplot(data=melted_df, x='r2_score', y='variable', hue='metric', ax=ax)
        ax.set_title('PLS R² Comparison')
        ax.set_xlabel('R² Score')
        
    plt.tight_layout()
    plt.show()

def create_detailed_comparison_plot(results_df):
    """
    Create a detailed comparison plot with both metrics
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sort by cross-validated R² (more reliable metric)
    results_sorted = results_df.sort_values('cv_r2', ascending=True)
    
    y_pos = np.arange(len(results_sorted))
    
    # Plot bars
    bars1 = ax.barh(y_pos - 0.2, results_sorted['cv_r2'], 0.4, 
                   label='Cross-validated R²', color='steelblue', alpha=0.8)
    bars2 = ax.barh(y_pos + 0.2, results_sorted['in_sample_r2'], 0.4, 
                   label='In-sample R²', color='orange', alpha=0.8)
    
    # Add value labels on bars
    for i, (cv_r2, in_r2) in enumerate(zip(results_sorted['cv_r2'], results_sorted['in_sample_r2'])):
        ax.text(cv_r2 + 0.01, i - 0.2, f'{cv_r2:.3f}', va='center', fontsize=9)
        ax.text(in_r2 + 0.01, i + 0.2, f'{in_r2:.3f}', va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['variable'])
    ax.set_xlabel('R² Score')
    ax.set_title('PLS Regression: Cross-validated vs In-sample R²')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage (uncomment and modify as needed):
"""
# After loading your data and processing as in the original file:

# Run the comparison
results = run_pls_comparison(mat, syllables_df)
print("\\nSummary of Results:")
print(results)

# Create plots
plot_r2_comparison(results, plot_type='bar')
create_detailed_comparison_plot(results)

# You can also test specific variables:
custom_variables = ['log_training_time', 'correct', 'bias']
custom_results = run_pls_comparison(mat, syllables_df, custom_variables)
create_detailed_comparison_plot(custom_results)
"""