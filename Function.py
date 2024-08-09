import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def test_models(X, y, models=None, test_size=0.2, random_state=42):
    """
    Test multiple models with different scaling methods.
    
    Parameters:
    X (array-like): Features
    y (array-like): Target variable
    models (dict): Dictionary of models to test. If None, default models will be used.
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random state for reproducibility
    
    Returns:
    dict: Results of all models and scaling methods
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize the features
    scaler_standard = StandardScaler()
    X_train_scaled = scaler_standard.fit_transform(X_train)
    X_test_scaled = scaler_standard.transform(X_test)
    
    # Normalize the data
    scaler_minmax = MinMaxScaler()
    X_train_normalized = scaler_minmax.fit_transform(X_train)
    X_test_normalized = scaler_minmax.transform(X_test)
    
    # Define default models if none provided
    if models is None:
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Support Vector Machine': SVR(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state)
        }
    
    results = {}
    
    for name, model in models.items():
        results[name] = {}
        
        # Standardized data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name]['Standardized'] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Normalized data
        model.fit(X_train_normalized, y_train)
        y_pred = model.predict(X_test_normalized)
        results[name]['Normalized'] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    
    return results

# Function to print results
def print_results(results):
    for model_name, scaling_results in results.items():
        print(f"\n{model_name}:")
        for scaling_method, metrics in scaling_results.items():
            print(f"  {scaling_method} - MAE: {metrics['MAE']:.2f}, MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.2f}")





def read_house_price_data(file_path):
    import pandas as pd
    """
    Read house price data from an Excel file, drop null values, and remove the 'Id' column.
    
    Parameters:
    file_path (str): Path to the Excel file containing the house price data.
    
    Returns:
    pandas.DataFrame: Preprocessed house price data.
    """
    # Read the Excel file
    data = pd.read_excel(file_path)
    
    # Drop rows with null values
    data = data.dropna()
    
    # Remove the 'Id' column
    data = data.drop(['Id'], axis=1)
    
    return data
