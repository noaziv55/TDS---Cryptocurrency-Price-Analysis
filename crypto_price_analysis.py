import pandas as pd
import numpy as np

# Machine learning library
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

import warnings

warnings.filterwarnings("ignore")

# Define colors for each cryptocurrency
crypto_colors = {
    'Bitcoin (BTC)': 'midnightblue',
    'Ethereum (ETH)': 'cyan',
    'Litecoin (LTC)': 'blue',
    'Ripple (XRP)': 'pink'
}


# Function to impute missing values with the average of the day before and day after
def impute_missing_values(row):
    if pd.isnull(row['Close']):
        before_previous_day = dtf[
            (dtf['Crypto'] == row['Crypto']) & (dtf['Date'] == (row['Date'] - pd.DateOffset(2)))].squeeze()
        previous_day = dtf[
            (dtf['Crypto'] == row['Crypto']) & (dtf['Date'] == (row['Date'] - pd.DateOffset(1)))].squeeze()
        next_day = dtf[(dtf['Crypto'] == row['Crypto']) & (dtf['Date'] == (row['Date'] + pd.DateOffset(1)))].squeeze()
        after_next_day = dtf[
            (dtf['Crypto'] == row['Crypto']) & (dtf['Date'] == (row['Date'] + pd.DateOffset(2)))].squeeze()

        if not pd.isnull(previous_day['Close']) and not pd.isnull(next_day['Close']):
            row['Close'] = (previous_day['Close'] + next_day['Close']) / 2
            row['Open'] = (previous_day['Open'] + next_day['Open']) / 2
            row['High'] = (previous_day['High'] + next_day['High']) / 2
            row['Low'] = (previous_day['Low'] + next_day['Low']) / 2
        elif not pd.isnull(previous_day['Close']) and not pd.isnull(after_next_day['Close']):
            row['Close'] = (previous_day['Close'] + after_next_day['Close']) / 2
            row['Open'] = (previous_day['Open'] + after_next_day['Open']) / 2
            row['High'] = (previous_day['High'] + after_next_day['High']) / 2
            row['Low'] = (previous_day['Low'] + after_next_day['Low']) / 2
        elif not pd.isnull(before_previous_day['Close']) and not pd.isnull(next_day['Close']):
            row['Close'] = (before_previous_day['Close'] + next_day['Close']) / 2
            row['Open'] = (before_previous_day['Open'] + next_day['Open']) / 2
            row['High'] = (before_previous_day['High'] + next_day['High']) / 2
            row['Low'] = (before_previous_day['Low'] + next_day['Low']) / 2
        else:
            row['Close'] = (before_previous_day['Close'] + after_next_day['Close']) / 2
            row['Open'] = (before_previous_day['Open'] + after_next_day['Open']) / 2
            row['High'] = (before_previous_day['High'] + after_next_day['High']) / 2
            row['Low'] = (before_previous_day['Low'] + after_next_day['Low']) / 2

    return row


def run_impute_missing_values(dtf_list_miss, dtf_names):
    dtf_list = []
    for (dtf, name) in zip(dtf_list_miss, dtf_names):
        # Convert 'Date' column to datetime format
        dtf['Date'] = pd.to_datetime(dtf['Date'], format='%m/%d/%y')

        # Sort the DataFrame by 'Date'
        dtf = dtf.sort_values(by=['Crypto', 'Date'])

        # Apply the imputation function to fill missing values
        dtf_copy = dtf.copy()
        dtf_copy = dtf_copy.apply(impute_missing_values, axis=1)

        dtf_list.append(dtf_copy)

        # Save the imputed dataset
        dtf.to_csv('imputed_dataset_' + name + '.csv', index=False)

    return dtf_list


# Function to build and evaluate the enhanced ML pipeline
def build_and_evaluate_enhanced_pipeline(dtf):
    # Specify features and target variable
    X = dtf[['Open', 'High', 'Low']]
    y = dtf['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a simple ML pipeline with XGBoost
    model = Pipeline([
        ('scaler', StandardScaler()),  # Data preprocessing
        ('regressor', XGBRegressor())  # XGBoost model
    ])

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate Percentage MAE (PMAE)
    pmae = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Implement a 10-fold cross-validation process
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Calculate Percentage MAE (PMAE) for cross-validation
    cv_pmae = np.mean(np.abs(cv_scores / y.mean())) * 100

    # Placeholder for risk metrics (customize based on your needs)
    # Example: Calculate volatility
    volatility = np.std(y_pred - y_test)

    return mae, pmae, cv_mae, cv_pmae, volatility


def run_initial_model(dtf_list, dtf_names):
    # Initialize an empty list to store results
    results = []

    # Apply the enhanced pipeline to each DataFrame
    for i, (dtf, name) in enumerate(zip(dtf_list, dtf_names)):
        # Build and evaluate the enhanced pipeline
        mae, pmae, cv_mae, cv_pmae, volatility = build_and_evaluate_enhanced_pipeline(dtf_list[i])

        # Store the results in a dictionary and append it to the results list
        results.append({
            'Crypto': name,
            'MAE': mae,
            'PMAE': pmae,
            'Cross-Validation MAE': cv_mae,
            'Cross-Validation PMAE': cv_pmae,
            'Volatility': volatility
        })

    # Display the comparison across cryptocurrencies
    print("\nComparison Across Cryptocurrencies:")
    print("{:<15} {:<15} {:<15} {:<25} {:<20}".format('Cryptocurrency', 'MAE', 'PMAE', 'Cross-Validation MAE',
                                                      'Cross-Validation PMAE'))

    for result in results:
        print("{:<15} {:<15} {:<15} {:<25} {:<20}".format(
            result['Crypto'],
            f"{result['MAE']:.4f}",
            f"{result['PMAE']:.4f}%",
            f"{result['Cross-Validation MAE']:.4f}",
            f"{result['Cross-Validation PMAE']:.4f}%"
        ))

    return results


# Calculate Relative Strength Index (RSI) for a single cryptocurrency
def calculate_rsi(dtf, window_size=14):
    delta = dtf['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi



def improved_model(dtf):
    # Convert Date column to datetime
    dtf['Date'] = pd.to_datetime(dtf['Date'])

    # dtf['Date'] = pd.to_datetime(dtf['Date'], format='%m/%d/%y')

    # Calculate Daily Returns
    dtf['Daily_Return'] = dtf.groupby('Crypto')['Close'].pct_change() * 100

    # Specify the window sizes for moving averages
    window_sizes = [5, 10, 20]
    # Calculate Moving Averages for each window size
    for window_size in window_sizes:
        dtf[f'MA_{window_size}'] = dtf.groupby('Crypto')['Close'].rolling(window=window_size).mean().reset_index(
            level=0, drop=True)

    dtf['RSI'] = calculate_rsi(dtf)

    # Specify features and target variable
    features = ['Open', 'High', 'Low']
    target = 'Close'

    # Add the new features to the list of features
    new_features = ['Daily_Return', 'RSI', 'MA_5', 'MA_10', 'MA_20']
    all_features = features + new_features

    # Select the features and target variable from the DataFrame
    X = dtf[all_features]
    y = dtf[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a simple ML pipeline with XGBoost
    model = Pipeline([
        ('scaler', StandardScaler()),  # Data preprocessing
        ('regressor', XGBRegressor(importance_type='weight'))  # XGBoost model
    ])

    # Hyperparameter tuning using Random Search
    param_dist = {
        'regressor__n_estimators': randint(100, 500),
        'regressor__max_depth': randint(3, 5),
        'regressor__learning_rate': uniform(0.05, 0.1),
        'regressor__subsample': uniform(0.8, 0.2),
        'regressor__colsample_bytree': uniform(0.8, 0.2)
    }

    # Instantiate Random Search with the model and hyperparameter grid
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=5,
                                       scoring='roc_auc', verbose=1, n_jobs=-1)

    # Fit the Random Search to the training data
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters found by Random Search
    best_params = random_search.best_params_
    print(f"Best Hyperparameters: {best_params}\n")

    # Update the model with the best hyperparameters
    model.set_params(**best_params)

    # Fit the updated model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate Percentage MAE (PMAE)
    pmae = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Cross-validation process and metrics
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Calculate Percentage MAE (PMAE) for cross-validation
    cv_pmae = np.mean(np.abs(cv_scores / y.mean())) * 100

    # Calculate Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(dtf['Daily_Return'])

    return mae, pmae, cv_mae, cv_pmae, volatility, model, X_test, y_test, y_pred


def run_improved_model(dtf_list, dtf_names):
    # Initialize an empty list to store results
    results = []

    # Apply the enhanced pipeline to each DataFrame
    for i, (dtf, name) in enumerate(zip(dtf_list, dtf_names)):
        print("Enhanced Pipeline Evaluation for " + name + ":")

        # Build and evaluate the enhanced pipeline
        mae, pmae, cv_mae, cv_pmae, volatility, model, X_test, y_test, y_pred = improved_model(dtf_list[i])

        # Store the results in a dictionary and append it to the results list
        results.append({
            'Crypto': name,
            'MAE': mae,
            'PMAE': pmae,
            'Cross-Validation MAE': cv_mae,
            'Cross-Validation PMAE': cv_pmae,
            'Volatility': volatility,
            'model': model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        })

    # Display the comparison across cryptocurrencies
    print("\nComparison Across Cryptocurrencies:")
    print("{:<20} {:<10} {:<10} {:<20} {:<20} {:<30}".format('Cryptocurrency', 'MAE', 'PMAE %', 'Cross-Validation MAE',
                                                             'Cross-Validation PMAE %', 'Volatility'))

    for result in results:
        print("{:<20} {:<10} {:<10} {:<20} {:<20} {:<30}".format(
            result['Crypto'],
            f"{result['MAE']:.4f}",
            f"{result['PMAE']:.4f}",
            f"{result['Cross-Validation MAE']:.4f}",
            f"{result['Cross-Validation PMAE']:.4f}",
            f"{result['Volatility']:.4f}"
        ))

    # Extract the relevant metrics
    metrics = ['MAE', 'PMAE', 'Cross-Validation MAE', 'Cross-Validation PMAE', 'Volatility']

    # Initialize lists to store the normalized metrics
    normalized_metrics = {metric: [] for metric in metrics}

    # Extract the metrics for each cryptocurrency
    for result in results:
        for metric in metrics:
            normalized_metrics[metric].append(result[metric])

    # Apply min-max normalization to each metric
    for metric in metrics:
        min_value = min(normalized_metrics[metric])
        max_value = max(normalized_metrics[metric])
        normalized_metrics[metric] = [(x - min_value) / (max_value - min_value) for x in normalized_metrics[metric]]

    # Print the normalized metrics along with the cryptocurrency names
    print("\nComparison Across The Normalized Metrics Of Cryptocurrencies:")
    print("{:<20} {:<10} {:<10} {:<20} {:<20} {:<30}".format('Cryptocurrency', 'MAE', 'PMAE %', 'Cross-Validation MAE',
                                                             'Cross-Validation PMAE %', 'Volatility'))
    for result, normalized_metric in zip(results, zip(*normalized_metrics.values())):
        print("{:<20} {:<10.4f} {:<10.4f} {:<20.4f} {:<20.4f} {:<30.4f}".format(
            result['Crypto'],
            *normalized_metric
        ))

    return results, normalized_metrics


def create_radar_chart(ax, crypto, metrics, values, color):
    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Repeat the first angle to close the circle

    # Plot data
    ax.plot(angles, values + values[:1], color=color, linewidth=1, linestyle='solid', label=crypto)

    # Fill area
    ax.fill(angles, values + values[:1], color=color, alpha=0.25)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Set the title of the radar chart
    ax.set_title(f'Radar Chart for normalized metrics of all cryptocurrencies')

    # Add a legend
    ax.legend(loc='upper right')


def get_performance(initial_results, improved_results):
    performance_comparison = []

    # Loop through each result dictionary in the initial_results list
    for result_initial, result_improved in zip(initial_results, improved_results):
        # Assuming the results are in the same order in both initial_results and improved_results
        crypto = result_initial['Crypto']

        # Calculate the difference in MAE and cross-validation MAE between initial and improved models
        mae_difference = result_improved['MAE'] - result_initial['MAE']
        cv_mae_difference = result_improved['Cross-Validation MAE'] - result_initial['Cross-Validation MAE']

        # Store the results in a dictionary
        comparison_result = {
            'Crypto': crypto,
            'MAE Difference': mae_difference,
            'CV MAE Difference': cv_mae_difference
        }

        # Append the comparison result to the list
        performance_comparison.append(comparison_result)

    return performance_comparison


def get_params_for_SHAP(crypto_name, results):
    for result in results:
        if result['Crypto'] == crypto_name:
            # Extract the trained XGBoost model from the trained_model dictionary
            xgboost_model = result['model']['regressor']
            X_test = result['X_test']
            return xgboost_model, X_test
