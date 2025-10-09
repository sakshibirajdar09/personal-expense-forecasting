# scripts/03_train_models.py
"""
Loads the feature-rich training data, trains multiple models, evaluates them,
compares their performance, and saves the champion model.
"""

import pandas as pd
import numpy as np
import pickle
import os

# Import model and metric libraries
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Utility Functions ---
def calculate_safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask): return 0.0
    y_true_safe, y_pred_safe = y_true[non_zero_mask], y_pred[non_zero_mask]
    return np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100

def calculate_directional_accuracy(y_true, y_pred):
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    correct_direction = (np.sign(y_true_diff) == np.sign(y_pred_diff))
    correct_direction[y_true_diff == 0] = True
    return np.mean(correct_direction) * 100

def run_model_training():
    """Main function to execute the model training and evaluation pipeline."""
    print("🚀 [START] Running Model Training & Evaluation Pipeline...")

    # Define file paths
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Feature-Rich Data
    print("  - Step 1: Loading feature-rich dataset...")
    try:
        input_path = os.path.join(PROCESSED_DATA_DIR, 'featured_training_data.csv')
        df = pd.read_csv(input_path, parse_dates=['date'], index_col='date')
        print(f"    - Loaded {len(df)} daily records with {len(df.columns)} features.")
    except FileNotFoundError:
        print(f"    - ❌ ERROR: featured_training_data.csv not found. Please run 02_create_features.py first.")
        return

    results = {}
    
    # 2. Train and Evaluate ARIMA Model
    print("  - Step 2: Training and evaluating ARIMA model...")
    time_series = df['total_daily_expense']
    log_time_series = np.log1p(time_series)
    train_arima, test_arima = log_time_series[:-60], log_time_series[-60:]
    
    model_arima = ARIMA(train_arima, order=(1, 1, 1)).fit()
    preds_log = model_arima.predict(start=len(train_arima), end=len(train_arima) + len(test_arima) - 1)
    preds_arima = np.expm1(preds_log)
    actuals_arima = np.expm1(test_arima)

    results['ARIMA'] = {
        'MAE': mean_absolute_error(actuals_arima, preds_arima),
        'MAPE (%)': calculate_safe_mape(actuals_arima, preds_arima),
        'RMSE': np.sqrt(mean_squared_error(actuals_arima, preds_arima)),
        'Dir. Acc. (%)': calculate_directional_accuracy(actuals_arima.values, preds_arima.values)
    }

    # 3. Prepare Data and Train ML Models
    print("  - Step 3: Training and evaluating Machine Learning models...")
    X = df.drop('total_daily_expense', axis=1)
    y = df['total_daily_expense']
    X_train, X_test = X[:-60], X[-60:]
    y_train, y_test = y[:-60], y[-60:]

    models_to_train = {
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    for name, model in models_to_train.items():
        print(f"    - Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, preds),
            'MAPE (%)': calculate_safe_mape(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'Dir. Acc. (%)': calculate_directional_accuracy(y_test.values, preds)
        }
        
    # 4. Compare Models and Save Champion
    print("  - Step 4: Comparing models and saving the champion...")
    results_df = pd.DataFrame(results).T.sort_values(by='MAE')
    
    print("\n" + "="*60)
    print("--- FINAL MODEL EVALUATION SUMMARY ---")
    print(results_df.to_string(formatters={col: '{:,.2f}'.format for col in results_df.columns}))
    print("="*60 + "\n")

    champion_name = results_df.index[0]
    print(f"🏆 Champion Model: {champion_name} (based on lowest MAE)")
    
    # Retrain champion model on the entire dataset
    if champion_name == 'ARIMA':
        print(f"Retraining {champion_name} on the full time series...")
        final_model = ARIMA(log_time_series, order=(1, 1, 1)).fit()
    else:
        print(f"Retraining {champion_name} on the full feature set...")
        final_model = models_to_train[champion_name]
        final_model.fit(X, y)
        
    # Save the final model
    output_path = os.path.join(MODELS_DIR, 'champion_model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"    - Successfully saved champion model to {output_path}")
    print("✅ [SUCCESS] Model training pipeline finished.")

if __name__ == "__main__":
    run_model_training()