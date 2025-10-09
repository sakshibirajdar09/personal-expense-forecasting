# scripts/02_create_features.py

import pandas as pd
import numpy as np
import os

def run_feature_engineering():
    """Main function to execute the feature engineering pipeline."""
    print("🚀 [START] Running Feature Engineering Pipeline (with Holiday Features)...")

    # Define file paths
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
    
    # 1. Load Cleaned Data
    print("  - Step 1: Loading cleaned dataset...")
    try:
        input_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_transactions.csv')
        df = pd.read_csv(input_path, parse_dates=['date'])
        print(f"    - Loaded {len(df)} cleaned records.")
    except FileNotFoundError:
        print(f"    - ❌ ERROR: cleaned_transactions.csv not found. Please run 01_preprocess_data.py first.")
        return

    # 2. Aggregate to Daily Time Series
    print("  - Step 2: Aggregating data to daily expenses...")
    expenses_df = df[df['transaction_type'] == 'expense'].copy()
    daily_expenses = expenses_df.set_index('date')['amount'].resample('D').sum().to_frame(name='total_daily_expense')
    daily_outliers = expenses_df.set_index('date')['is_outlier'].resample('D').sum().to_frame(name='outlier_count_today')
    daily_df = daily_expenses.join(daily_outliers).fillna(0)
    print(f"    - Aggregated data into {len(daily_df)} daily records.")

    # 3. Create Features
    print("  - Step 3: Engineering features...")
    
    # Time-based features
    daily_df['dayofweek'] = daily_df.index.dayofweek
    daily_df['month'] = daily_df.index.month
    daily_df['year'] = daily_df.index.year
    daily_df['dayofyear'] = daily_df.index.dayofyear
    
    # Payday features
    daily_df['is_month_start'] = (daily_df.index.day <= 7).astype(int)
    
    # --- NEW: Holiday Features ---
    # List of major Indian public holidays for the relevant years in your data
    indian_holidays = [
        '2019-01-26', '2019-03-21', '2019-08-15', '2019-10-02', '2019-10-27', '2019-12-25',
        '2020-01-26', '2020-03-10', '2020-08-15', '2020-10-02', '2020-11-14', '2020-12-25',
        '2021-01-26', '2021-03-29', '2021-08-15', '2021-10-02', '2021-11-04', '2021-12-25',
        '2022-01-26', '2022-03-18', '2022-08-15', '2022-10-02', '2022-10-24', '2022-12-25',
        '2023-01-26', '2023-03-08', '2023-08-15', '2023-10-02', '2023-11-12', '2023-12-25',
        '2024-01-26', '2024-03-25', '2024-08-15', '2024-10-02', '2024-10-31', '2024-12-25',
        '2025-01-26', '2025-03-14', '2025-08-15', '2025-10-02', '2025-10-20', '2025-12-25'
    ]
    holiday_dates = pd.to_datetime(indian_holidays)
    daily_df['is_holiday'] = daily_df.index.normalize().isin(holiday_dates).astype(int)
    print("    - Added new 'is_holiday' feature.")
    
    # Lag features
    daily_df['lag_1'] = daily_df['total_daily_expense'].shift(1)
    daily_df['lag_7'] = daily_df['total_daily_expense'].shift(7)
    
    # Rolling window features
    daily_df['rolling_mean_7'] = daily_df['total_daily_expense'].rolling(window=7).mean()
    daily_df['rolling_std_7'] = daily_df['total_daily_expense'].rolling(window=7).std()
    
    # Outlier-based features
    daily_df['rolling_outlier_sum_7'] = daily_df['outlier_count_today'].rolling(window=7).sum()

    daily_df.dropna(inplace=True)
    print("    - Successfully created all features.")

    # 4. Save Feature-Rich Data
    print("  - Step 4: Saving final training data...")
    output_path = os.path.join(PROCESSED_DATA_DIR, 'featured_training_data.csv')
    daily_df.to_csv(output_path)
    
    print(f"    - Successfully saved feature-rich data to {output_path}")
    print("✅ [SUCCESS] Feature engineering pipeline finished.")
    
    return daily_df

if __name__ == "__main__":
    featured_data = run_feature_engineering()
    if featured_data is not None:
        print("\n--- Feature Engineering Test Output ---")
        print("Final DataFrame shape:", featured_data.shape)
        print("New 'is_holiday' column stats:", featured_data['is_holiday'].value_counts())
        print("Final DataFrame head:")
        print(featured_data.head())