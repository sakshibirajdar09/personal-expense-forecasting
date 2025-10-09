# scripts/01_preprocess_data.py
"""
Enhanced data preprocessing pipeline.
- Loads raw data.
- Cleans and standardizes columns.
- Adds quality flag columns (e.g., is_outlier, amount_was_invalid).
- Saves the processed data to data/processed/cleaned_transactions.csv
"""

import pandas as pd
import numpy as np
import os

def run_preprocessing():
    """Main function to execute the enhanced preprocessing pipeline."""
    
    print("🚀 [START] Running Enhanced Data Preprocessing Pipeline...")

    # Define file paths
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
    
    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # 1. Load Data
    print("  - Step 1: Loading raw datasets...")
    try:
        df1 = pd.read_csv(os.path.join(RAW_DATA_DIR, 'budgetwise_finance_dataset.csv'))
        df2 = pd.read_csv(os.path.join(RAW_DATA_DIR, 'budgetwise_synthetic_dirty.csv'))
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"    - Loaded and merged {len(df)} total records.")
    except FileNotFoundError as e:
        print(f"    - ❌ ERROR: Raw data file not found. {e}")
        return

    # 2. Basic Cleaning & Standardization
    print("  - Step 2: Performing basic cleaning and standardization...")
    df.columns = df.columns.str.strip().str.lower()
    df.drop_duplicates(inplace=True)

    # 3. Enhanced Date Cleaning
    print("  - Step 3: Cleaning 'date' column and adding quality flags...")
    df['date_original'] = df['date']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date_imputed'] = df['date'].isnull()
    # Simple imputation: fill missing dates with the mode (most frequent date)
    if not df['date'].isnull().all():
        mode_date = df['date'].mode()[0]
        df['date'].fillna(mode_date, inplace=True)
    
    # 4. Enhanced Amount Cleaning
    print("  - Step 4: Cleaning 'amount' column and adding quality flags...")
    df['amount_original'] = df['amount']
    # Force to string for robust cleaning
    df['amount'] = df['amount'].astype(str).str.replace(r'[$,]', '', regex=True).str.strip()
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    df['amount_was_invalid'] = df['amount'].isnull()
    # Simple imputation: fill invalid amounts with the median of their category
    df['amount'] = df.groupby('category')['amount'].transform(lambda x: x.fillna(x.median()))
    # If any still remain (e.g., whole category was invalid), fill with global median
    df['amount'].fillna(df['amount'].median(), inplace=True)

    # 5. Outlier Detection
    print("  - Step 5: Detecting and flagging outliers...")
    # Use IQR method for outlier detection on expense amounts
    expense_amounts = df[df['transaction_type'] == 'expense']['amount']
    Q1 = expense_amounts.quantile(0.25)
    Q3 = expense_amounts.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    
    df['is_outlier'] = (df['transaction_type'] == 'expense') & (df['amount'] > outlier_threshold)
    print(f"    - Found {df['is_outlier'].sum()} outliers based on IQR method.")

    # 6. Categorical Cleaning
    print("  - Step 6: Cleaning categorical columns...")
    for col in ['transaction_type', 'category', 'payment_mode', 'location']:
        df[col] = df[col].str.strip().str.lower().fillna('unknown')
    
    # Simple standardization mappings
    payment_mapping = {'card': 'credit card', 'csh': 'cash'}
    df['payment_mode'] = df['payment_mode'].replace(payment_mapping)
    
    # 7. Final selection and save
    print("  - Step 7: Saving processed data...")
    final_cols = [
        'transaction_id', 'user_id', 'date', 'transaction_type', 'category', 
        'amount', 'payment_mode', 'location', 'notes', 'is_outlier', 
        'date_imputed', 'amount_was_invalid'
    ]
    # Ensure all columns exist before selecting
    final_df = df[[col for col in final_cols if col in df.columns]]
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'cleaned_transactions.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"    - Successfully saved cleaned data to {output_path}")
    print("✅ [SUCCESS] Enhanced preprocessing pipeline finished.")


if __name__ == "__main__":
    run_preprocessing()