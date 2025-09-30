import pandas as pd

def preprocess_expense_data(df):
    """
    Preprocess the raw expense dataset:
    - Convert date column to datetime
    - Add year, month, day, weekday
    - Handle missing values
    - Add income/expense flag
    """
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Add date parts
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

    # Handle missing categories or descriptions
    df['category'].fillna('Unknown', inplace=True)
    df['description'].fillna('Unknown', inplace=True)

    # Add income/expense flag (assume positive = expense)
    df['income/expense'] = df['amount'].apply(lambda x: 'expense' if x > 0 else 'income')

    return df

if __name__ == "__main__":
    from data_loader import load_expense_data

    df = load_expense_data()
    processed_df = preprocess_expense_data(df)
    print("Processed dataset shape:", processed_df.shape)
    print(processed_df.head())
