import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_expense_data
from data_loader import load_expense_data

def run_eda():
    # Load and preprocess
    raw_df = load_expense_data()
    df = preprocess_expense_data(raw_df)

    # Category-wise total spend (expenses only)
    category_spend = df[df['income/expense']=='expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=category_spend.values, y=category_spend.index)
    plt.title("Total Spend by Category")
    plt.xlabel("Amount")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig('outputs/figures/category_spend.png')
    plt.show()

    # Monthly spending trend
    monthly_spend = df[df['income/expense']=='expense'].groupby('month')['amount'].sum()
    plt.figure(figsize=(10,6))
    sns.lineplot(x=monthly_spend.index, y=monthly_spend.values, marker='o')
    plt.title("Monthly Expense Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Amount")
    plt.tight_layout()
    plt.savefig('outputs/figures/monthly_trend.png')
    plt.show()

    # Weekday spending pattern
    weekday_spend = df[df['income/expense']=='expense'].groupby('weekday')['amount'].sum()
    plt.figure(figsize=(8,5))
    sns.barplot(x=weekday_spend.index, y=weekday_spend.values)
    plt.title("Weekday Expense Pattern")
    plt.xlabel("Weekday (0=Monday)")
    plt.ylabel("Total Amount")
    plt.tight_layout()
    plt.savefig('outputs/figures/weekday_spend.png')
    plt.show()

if __name__ == "__main__":
    run_eda()
