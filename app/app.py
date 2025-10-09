# app/app.py [ULTIMATE VERSION: POLISHED UI + ALL FEATURES]

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Personal Expense Forecaster",
    page_icon="💸",
    layout="wide"
)

# --- DATA PROCESSING FUNCTION ---
@st.cache_data
def process_data(df):
    """Takes a raw dataframe and returns a processed daily expenses series."""
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    expenses_df = df[df['transaction_type'] == 'expense'].copy()
    daily_expenses = expenses_df.set_index('date')['amount'].resample('D').sum().fillna(0)
    return daily_expenses, df

# --- UI: SIDEBAR ---
with st.sidebar:
    st.title("💸 Expense Forecaster")
    st.header("Upload Your Data")
    st.markdown("Upload a CSV with `date`, `transaction_type`, `amount`, and `category` columns.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    # st.info("This app uses a champion ARIMA model, selected from a rigorous benchmarking pipeline.")

# --- DATA LOGIC & ERROR HANDLING ---
model = None 
try:
    model_path = os.path.join('models', 'champion_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("**ERROR:** `champion_model.pkl` not found. Please run the training pipeline first (`python scripts/03_train_models.py`).")
    st.stop()

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        daily_expenses, full_df = process_data(user_df)
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
        st.stop()
else:
    # st.sidebar.info("Using default demo data.")
    data_path = os.path.join('data/processed', 'cleaned_transactions.csv')
    try:
        default_df = pd.read_csv(data_path)
        daily_expenses, full_df = process_data(default_df)
    except FileNotFoundError:
        st.error("`cleaned_transactions.csv` not found. Please run `python scripts/01_preprocess_data.py`.")
        st.stop()

# --- UI: MAIN PAGE ---
st.title("Personal Expense Forecasting Dashboard")

# --- NEW: KPI DASHBOARD ---
st.header("Your Financial Snapshot")
col1, col2, col3 = st.columns(3)
last_30_days_spend = daily_expenses.tail(30).sum()
avg_daily_spend = daily_expenses[daily_expenses > 0].mean()
top_category = full_df[full_df['transaction_type'] == 'expense'].groupby('category')['amount'].sum().idxmax()
col1.metric("Spend (Last 30 Days)", f"₹{last_30_days_spend:,.0f}")
col2.metric("Avg. Daily Spend", f"₹{avg_daily_spend:,.0f}")
col3.metric("Top Spending Category", top_category.capitalize())
st.divider()

# --- NEW: TAB-BASED LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📈 Forecast & Budget", "📊 Data Analysis", "🏆 Model Performance"])

with tab1:
    # --- FORECASTING SECTION ---
    st.header("Generate a New Forecast")
    num_days = st.slider("Select number of days to forecast:", 7, 90, 30)
    if 'forecast_df' not in st.session_state: st.session_state.forecast_df = None
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast with the champion ARIMA model..."):
            first_valid_index = daily_expenses.ne(0).idxmax()
            trimmed_daily_expenses = daily_expenses[first_valid_index:]
            log_daily_expenses = np.log1p(trimmed_daily_expenses)
            model_to_fit = ARIMA(log_daily_expenses, order=(1, 1, 1)).fit()
            forecast_log = model_to_fit.forecast(steps=num_days)
            forecast_original = np.expm1(forecast_log)
            st.success("Forecast generated successfully!")
            st.session_state.forecast_df = pd.DataFrame({'Date': forecast_log.index, 'Forecasted Amount': forecast_original})
    if st.session_state.forecast_df is not None:
        forecast_df = st.session_state.forecast_df
        forecast_df['Forecasted Amount'] = forecast_df['Forecasted Amount'].clip(lower=0)
        st.subheader(f"Forecasted Expenses for the Next {len(forecast_df)} Days")
        st.dataframe(forecast_df, use_container_width=True)
        
        # --- NEW: INTERACTIVE PLOTLY CHART ---
        plot_df = pd.DataFrame({'Historical Expenses': daily_expenses.tail(90)})
        forecast_plot_df = forecast_df.set_index('Date')
        plot_df = pd.concat([plot_df, forecast_plot_df.rename(columns={'Forecasted Amount': 'Forecasted Expenses'})])
        fig = px.line(plot_df, title="Expense Forecast vs. Historical Data", labels={"value": "Amount (₹)", "Date": "Date"})
        fig.update_layout(legend_title_text='Legend')
        st.plotly_chart(fig, use_container_width=True)

        # --- BUDGET OPTIMIZATION ---
        st.header("Budget Optimization")
        total_forecasted = forecast_df['Forecasted Amount'].sum()
        st.metric(label=f"Total Forecasted Spend for {len(forecast_df)} Days", value=f"₹{total_forecasted:,.2f}")
        savings_percentage = st.slider("Select your savings goal (%):", 0, 50, 10, key="savings_slider")
        if st.button("Calculate Budget"):
            target_spend = total_forecasted * (1 - savings_percentage / 100)
            reduction_needed = total_forecasted - target_spend
            st.subheader("Your Optimized Budget")
            b_col1, b_col2, b_col3 = st.columns(3); b_col1.metric("Original Forecast", f"₹{total_forecasted:,.2f}"); b_col2.metric("Savings Goal", f"₹{reduction_needed:,.2f}"); b_col3.metric("New Target Budget", f"₹{target_spend:,.2f}")
            st.markdown("#### Suggested Reductions by Category")
            non_essential_cats = ['food', 'shopping', 'entertainment', 'travel', 'other']
            expense_by_cat = full_df[full_df['transaction_type'] == 'expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
            top_cats_filtered = expense_by_cat[expense_by_cat.index.isin(non_essential_cats)].head(3)
            if top_cats_filtered.empty:
                st.warning("Could not identify top discretionary spending categories.")
            else:
                reductions = (top_cats_filtered / top_cats_filtered.sum()) * reduction_needed
                st.markdown("To meet your savings goal, consider reducing these areas:")
                for category, reduction_amount in reductions.items():
                    st.write(f"- **{category.capitalize()}:** Reduce by **₹{reduction_amount:,.2f}**")

with tab2:
    # --- EDA DASHBOARD ---
    st.header("Exploratory Analysis of Your Spending")
    fig_col1, fig_col2 = st.columns(2)
    expense_by_cat_eda = full_df[full_df['transaction_type'] == 'expense'].groupby('category')['amount'].sum().sort_values(ascending=False)
    with fig_col1:
        st.markdown("#### Total Spend per Category (Top 10)")
        fig1 = px.pie(expense_by_cat_eda.head(10), values='amount', names=expense_by_cat_eda.head(10).index, title='Top 10 Spending Categories')
        st.plotly_chart(fig1, use_container_width=True)
    with fig_col2:
        st.markdown("#### Top 10 Most Frequent Categories")
        top_10_cats = full_df[full_df['transaction_type'] == 'expense']['category'].value_counts().nlargest(10)
        fig2 = px.bar(top_10_cats, x=top_10_cats.values, y=top_10_cats.index, orientation='h', title='Top 10 Most Frequent Categories', labels={'x': 'Number of Transactions', 'y': 'Category'})
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # --- MODEL COMPARISON SECTION ---
    st.header("Model Performance Comparison")
    st.markdown("We ran a full pipeline to benchmark multiple models. The best model is highlighted for each metric.")
    eval_data = {
        'MAE (₹)': [34008.95, 40765.15, 220219.29, 7281243.57],
        'RMSE (₹)': [127683.89, 112313.23, 236988.83, 28357192.37],
        'MAPE (%)': [379.43, 418.91, 3316.11, 30792.19]
    }
    eval_df = pd.DataFrame(eval_data, index=['ARIMA', 'Random Forest', 'XGBoost', 'LightGBM'])
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("#### MAE (Lower is Better)")
        mae_df = eval_df[['MAE (₹)']].sort_values(by='MAE (₹)', ascending=True)
        fig = px.bar(mae_df, x='MAE (₹)', y=mae_df.index, orientation='h', color_discrete_sequence=px.colors.sequential.Viridis); st.plotly_chart(fig, use_container_width=True)
    with m_col2:
        st.markdown("#### RMSE (Lower is Better)")
        rmse_df = eval_df[['RMSE (₹)']].sort_values(by='RMSE (₹)', ascending=True)
        fig = px.bar(rmse_df, x='RMSE (₹)', y=rmse_df.index, orientation='h', color_discrete_sequence=px.colors.sequential.Plasma); st.plotly_chart(fig, use_container_width=True)
    with m_col3:
        st.markdown("#### MAPE (Lower is Better)")
        mape_df = eval_df[['MAPE (%)']].sort_values(by='MAPE (%)', ascending=True)
        fig = px.bar(mape_df, x='MAPE (%)', y=mape_df.index, orientation='h', color_discrete_sequence=px.colors.sequential.Magma); st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conclusion:** The **ARIMA** model was chosen as the champion for its superior accuracy in predicting currency amounts (lowest MAE).")


# Save this `app.py` file and rerun your application from the terminal:
# ```powershell
# python -m streamlit run app/app.py
