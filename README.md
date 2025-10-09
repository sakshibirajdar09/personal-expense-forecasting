Personal Expense Forecasting System 💸
 <!-- You will update this URL after deployment -->

An end-to-end data science project that cleans financial transaction data, benchmarks multiple forecasting models, and deploys a champion model in a feature-rich, interactive Streamlit web application.

<!-- Replace with a URL to a screenshot of your app -->

✨ Features
Dynamic Data Upload: Users can upload their own CSV transaction data for a personalized forecast.

KPI Dashboard: An at-a-glance summary of key financial metrics like recent spending and top categories.

Interactive EDA: A dedicated tab with interactive Plotly charts for exploring spending habits.

Time Series Forecasting: Generates a daily expense forecast for the next 7-90 days using a trained ARIMA model.

Budget Optimization: Provides actionable recommendations for meeting user-defined savings goals.

Model Benchmarking: A visual comparison of four different models (ARIMA, Random Forest, XGBoost, LightGBM) across multiple performance metrics.

🛠️ Tech Stack
Language: Python

Data Science: Pandas, NumPy, Scikit-learn

Modeling: Statsmodels (ARIMA), LightGBM, XGBoost, Scikit-learn (RandomForest)

Web Framework: Streamlit

Plotting: Plotly, Matplotlib, Seaborn

📂 Project Structure
The project is structured as a professional data science pipeline:

personal_expense_forecasting/
├── app/
│   └── app.py              # The final Streamlit application script
├── data/
│   ├── raw/                # Raw, untouched datasets
│   └── processed/          # Cleaned and feature-engineered data
├── models/
│   └── champion_model.pkl  # The final, best-performing model
├── notebooks/
│   └── ...                 # Jupyter notebooks for exploration and research
├── scripts/
│   ├── 01_preprocess_data.py   # Pipeline: Cleans raw data
│   ├── 02_create_features.py   # Pipeline: Creates features for ML
│   └── 03_train_models.py      # Pipeline: Trains and benchmarks all models
├── .streamlit/
│   └── config.toml         # Custom theme for the Streamlit app
├── .gitignore
├── README.md
└── requirements.txt        # All Python dependencies

⚙️ How to Run Locally
Clone the repository:

git clone [https://github.com/your-username/personal-expense-forecasting.git](https://github.com/your-username/personal-expense-forecasting.git)
cd personal-expense-forecasting

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

Run the Data Science Pipeline:
Execute the scripts in order to generate the processed data and the champion model.

python scripts/01_preprocess_data.py
python scripts/02_create_features.py
python scripts/03_train_models.py

Launch the Streamlit App:

streamlit run app/app.py

🎯 Model Performance
A comprehensive pipeline was run to evaluate four models. The ARIMA model was selected as the champion due to its superior Mean Absolute Error (MAE), making it the most accurate for predicting the actual currency amount on an average day.

Metric

ARIMA

Random Forest

XGBoost

LightGBM

MAE (₹)

34,008.95

40,765.15

220,219.29

7,281,243.57

RMSE (₹)

127,683.89

112,313.23

236,988.83

28,357,192.37

MAPE (%)

379.43

418.91

3,316.11

30,792.19

Dir. Acc. (%)

8.47

69.49

62.71

61.02

🚀 Future Improvements
Deploy to Cloud: Deploy the Streamlit application to a public cloud service.

Add External Data: Integrate external features like inflation rates or public holiday calendars.

Hyperparameter Tuning: Implement automated hyperparameter tuning to further optimize model performance.

Unit Testing: Add formal unit tests for the data processing and feature engineering pipelines.