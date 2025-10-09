# 🚀 Personal Expense Forecasting System 💸  

An advanced, end-to-end **data science system** that leverages a professional, multi-script pipeline to benchmark a suite of machine learning models and deploy a **champion forecaster** in a feature-rich, interactive **Streamlit** web application.

---

## 🌟 Key Highlights  

- 🏆 **Champion Model (ARIMA):** Selected as the most accurate model for predicting daily spending with the lowest Mean Absolute Error (MAE).  
- 🤖 **Automated ML Pipeline:** End-to-end automation from data cleaning to model training and evaluation.  
- 🧠 **Multi-Model Benchmarking:** Benchmarks 4 different model families — *Statistical, Random Forest, XGBoost, LightGBM* — across 4 key metrics.  
- 📊 **Fully Interactive Dashboard:** Production-ready Streamlit app featuring KPIs, dynamic forecasting, budget optimization, and Plotly charts.  
- 💡 **Data-Driven Insights:** Explore spending habits and visualize model performance transparently.  
- ⚙️ **Professional & Modular Structure:** Clean project organization with dedicated folders for app, scripts, models, and utilities.  

---

## ✨ Features  

### 🎯 Core Capabilities  
- **AI-Powered Forecasting:** Predicts daily expenses for a customizable 7–90 day horizon using the ARIMA champion model.  
- **Multi-Model Architecture:** Easily test and compare multiple model families.  
- **Interactive Web App:** Designed for both technical and non-technical users.  
- **Budget Optimization Engine:** Provides actionable recommendations to achieve savings goals.  

### 🔬 Advanced Data Science Pipeline  
- **Enhanced Preprocessing:** Cleans, standardizes, and enriches data with outlier and imputation flags.  
- **Advanced Feature Engineering:** Automatically generates 10+ predictive features such as holiday flags, payday indicators, lags, and rolling stats.  
- **Comprehensive Evaluation:** Models evaluated on MAE, RMSE, MAPE, and Directional Accuracy.  

---

## 📊 Interactive Dashboard Features  

- 🏠 **Main Dashboard:** Forecasting and budget optimization.  
- 📈 **Data Analysis Tab:** Interactive Plotly charts for historical spending and category breakdowns.  
- 🏆 **Model Comparison Tab:** Visualizes and compares performance of all benchmarked models for transparency.  

---

## 🛠️ Technology Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Core Framework** | Python 3.9+, NumPy, Pandas |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Time Series** | Statsmodels (ARIMA) |
| **Web Framework** | Streamlit |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Pipeline** | Modular Python scripts for reproducible workflow |

---

## 📂 Project Structure  

```
personal_expense_forecasting/
├── 📂 app/
│   └── 📜 app.py              # Streamlit application script
├── 📂 data/
│   ├── 📂 raw/                # Original, untouched datasets
│   └── 📂 processed/          # Cleaned & feature-engineered data
├── 📂 models/
│   └── 🏆 champion_model.pkl  # Best-performing ARIMA model
├── 📂 notebooks/
│   └── 🔬 ...                 # Jupyter notebooks for research
├── 📂 scripts/
│   ├── 📜 01_preprocess_data.py   # Cleans & flags raw data
│   ├── 📜 02_create_features.py   # Generates advanced features
│   └── 📜 03_train_models.py      # Trains, benchmarks & saves model
├── 📂 .streamlit/
│   └── 📜 config.toml         # Custom Streamlit theme
├── 📜 .gitignore
├── 📜 README.md
└── 📜 requirements.txt        # Dependencies
```

---

## 🎯 Final Model Performance  

| Metric | **ARIMA** | Random Forest | XGBoost | LightGBM |
|:-------|:----------:|:--------------:|:--------:|:---------:|
| **MAE (₹)** | **34,008.95** | 40,765.15 | 220,219.29 | 7,281,243.57 |
| **RMSE (₹)** | **127,683.89** | 112,313.23 | 236,988.83 | 28,357,192.37 |
| **MAPE (%)** | **379.43** | 418.91 | 3,316.11 | 30,792.19 |
| **Directional Accuracy (%)** | 8.47 | **69.49** | 62.71 | 61.02 |

> 🏆 The **ARIMA** model emerged as the champion due to its superior **MAE** performance.

---

## ⚙️ How to Run Locally  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/personal-expense-forecasting.git
cd personal-expense-forecasting
```

### 2️⃣ Set Up the Environment  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Data Science Pipeline  
Execute the pipeline scripts in order (only once needed):  
```bash
python scripts/01_preprocess_data.py
python scripts/02_create_features.py
python scripts/03_train_models.py
```

### 4️⃣ Launch the Streamlit App  
```bash
streamlit run app/app.py
```

---

## 🚀 Future Improvements  

- ☁️ **Deployment:** Host the Streamlit app on Streamlit Cloud or other platforms.  
- 🎯 **Hyperparameter Tuning:** Integrate GridSearchCV or Optuna for automated tuning.  
- 🌍 **External Data Integration:** Use public holidays, macroeconomic data, or inflation indicators.  
- 🧪 **Unit Testing:** Add tests for preprocessing and feature engineering to ensure pipeline robustness.  

---

**💡 Developed with ❤️ for Data Science and AI Innovation**
