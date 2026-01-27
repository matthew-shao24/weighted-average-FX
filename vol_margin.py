# %% Import Libraries
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import lmoments3 as lm
import seaborn as sns
from scipy.stats import shapiro, normaltest
from scipy.stats import norm

# %%
api_key = "2ca3d3200b0a0371686937b4"  # Exchangerate-API key
base = "USD"
target_currencies = ["KES", "UGX", "RWF"]

# Set historical period
start_date = "1999-01-01"  # earliest date you want
end_date = "2026-01-03"    # latest date you want

# Convert strings to datetime
current_date = datetime.strptime(start_date, "%Y-%m-%d")
end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

# Prepare DataFrame
df_all = pd.DataFrame(columns=["date"] + target_currencies)

# Loop through each day
while current_date <= end_date_dt:
    date_str = current_date.strftime("%Y/%m/%d")  # Exchangerate-API expects YYYY/MM/DD
    print(f"Pulling rates for {date_str}...")  # tracker

    # Construct API URL
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/history/{base}/{current_date.year}/{current_date.month:02}/{current_date.day:02}"
    
    try:
        response = requests.get(url)
        data = response.json()

        # Check if rates are present
        if "conversion_rates" not in data:
            print(f"No data for {date_str}, skipping.")
        else:
            rates = data["conversion_rates"]
            row = {"date": current_date.strftime("%Y-%m-%d")}
            for c in target_currencies:
                row[c] = rates.get(c, None)
            
            df_all = pd.concat([df_all, pd.DataFrame([row])], ignore_index=True)

    except Exception as e:
        print(f"Error on {date_str}: {e}")

    # Move to next day
    current_date += timedelta(days=1)

# Save to CSV
df_all.to_csv("data/daily_fx.csv", index=False)

# %% Import CSV of Daily FX data
script_folder = os.path.dirname(os.path.abspath(__file__))
data_path = script_folder + "\\data\\daily_fx.csv"
df_fx = pd.read_csv(data_path)
df_fx["date"] = pd.to_datetime(df_fx["date"], dayfirst=True)
df_fx = df_fx[df_fx["date"] > "2021-01-01"].copy()

# %% ####################################################################################################
# Price Volatility Component
#########################################################################################################

# %% Calculating daily returns and realised volatility
df_fx["kes_returns"] = np.log(df_fx["KES"]/df_fx["KES"].shift(1))
df_fx = df_fx.dropna(subset=["kes_returns"])
df_fx["rv_kes"] = df_fx["kes_returns"]**2

# This is to aggregate the intraday data into a daily realised RV once intraday data is available
# df_fx["date"] = df_fx["timestamp"].dt.date
# rv_daily = (
#     df_fx
#     .groupby("date")["rv_kes"]
#     .agg("sum")
#     .to_frame(name="RV")
# )

# %% Creating HAR Variables
short_term = 1 # Lady day
medium_term = 5 # Last week
long_term = 22 # Last month

epsilon = 1e-8
df_fx["log_rv"] = np.log(df_fx["rv_kes"] + epsilon)
df_fx["st_rv"] = df_fx["log_rv"].rolling(short_term).mean()
df_fx["mt_rv"] = df_fx["log_rv"].rolling(medium_term).mean()
df_fx["lt_rv"] = df_fx["log_rv"].rolling(long_term).mean()

# %% Predict tomorrows RV
df_fx["forecast_rv"] = df_fx["log_rv"].shift(-1)
df_fx = df_fx.dropna(subset=["st_rv", "mt_rv", "lt_rv", "forecast_rv"])

# Defining variables
X = df_fx[["st_rv", "mt_rv", "lt_rv"]]
y = df_fx["forecast_rv"]

# %% Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% Elastic Net
param_grid = {
    "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1],
    "l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0]
}

# Setting up exhaustive hyperparater search
tscv = TimeSeriesSplit(n_splits=5)
enet = ElasticNet()
grid = GridSearchCV(
    ElasticNet(),
    param_grid,
    cv=tscv
)

# Executes exhaustive hyperparameter search
grid.fit(X_train_scaled, y_train)
print("Best params:", grid.best_params_)

# Extract the best model and extracts the forecasts from that model
model = grid.best_estimator_
y_pred = model.predict(X_test_scaled)
y_pred = pd.Series(y_pred, index=y_test.index)

# %% Forecast Volatility and Chart
test_dates = df_fx.loc[y_test.index, "date"]

plt.figure(figsize=(12,5))
plt.plot(test_dates, y_test, label="Actual Log RV")
plt.plot(test_dates, y_pred, label="HAR + Elastic Net Prediction")
plt.xlabel("Date")
plt.ylabel("Log Realized Variance")
plt.legend()
plt.show()

# %% Out-of-sample forecast
last_st = df_fx['st_rv'].iloc[-1]
last_mt = df_fx['mt_rv'].iloc[-1]
last_lt = df_fx['lt_rv'].iloc[-1]
X_future = pd.DataFrame(
    [[last_st, last_mt, last_lt]], 
    columns=["st_rv", "mt_rv", "lt_rv"]
)
X_future_scaled = scaler.transform(X_future)
y_future_pred = model.predict(X_future_scaled)

rv_forecast = np.exp(y_future_pred[0])
print(rv_forecast)

# %% Autocorrelation
residuals = y_test - y_pred
sm.graphics.tsa.plot_acf(residuals, lags=30)
plt.show()

# %% QQ Plot
sm.qqplot(residuals, line='s')
plt.show()

# %% Hetoerskedasticity
arch_test = het_arch(residuals)
print("ARCH test:", arch_test)

# %% Performance Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
# %% Mincer-Zarnowitz
X_mz = sm.add_constant(y_pred)
mz_model = sm.OLS(y_test, X_mz).fit()
print(mz_model.summary())

# %% ######################################################################
# Tail Risk Component
###########################################################################

# %% Testing the distribution - Histogram
sns.histplot(df_fx['kes_returns'], bins=50, kde=True)

# %% Testing the distribution - Histogram
shapiro_test = shapiro(df_fx['kes_returns'])
dagostino_test = normaltest(df_fx['kes_returns'])
print(shapiro_test, dagostino_test)

# %% Calculating moments
mu = df_fx['kes_returns'].mean()
sigma = df_fx['kes_returns'].std()
skew = df_fx['kes_returns'].skew()
kurt = df_fx['kes_returns'].kurtosis() 

# %% Compute L-moments
l_mom = lm.lmom_ratios(df_fx['kes_returns'].values)
l_skew = l_mom[2]
l_kurt = l_mom[3]

print(f"L-skewness (t3): {l_skew}")
print(f"L-kurtosis (t4): {l_kurt}")

# %% Compute Cornish Fisher VaR
alpha = 0.01  # 1% VaR
z = norm.ppf(alpha)

gamma1 = l_skew 
gamma2 = l_kurt 

z_cf = z + (1/6)*(z**2 - 1)*gamma1 + (1/24)*(z**3 - 3*z)*gamma2 - (1/36)*(2*z**3 - 5*z)*gamma1**2
cf_var = mu + z_cf * sigma

print(f"Cornish-Fisher 1% VaR: {cf_var}")

# %% Kupiec test (might be worth adding formalised tests once intraday data comes)
exceptions = (df_fx['kes_returns'] < cf_var).sum()
expected = alpha * len(df_fx)
print(f"Observed exceptions: {exceptions}, Expected: {expected}")

# %% Compute rolling VaR
window = 22 # One month
alpha = 0.01  # 1% VaR

# Rolling VaR function
def rolling_var(x, alpha=0.01):
    return np.percentile(x, alpha*100)
df_fx['VaR_1pct'] = df_fx['kes_returns'].rolling(window).apply(rolling_var, raw=True)

# Charting Rolling VaR
plt.figure(figsize=(12,5))
plt.plot(df_fx['date'], df_fx['kes_returns'],
         color='gray', alpha=0.5, label='Daily Returns')
plt.plot(df_fx['date'], df_fx['VaR_1pct'],
         color='red', linewidth=2, label='Rolling 1% VaR (22d)')
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('KES Returns and Rolling 1% VaR (22-day)')
plt.legend()
plt.show()

# %% Rolling VaR Exceedance Rate
df_fx['Exceedance'] = (df_fx['kes_returns'] < df_fx['VaR_1pct']).astype(int)
df_fx['Exceedance_Rate'] = df_fx['Exceedance'].rolling(window).mean()

plt.figure(figsize=(12,5))
plt.plot(df_fx['date'], df_fx['Exceedance_Rate'], label='Rolling 1% Exceedance Rate')
plt.axhline(y=alpha, color='red', linestyle='--', label='Expected 1%')
plt.xlabel('Date')
plt.ylabel('Exceedance Rate')
plt.title('Rolling 22-day 1% VaR Exceedance Rate')
plt.legend()
plt.show()

# %% Calculating CVaR
var_1pct = cf_var
returns = df_fx['kes_returns'].values
tail_losses = returns[returns < var_1pct]
cvar_1pct = tail_losses.mean()
print(f"1% CVaR: {cvar_1pct:.6f}")

# %% Calculate Rolling CVaR
window = 22  # rolling window size
alpha = 0.01  # 1% CVaR

def rolling_cvar(x, alpha=0.01):
    var_alpha = np.percentile(x, alpha*100)
    tail_losses = x[x < var_alpha]
    if len(tail_losses) == 0:
        return np.nan
    return tail_losses.mean()

df_fx['CVaR_1pct'] = df_fx['kes_returns'].rolling(window).apply(rolling_cvar, raw=True)

# %% ######################################################################
# Regime Instability Component
###########################################################################

# %% Short-long volatility divergence
short_term = 5
long_term = 22

df_fx['sigma_short'] = df_fx['kes_returns'].rolling(short_term).std()
df_fx['sigma_long'] = df_fx['kes_returns'].rolling(long_term).std()

N_short = short_term
N_long = long_term

numerator = np.abs(np.log(df_fx['sigma_short']) - np.log(df_fx['sigma_long']))
denominator = np.sqrt( (df_fx['sigma_short']**2)/N_short + (df_fx['sigma_long']**2)/N_long )

D_raw = numerator / denominator

# EWMA Smoothing
lambda_ = 0.94

D_t = D_raw.ewm(span=20, adjust=False).mean()  # span can be adjusted
df_fx['D_t'] = D_t
print(df_fx[['date', 'sigma_short', 'sigma_long', 'D_t']].tail())

# %% Volatility of Volatility
vov_window = 22
rolling_mean_sigma = df_fx['sigma_short'].rolling(vov_window).mean()
df_fx['VoV_short'] = df_fx['sigma_short'].rolling(vov_window).std()
df_fx['VoV_long'] = df_fx['sigma_long'].rolling(vov_window).std()

plt.figure(figsize=(12,5))
plt.plot(df_fx['date'], df_fx['VoV_short'], label='VoV Short')
plt.plot(df_fx['date'], df_fx['VoV_long'], label='VoV Long')
plt.xlabel('Date')
plt.ylabel('Volatility of Volatility')
plt.legend()
plt.show()

# %% Tail Dominance
df_fx['T_tail'] = df_fx['CVaR_1pct'] / df_fx['sigma_short']

# %%
