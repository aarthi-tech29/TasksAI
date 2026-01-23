# ===============================
# SALES FORECASTING SYSTEM
# ===============================

# 1. Import required libraries
import warnings
import pandas as pd # Store and handle data
import matplotlib.pyplot as plt # Plot sales trends
from statsmodels.tsa.arima.model import ARIMA # Time series forecasting
from statsmodels.tools.sm_exceptions import ValueWarning

# Suppress ARIMA / statsmodels warnings ONLY
warnings.simplefilter("ignore", ValueWarning)
warnings.simplefilter("ignore", UserWarning)

# 2 Create sample sales data
data = {
    "date": [
        "2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01",
        "2022-05-01", "2022-06-01", "2022-07-01", "2022-08-01",
        "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01",
        "2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"
    ],
    "sales": [
        120, 135, 150, 160,
        170, 180, 190, 200,
        210, 220, 240, 260,
        280, 300, 320, 340
    ]
}

df = pd.DataFrame(data)


# 3. Convert date column to datetime
df["date"] = pd.to_datetime(df["date"]) # Convert to datetime format


# 4. Set date as index (Time Series requirement)
df.set_index("date", inplace=True)
# Set date as index for time series analysis
# Index = Dates
# Values = Sales

# Explicitly set monthly frequency for time series
df = df.asfreq("MS")   # MS = Month Start


# 5. Plot sales trend
plt.figure()
plt.plot(df.index, df["sales"])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Trend Over Time")
plt.show()


# 6. Train ARIMA model
model = ARIMA(df["sales"], order=(2, 1, 2))
model_fit = model.fit()
# AR-Past values
# I	- Differencing
# MA - Past errors

# 7. Forecast future sales (next 6 months)
forecast = model_fit.forecast(steps=6)

# Predicts next 6 months
# Uses learned trend pattern to forecast future sales


# 8. Print forecasted sales
print("Future Sales Forecast:")
print(forecast)

# Sales Forecasting System predicts future sales using past sales data.
