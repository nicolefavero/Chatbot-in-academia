import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# 1. Create a DataFrame from sample data
data = {
    'date': [
        '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05',
        '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01', '2024-02',
        '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08'
    ],
    'diesel': [
        87.69, 90.04, 89.26, 85.67, 86.00, 93.25, 92.49, 87.46, 86.15, 86.51, 85.39, 83.23, 
        86.83, 78.84, 78.32, 79.66, 81.55, 80.55, 79.14, 77.38, 74.76, 73.23, 71.76, 71.87
    ],
    'electric': [
        12.31,  9.95, 10.74, 14.33,  6.75,  7.50, 11.40, 11.93, 12.79, 13.05, 13.70, 15.61, 
        13.17, 19.09, 18.75, 17.83, 16.26, 17.00, 19.14, 21.29, 23.92, 26.99, 27.54, 27.50
    ]
}
df = pd.DataFrame(data)

# Convert 'date' to datetime and set as index
df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
df.set_index('date', inplace=True)

# 2. Fit Holt’s Exponential Smoothing model for each series
forecast_horizon = 41

diesel_model = ExponentialSmoothing(
    df['diesel'], 
    trend='add',  
    seasonal=None, 
    initialization_method='estimated'
).fit()

diesel_forecast = diesel_model.forecast(forecast_horizon)

electric_model = ExponentialSmoothing(
    df['electric'], 
    trend='add', 
    seasonal=None, 
    initialization_method='estimated'
).fit()

electric_forecast = electric_model.forecast(forecast_horizon)

# 3. Visualize everything on one single plot
plt.figure(figsize=(10, 5))

# Historical Diesel
plt.plot(df.index, df['diesel'], label='Historical Diesel')

# Forecast Diesel (line, no markers)
plt.plot(diesel_forecast.index, diesel_forecast, label='Forecast Diesel')

# Annotate the last forecasted Diesel value on the plot
last_diesel_date = diesel_forecast.index[-1]
last_diesel_value = diesel_forecast.iloc[-1]
plt.text(last_diesel_date, last_diesel_value, 
         f"{last_diesel_value:.2f}",  # display the numeric value
         va='bottom', ha='left', 
         fontsize=9)

# Historical Electric
plt.plot(df.index, df['electric'], label='Historical Electric')

# Forecast Electric (line, no markers)
plt.plot(electric_forecast.index, electric_forecast, label='Forecast Electric')

# Annotate the last forecasted Electric value on the plot
last_electric_date = electric_forecast.index[-1]
last_electric_value = electric_forecast.iloc[-1]
plt.text(last_electric_date, last_electric_value, 
         f"{last_electric_value:.2f}",
         va='bottom', ha='left',
         fontsize=9)

plt.title("Diesel & Electric Price Forecast (Holt’s Method)")
plt.legend()
plt.show()

# 4. Print the forecasted values
print("==== Diesel Forecast ====")
print(diesel_forecast)

print("\n==== Electric Forecast ====")
print(electric_forecast)
