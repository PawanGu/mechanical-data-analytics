import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import os

# Ensure the output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

def generate_synthetic_data():
    # Generate synthetic time series data
    np.random.seed(42)
    n = 200
    time = np.arange(n)
    data = 20 + 0.5 * time + np.random.normal(scale=5, size=n)  # Linear trend with noise
    return time, data

def adf_test(data):
    # Check for stationarity using Augmented Dickey-Fuller (ADF) Test
    adf_test = adfuller(data)
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")
    if adf_test[1] > 0.05:
        print("The data is non-stationary, applying differencing to make it stationary.")
    else:
        print("The data is stationary.")

def fit_arima_model(data):
    # Fit the ARIMA model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def plot_data_and_forecast(time, data, model_fit, steps=20):
    # Forecasting future values
    n_forecast = steps
    forecast = model_fit.forecast(steps=n_forecast)
    forecast_index = np.arange(len(time), len(time) + n_forecast)

    # Plot original data and forecast
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, label="Original Data")
    plt.plot(forecast_index, forecast, label="Forecasted Data", color='red')
    plt.title("ARIMA Model Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output/arima_forecast.png")
    plt.show()

def plot_residuals(model_fit):
    # Get residuals from the model
    residuals = model_fit.resid

    # Plot residuals over time
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(residuals, label='Residuals')
    plt.title("Residuals from ARIMA Model")
    plt.xlabel("Time")
    plt.ylabel("Residuals")
    plt.legend()

    # Plot histogram of residuals
    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/arima_residuals.png")
    plt.show()

    # Q-Q Plot for normality check
    sm.qqplot(residuals, line='s')
    plt.title("Q-Q Plot of Residuals")
    plt.savefig("output/arima_qqplot.png")
    plt.show()

    # Autocorrelation plot of residuals
    sm.graphics.tsa.plot_acf(residuals, lags=30)
    plt.title("Autocorrelation of Residuals")
    plt.savefig("output/arima_acf.png")
    plt.show()

def main():
    # Step 1: Generate synthetic time series data
    time, data = generate_synthetic_data()

    # Step 2: Plot the original data
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, label="Original Data")
    plt.title("Synthetic Time Series Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("output/original_data.png")
    plt.show()

    # Step 3: Perform the ADF test for stationarity
    adf_test(data)

    # Step 4: Fit the ARIMA model
    model_fit = fit_arima_model(data)

    # Step 5: Plot original data and forecasted data
    plot_data_and_forecast(time, data, model_fit)

    # Step 6: Plot diagnostic residual plots
    plot_residuals(model_fit)

if __name__ == "__main__":
    main()
