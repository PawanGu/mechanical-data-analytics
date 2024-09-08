# Data Center ARIMA Time Series Forecasting

## Overview
This project implements an ARIMA (AutoRegressive Integrated Moving Average) model to forecast time series data related to energy, cooling, or any other system in data centers. The code generates synthetic data, fits the ARIMA model, and provides diagnostic tools to evaluate the model's performance.

## Code Explanation
The repository contains the following main files:
- **`main.py`:** This is the main script that generates the synthetic time series data, fits an ARIMA model, and outputs forecasts along with diagnostic plots. Below is a breakdown of each step in the code:

    1. **Synthetic Data Generation**: The `generate_synthetic_data()` function creates a time series with a linear trend and random noise to simulate real-world data.
    2. **ADF Test**: The `adf_test()` function performs the Augmented Dickey-Fuller test to determine whether the data is stationary.
    3. **ARIMA Model**: The `fit_arima_model()` function fits an ARIMA(1, 1, 1) model to the data and prints a summary of the model.
    4. **Forecasting**: The `plot_data_and_forecast()` function forecasts future values and plots both the original data and forecasted data.
    5. **Residual Diagnostics**: The `plot_residuals()` function creates residual plots, including a Q-Q plot and autocorrelation plot, to check the model’s performance.

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/PawanGu/mechanical-data-analytics.git
   cd mechanical-data-analytics
