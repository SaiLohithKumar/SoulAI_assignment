import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
transactions = pd.read_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\cleaned_transactions.csv')
transactions['TransactionDate'] = pd.to_datetime(transactions['InvoiceDate'], dayfirst=True, errors='coerce')

# Function to forecast for a given product
def forecast_product(product_code, weeks_ahead=15):
    product_sales = transactions[transactions['StockCode'] == product_code]
    weekly_sales = product_sales.set_index('Transaction Date').resample('W')['Quantity'].sum()

    # Check if there are enough data points
    if len(weekly_sales) < 30:
        print(f"Not enough data points for product {product_code}.")
        return None

    # Split into train and test sets
    train_size = int(len(weekly_sales) * 0.8)
    train, test = weekly_sales.iloc[:train_size], weekly_sales.iloc[train_size:]

    # Fit ARIMA model (parameters can be adjusted based on ACF/PACF)
    model = ARIMA(train, order=(1,1,1))  # Placeholder order, adjust based on ACF/PACF
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=weeks_ahead)

    # Evaluate on test set
    predictions = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

    # Generate future dates for the forecast
    last_date = weekly_sales.index[-1]
    forecast_dates = pd.date_range(last_date, periods=weeks_ahead+1, freq='W')[1:]

    # Plotting
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(test.index, predictions, label='Predictions', color='red')
    plt.plot(forecast_dates, forecast, label='Forecast', color='green')
    plt.legend()
    plt.title(f'ARIMA Forecast for Product {product_code}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Quantity Sold')
    plt.show()

    # Return forecast
    return pd.Series(forecast, index=forecast_dates)

# Example usage
if __name__ == "__main__":
    forecast = forecast_product("85123A")  # Replace with actual product code from top 10 list
