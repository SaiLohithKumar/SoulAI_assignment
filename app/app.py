import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set up the page
st.title("Demand Forecasting App")

# Load data
@st.cache
def load_data():
    transactions = pd.read_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\cleaned_transactions.csv')
    transactions['TransactionDate'] = pd.to_datetime(transactions['InvoiceDate'], dayfirst=True, errors='coerce')
    return transactions

transactions = load_data()

# Get top 10 products by quantity sold
top_10_selling = transactions.groupby('StockCode')['Quantity'].sum().nlargest(10)
top_products = top_10_selling.index.tolist()

# User input
st.sidebar.header('User Input Parameters')

stock_code = st.sidebar.selectbox('Select Stock Code', top_products)
weeks_ahead = st.sidebar.slider('Forecast Weeks Ahead', min_value=1, max_value=52, value=15)

# Forecasting function
def forecast_product(product_code, weeks_ahead=15):
    product_sales = transactions[transactions['StockCode'] == product_code]
    weekly_sales = product_sales.set_index('TransactionDate').resample('W')['Quantity'].sum()

    # Check if there are enough data points
    if len(weekly_sales) < 30:
        st.write("Not enough data points for forecasting.")
        return None, None

    # Split into train and test sets
    train_size = int(len(weekly_sales) * 0.8)
    train, test = weekly_sales.iloc[:train_size], weekly_sales.iloc[train_size:]

    # Fit ARIMA model
    model = ARIMA(train, order=(1,1,1))  # Adjust order based on previous analysis
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=weeks_ahead)

    # Evaluate
    predictions = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)

    # Plotting
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(train.index, train, label='Training Data')
    ax.plot(test.index, test, label='Test Data')
    ax.plot(test.index, predictions, label='Predictions', color='red')
    ax.plot(forecast.index, forecast, label='Forecast', color='green')
    ax.legend()
    ax.set_title(f'ARIMA Forecast for Product {product_code}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weekly Quantity Sold')

    # Error Histograms
    train_errors = train - model_fit.predict(start=train.index[0], end=train.index[-1])
    test_errors = test - predictions

    fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.hist(train_errors, bins=20, color='blue', alpha=0.7)
    ax1.set_title('Training Error Distribution')
    ax2.hist(test_errors, bins=20, color='orange', alpha=0.7)
    ax2.set_title('Test Error Distribution')

    return fig, fig2, forecast, rmse, mae

# Run forecast
if stock_code:
    with st.spinner('Forecasting...'):
        fig, fig2, forecast, rmse, mae = forecast_product(stock_code, weeks_ahead)

    if fig:
        st.pyplot(fig)
        st.write(f'RMSE: {rmse}')
        st.write(f'MAE: {mae}')

        st.write("### Error Histograms")
        st.pyplot(fig2)

        # Download forecast
        forecast_df = forecast.reset_index()
        forecast_df.columns = ['Date', 'Forecasted Quantity']
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f'forecast_{stock_code}.csv',
            mime='text/csv',
        )
