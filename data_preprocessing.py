# data_preprocessing.py

import pandas as pd
import numpy as np

# Load datasets
transactions_01 = pd.read_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\Transactional_data_retail_01.csv')
transactions_02 = pd.read_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\Transactional_data_retail_02.csv')
customer_data = pd.read_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\CustomerDemographics.csv')
product_data = pd.read_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\ProductInfo.csv')

# Combine transactional data
transactions = pd.concat([transactions_01, transactions_02], ignore_index=True)

# Convert transaction_date to datetime
transactions['TransactionDate'] = pd.to_datetime(transactions['InvoiceDate'], dayfirst=True, errors='coerce')

# Merge with product data
transactions = transactions.merge(product_data, on='StockCode', how='left')

# Merge with customer data
transactions = transactions.merge(customer_data, on='Customer ID', how='left')

# Handle missing values
transactions.fillna({'Quantity': 0, 'price': 0}, inplace=True)
transactions.dropna(subset=['InvoiceDate'], inplace=True)

# Save the cleaned data
transactions.to_csv('C:\\Users\\lohit\\OneDrive\\Desktop\\1\\data\\cleaned_transactions.csv', index=False)
