import pandas as pd
import numpy as np


def dataClean():
    data = pd.read_csv("bank_transactions.csv", low_memory=False)
    data['TransactionTime'] = data['TransactionTime'].apply(
        lambda x: x / 10000)
    data['TransactionTime'] = np.floor(data['TransactionTime'])
    data['CustAccountBalance'] = np.floor(data['CustAccountBalance'])
    data['CustomerDOB'] = pd.to_datetime(data['CustomerDOB'])
    data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
    data = data[(data['CustomerDOB'].dt.year >= 1900) &
                (data['CustomerDOB'].dt.year <= 2020)]
    data['CustomerDOB'] = data['CustomerDOB'].dt.year
    data['TransactionDate'] = data['TransactionDate'].dt.week
    data = data.dropna()
    data = data[data['CustGender'] != 'T']
    data['CustGender'] = data['CustGender'].map({'F': 0, 'M': 1})
    data = data.drop(columns=['CustLocation', 'TransactionID', 'CustomerID'])
    data.to_csv('cleanData.csv', index=False)


dataClean()
