import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from DataSet import DataSet

from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def main():
    raw = pd.read_csv("src/binance_BTCUSDT_minute.csv",
                      low_memory=False,
                      sep=",",
                      header=1,
                      nrows=10000,
                      usecols=["close"])

    data = DataSet(raw)
    data.get_indicators(indicators = [("sma",15)])
    data.get_labels(profit=0.05/100)  # Calculating labels with a tiny profit to balance out the data
    data.preprocess()

    #print(data.features_tensor)
    #print(data.labels_tensor)



    if False: # Set to True to check labels distribution
        for item in data.labels.columns:
            print(np.unique(data.labels[item], return_counts=True))


if __name__ == "__main__":
    main()
