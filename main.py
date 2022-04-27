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

    if True:  # Set to True to check labels distribution
        for item in data.labels.columns:
            print("{}: {}".format(item, np.unique(data.labels[item], return_counts=True)[1][1]))

    model = tf.keras.Sequential()
    model.add(LSTM(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    model.fit(data.x_train, data.y_train, epochs=5)


if __name__ == "__main__":
    main()
