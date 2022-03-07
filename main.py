import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from DataSet import DataSet

from tensorflow.keras.layers import Dense, LSTM, Bidirectional
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
    data.get_trades_long()


if __name__ == "__main__":
    main()
