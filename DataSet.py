import numpy as np
import pandas as pd
import indicator_functions
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

indicator_dict = {
    "sma": indicator_functions.get_sma,
    "std": indicator_functions.get_std
}


class DataSet:
    def __init__(self, df):
        """
        DataSet class contains the current working features for the dataset, as well as
        all the methods used to process the data.

        :param df: Pandas DataFrame with shape (X, 1). The features should contain only
        the closing price, as an absolute price level.
        """
        n_rows = df.shape[0]  # Calculate number of data points for code readability

        assert df.shape[1] == 1, "DataSet initialization failed: Incorrect shape. Expected (X, 1), got {}".format(
            df.shape)

        self.features = df.copy()  # Copy the features to not edit the raw input

        # Create the labels frame
        self.labels = pd.DataFrame(
            np.zeros(shape=(n_rows, 2)),
            columns=["win", "loss"])  # Initialize wins and losses as zeros
        self.labels["nil"] = 1.0  # Initialize nil trades as 1

        # Initialize training and testing tensors for later use
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def get_indicators(self, indicators=None):
        """
        Create the specified indicators and add them to the features. If they already exist within the
        dataset, then copy them instead.
        :param indicators: Indicators to be added to the features. Formatted as a tuple with ("func", period)
        where the func is one of the available functions in the indicator dictionary.
        """
        if indicators is None:
            indicators = [("sma", 10), ("std", 10)]

        max_period = 0

        for item in indicators:
            func, period = item
            if period > max_period:
                max_period = period
            self.features["{}-{}".format(func, period)] = indicator_dict[func](self.features["close"], period)

        self.features = self.features.iloc[max_period:]

    def get_labels(self, profit=0.5/100, reward_risk=2.0, n_future=5):
        """
        calc_trades method will calculate if a trade made at each
        data point would result in a take-profit or a stop-loss.
        For this first implementation, all trades are considered long
        trades, and will execute take-profit if the price goes up
        to the target, or a stop-loss if the price falls below.
        *** MAKE SURE DATA IS NOT RESCALED BEFORE CALLING THIS METHOD ***

        :param profit: The desired profit per trade, in percentage decimal form
        :param reward_risk: The reward-to-risk ratio for the trades
        :param n_future: The amount of time steps into the future to execute on
        :return: None. If return_frame is true, will return DataFrame whose entries
        are either [0,0,0], for stop-loss, indeterminate, or take-profit, respectively.
        """
        features = self.features["close"]  # Make sure we only have the "close" column
        labels = self.labels  # Pull the labels frame for encoding

        for i, val in enumerate(features[:-n_future]):  # Iterate over the close features while avoiding an IndexError
            # take_profit = val * (1 + profit) = val + val*profit The calculation for the take-profit level
            # stop_loss = val * (1 - (profit / reward_risk)) The calculation for the stop-loss level

            for k in range(i, i + n_future):  # Iterate over the next n_future entries in 'close'
                if features.iloc[k] >= val * (1 + profit):  # If the value is above take-profit
                    labels["win"].iloc[i] = 1  # Return 1 for a successful trade
                    labels["nil"].iloc[i] = 0  # Trade was executed, update nil column to 0
                    break  # The take-profit was triggered, so break the current loop
                elif features.iloc[k] <= val * (1 - (profit / reward_risk)):  # If the value is below stop-loss
                    labels["loss"].iloc[i] = 1  # Return -1 for a failed trade
                    labels["nil"].iloc[i] = 0  # Trade was executed, update nil column to 0
                    break  # The stop-loss was triggered, so break the current loop
        return None

    def preprocess(self, feature_window_size=5, test_split=0.5):
        """
        Scales all features data to be equal to the percent difference since the last data point. This allows negative
        values. Next, the data is re-normalized between -1 to 1 for use in the network. Lastly, the features and labels
        are reformatted into tensors of shape (batches, timesteps, features) for use in the LSTM layer. Note that for
        the labels tensor, the timesteps is always set to 1.
        :param test_split: Test-train ratio
        :param feature_window_size: The window size of the timesteps for the features tensor of shape (batches,
            timesteps, features)
        :return: None
        """

        assert isinstance(feature_window_size, int), "feature_window_size must be an integer. Found {}".format(
            str(type(feature_window_size)))

        # Scale the data using percent difference between timesteps, and then re-normalize from -1 to 1
        for item in self.features.columns:  # Iterate over all features
            df = self.features[item]  # Local variable for ease of writing

            for i in range(df.shape[0]-1, 0, -1):  # Iterate backwards over all elements, excluding the 0 position
                df.iloc[i] = 100*(df.iloc[i]-df.iloc[i-1]) / df.iloc[i-1]  # Percent difference from last time step

            df.iloc[0] = 0  # Initialize first value as zero

        # Reshape the features frame into a tensor of shape (batches, window_size, features)
        batches = self.features.shape[0] - feature_window_size
        feature_dim = self.features.shape[1]
        features_tensor = np.empty(shape=(batches, feature_window_size, feature_dim), dtype="float32")

        labels_trimmed = self.labels[feature_window_size:]
        for i in range(batches):
            features_tensor[i] = self.features.iloc[i: i+feature_window_size]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features_tensor, labels_trimmed, test_size=test_split, shuffle=False)

        scaler = preprocessing.StandardScaler(with_mean=False)

        for batch in self.x_train:
            scaler.partial_fit(batch)

        for tensor in [self.x_train, self.x_test]:
            for batch in tensor:
                batch = scaler.transform(batch)  # This only works because np arrays are mutable!

        return None


    def get_splits(self):
        # Function to return test-train split of the data set
        # sklearn.transform.fit(train) scale data according to training set
        #    then, sklearn.transform(test) will use the scaling factors from above
        #    to scale the test data
        pass

    def clear(self):
        # Reset the features to be just the close value
        self.features = self.features["close"]
