import numpy as np
import pandas as pd
import indicator_functions

indicator_dict = {
    "sma": indicator_functions.get_sma,
    "std": indicator_functions.get_std
}


class DataSet:
    def __init__(self, df):
        """
        DataSet class contains the current working frame for the dataset, as well as
        all the methods used to process the data.

        :param df: Pandas DataFrame with shape (X, 1). The frame should contain only
        the closing price, as an absolute price level.
        """
        assert df.shape[1] == 1, "DataSet initialization failed: Incorrect shape. Expected (X, 1), got {}".format(
            df.shape)
        self.frame = df.copy()  # Copy the frame to not edit the raw input

    def get_indicators(self, indicators=None):
        """
        Create the specified indicators and add them to the frame. If they already exist within the
        dataset, then copy them instead.
        :param indicators: Indicators to be added to the frame. Formatted as a tuple with ("func", period)
        where the func is one of the available functions in the indicator dictionary.
        """
        if indicators is None:
            indicators = [("sma", 10), ("std", 10)]

        for item in indicators:
            func, period = item
            self.frame[func] = indicator_dict[func](self.frame["close"], period)

    def get_trades_long(self, profit=0.5 / 100, reward_risk=2.0, n_future=5,
                        append_frame=True, return_frame=False):
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
        :param append_frame: Append the result to self.frame
        :param return_frame: Return the calculated frame (self.frame["long"])
        :return: None. If return_frame is true, will return DataFrame whose entries
        are either [0,0,0], for stop-loss, indeterminate, or take-profit, respectively.
        """
        df = self.frame["close"]  # Make sure we only have the "close" column
        long_trades = pd.DataFrame(np.zeros(shape=df.shape))  # Initialize output frame with zeros

        for i, val in enumerate(df[-n_future:]):  # Iterate over the close frame while avoiding an IndexError
            # take_profit = val * (1 + profit) The calculation for the take-profit level
            # stop_loss = val * (1 - (profit / reward_risk)) The calculation for the stop-loss level

            for k in range(i, i + n_future):  # Iterate over the next n_future entries in 'close'
                if df.iloc[k] >= val * (1 + profit):  # If the value is above take-profit
                    long_trades.iloc[i] = (1, 0, 0)  # Return 1 for a successful trade
                    break  # The take-profit was triggered, so break the current loop
                elif df.iloc[k] <= val * (1 - (profit / reward_risk)):  # If the value is below stop-loss
                    long_trades.iloc[i] = (0, 1, 0)  # Return -1 for a failed trade
                    break  # The stop-loss was triggered, so break the current loop

        #  Conditional function parameters
        if append_frame:
            self.frame["long"] = long_trades  # Append the results to self.frame
        if return_frame:
            return long_trades  # Return the results frame

        return None

    def scale(self):
        pass

    def get_splits(self):
        # Function to return test-train split of the data set
        # sklearn.transform.fit(train) scale data according to training set
        #    then, sklearn.transform(test) will use the scaling factors from above
        #    to scale the test data
        pass

    def clear(self):
        # Reset the frame to be just the close value
        self.frame = self.frame["close"]
