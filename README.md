# BTC-USDT-Analysis
001 - Attempt at trade-prediction bot using ML techniques.

Using a 1m BTC-USDT chart, the close price is used in a multiclass classification LSTM network which attempts to predict whether or not a trade should be made at each data point. 

The jupyter notebook file is the first implementation and exploration of this idea. The .py files are the refactored implementation, and are still a work in progress.

Later update: The project is still incomplete, I took some time away from it. If I were to revisit, my first priorities would be:
  1) Testing different approaches to solve the data imbalance problem through use of: upscaling/downscaling, changing trading hyperparameters
  2) Once the data imbalance is addressed, I could start properly tuning the model using some kind of random search or grid search.

## Trades
Trades are considered a "long trade", whereby the individual purchases at a certain price, and attempts to sell at a later, higher price. Upon purchasing an asset, the individual specifies the "take-profit" and "stop-loss" levels. The take-profit level defines the price at which the purchased asset will be automatically re-sold for positive profit. The stop-loss specifies at when the asset will be sold for negative profit. The take-profit and stop-loss levels can be completely specified by the closeing price, as well as the following parameters: profit, reward-to-risk ratio. The profit is the percentage profit in decimal form (e.g. 5% -> 0.05). The reward-to-risk ratio is the ratio between the potential profit, and the potential loss, as specified by the take-profit and stop-loss, respectively.

At each data point i, the program will create a label using one-hot encoding with the following scheme: [success, failure, indeterminate]. 

## Data Scaling
Data starts as a time series whereby each sequential data point is the absolute price level of the asset at that timestep. This will be reformatted to be the percentage difference since the previous point, with the very first data point asserted to be 0 (i.e. [10000, 11000, 12100] -> [0, 0.10, 0.10]). Once in this form, the data will be standardized to be within -1 and 1 for use in the LSTM network. 

This way, the data still represents meaningful information (the change in price from one point to the next), but now the data should roughly be a normal distribution centered around 0, and fully contained within the interval [-1,1]. Large changes in price will influence the weights of our network very strongly, and small fluctuations in price will be relatively ignored.
