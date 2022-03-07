# BTC-USDT-Analysis
001 - Attempt at trade-prediction bot using ML techniques.

Using a 1m BTC-USDT chart, the close price is used in a multiclass classification LSTM network which attempts to predict whether or not a trade should be made at each data point. 

## Trades
Trades are considered a "long trade", whereby the individual purchases at a certain price, and attempts to sell at a later, higher price. Upon purchasing an asset, the individual specifies the "take-profit" and "stop-loss" levels. The take-profit level defines the price at which the purchased asset will be automatically re-sold for positive profit. The stop-loss specifies at when the asset will be sold for negative profit. The take-profit and stop-loss levels can be completely specified by the closeing price, as well as the following parameters: profit, reward-to-risk ratio. The profit is the percentage profit in decimal form (e.g. 5% -> 0.05). The reward-to-risk ratio is the ratio between the potential profit, and the potential loss, as specified by the take-profit and stop-loss, respectively.
