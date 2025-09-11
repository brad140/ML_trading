import pandas as pd
import ta

def create_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    data['return_1d'] = data['Close'].pct_change(1)
    data['return_5d'] = data['Close'].pct_change(5)
    data['return_21d'] = data['Close'].pct_change(21)

    data['volatility_10d'] = data['return_1d'].rolling(window=10).std()
    data['volatility_63d'] = data['return_1d'].rolling(window=63).std()

    data['trend_strength_21d'] = data['Close'] / data['Close'].rolling(window=21).mean()
    data['trend_strength_63d'] = data['Close'] / data['Close'].rolling(window=63).mean()

    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['macd_diff'] = ta.trend.MACD(data['Close']).macd_diff()
    data['rsi_change_5d'] = data['rsi'].diff(5)

    data['day_of_week'] = data.index.dayofweek
    data['week_of_year'] = data.index.isocalendar().week.astype(int)
    data['month'] = data.index.month

    return data
