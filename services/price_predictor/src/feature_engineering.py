import talib
import pandas as pd
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the features.

    Args:
        df (pd.DataFrame): The input dataframe is expected to have the following columns:
        - 'open'
        - 'high'
        - 'low'
        - 'close'
        - 'volume'

    Returns:
        pd.DataFrame: The dataframe with the original features and the new technical indicators.

        The output dataframe will have the following columns:
        - 'open'
        - 'high'
        - 'low'
        - 'close'
        - 'volume'
        - 'SMA_7'
        - 'SMA_14'
        - 'SMA_28'
        - 'EMA_7'
        - 'EMA_14'
        - 'EMA_28'
        - 'RSI_14'
        - 'MACD'
        - 'MACD_Signal'
        - 'BB_Upper'
        - 'BB_Middle'
        - 'BB_Lower'
        - 'Stoch_K'
        - 'Stoch_D'
        - 'OBV'
        - 'ATR'
        - 'CCI'
        - 'CMF'
    """
    # 1. Add Simple Moving Average (SMA) with a
    # - 7-period window
    # - 14-period window
    # - 28-period window
    df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
    df['SMA_14'] = talib.SMA(df['close'], timeperiod=14)
    df['SMA_28'] = talib.SMA(df['close'], timeperiod=28)
    df['SMA_70'] = talib.SMA(df['close'], timeperiod=70)
    
    # 2. Add Exponential Moving Average (EMA) with a
    # - 7-period window
    # - 14-period window
    # - 28-period window
    df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
    df['EMA_14'] = talib.EMA(df['close'], timeperiod=14)
    df['EMA_28'] = talib.EMA(df['close'], timeperiod=28)
    df['EMA_70'] = talib.EMA(df['close'], timeperiod=70)

    # 3. Relative Strength Index (RSI)
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)

    # 4. Moving Average Convergence Divergence (MACD)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal

    # 5. Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # 6. Stochastic Oscillator
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['Stoch_K'] = slowk
    df['Stoch_D'] = slowd

    # 7. On-Balance Volume (OBV)
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    # 8. Average True Range (ATR)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # 9. Commodity Channel Index (CCI)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    # 10. Chaikin Money Flow (CMF)
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    return df