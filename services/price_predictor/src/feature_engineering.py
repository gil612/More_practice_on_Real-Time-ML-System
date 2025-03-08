import pandas as pd
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator,
    CCIIndicator, PSARIndicator, TRIXIndicator
)
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator,
    AwesomeOscillatorIndicator, UltimateOscillator
)
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel
)
from ta.volume import (
    AccDistIndexIndicator, ChaikinMoneyFlowIndicator,
    MFIIndicator, OnBalanceVolumeIndicator,
    VolumeWeightedAveragePrice
)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe should have a timestamp column
        (either 'timestamp_ms' or 'timestamp')

    Returns:
        pd.DataFrame: The dataframe with the temporal features added.
    """
    # Check which timestamp column exists
    if 'timestamp_ms' in df.columns:
        timestamp_col = 'timestamp_ms'
    elif 'timestamp' in df.columns:
        timestamp_col = 'timestamp'
    else:
        # If no timestamp column found, return dataframe unchanged
        return df

    # Convert timestamp to datetime and extract features
    df['hour'] = pd.to_datetime(df[timestamp_col], unit='ms').dt.hour
    df['day'] = pd.to_datetime(df[timestamp_col], unit='ms').dt.day
    df['month'] = pd.to_datetime(df[timestamp_col], unit='ms').dt.month
    df['weekday'] = pd.to_datetime(df[timestamp_col], unit='ms').dt.weekday
    
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add core technical indicators to the features.
    """
    # RSI
    rsi = RSIIndicator(close=df['close'], window=14)
    df['RSI_14'] = rsi.rsi()
    
    # CCI
    cci = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['CCI'] = cci.cci()
    
    # Chaikin Money Flow
    cmf = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
    df['CMF'] = cmf.chaikin_money_flow()

    return df

def add_technical_indicators_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based technical indicators to the features.
    """
    # Accumulation/Distribution Index
    adi = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    df['AD'] = adi.acc_dist_index()
    
    # Chaikin Money Flow
    cmf = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
    df['CMF'] = cmf.chaikin_money_flow()
    
    # On Balance Volume
    obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['OBV'] = obv.on_balance_volume()
    
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:  
    """
    Add momentum indicators to the features.
    """
    # ADX
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx.adx()
    
    # MACD
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    df['MACD_HIST'] = macd.macd_diff()
    
    # RSI
    rsi = RSIIndicator(close=df['close'], window=14)
    df['RSI'] = rsi.rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['STOCH_K'] = stoch.stoch()
    df['STOCH_D'] = stoch.stoch_signal()
    
    # Williams %R
    williams = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14)
    df['WILLR'] = williams.williams_r()
    
    return df

def add_technical_indicators_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to the features.
    """
    # Average True Range
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_UPPER'] = bb.bollinger_hband()
    df['BB_MIDDLE'] = bb.bollinger_mavg()
    df['BB_LOWER'] = bb.bollinger_lband()
    
    # Keltner Channel
    kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=10)
    df['KC_UPPER'] = kc.keltner_channel_hband()
    df['KC_MIDDLE'] = kc.keltner_channel_mband()
    df['KC_LOWER'] = kc.keltner_channel_lband()
    
    return df

def add_technical_indicators_overlap_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overlap study indicators to the features.
    """
    # SMA
    sma = SMAIndicator(close=df['close'], window=30)
    df['SMA'] = sma.sma_indicator()
    
    # EMA
    ema = EMAIndicator(close=df['close'], window=30)
    df['EMA'] = ema.ema_indicator()
    
    # Ichimoku
    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['ICHIMOKU_A'] = ichimoku.ichimoku_a()
    df['ICHIMOKU_B'] = ichimoku.ichimoku_b()
    
    # TRIX
    trix = TRIXIndicator(close=df['close'], window=30)
    df['TRIX'] = trix.trix()
    
    return df

def add_price_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom price-based features with focus on recent price action
    """
    # Short-term price momentum
    for period in [1, 2, 3, 5, 8, 13]:
        df[f'return_{period}'] = df['close'].pct_change(period)
        df[f'volume_change_{period}'] = df['volume'].pct_change(period)
    
    # Price volatility features
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['close'].rolling(window).std() / df['close'].rolling(window).mean()
        df[f'volume_volatility_{window}'] = df['volume'].rolling(window).std() / df['volume'].rolling(window).mean()
        
        # True Range based volatility
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df[f'tr_volatility_{window}'] = tr.rolling(window).mean() / df['close']
    
    # Price levels and ranges
    df['price_range'] = (df['high'] - df['low']) / df['open']
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Volume profile
    df['volume_price_ratio'] = df['volume'] / df['close']
    for window in [5, 10, 20]:
        df[f'vol_price_ratio_ma_{window}'] = df['volume_price_ratio'].rolling(window).mean()
        
    # Trend strength indicators
    for window in [10, 20, 30]:
        sma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        df[f'trend_strength_{window}'] = (df['close'] - sma) / std
        
    # Gap analysis
    df['gap_up'] = df['low'] > df['high'].shift(1)
    df['gap_down'] = df['high'] < df['low'].shift(1)
    
    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features with focus on predictive power
    """
    # Add price-based features first
    df = add_price_based_features(df)
    
    # Add only the most important technical indicators
    df = add_technical_indicators_volatility(df)  # ATR and volatility
    df = add_momentum_indicators(df)              # RSI, MACD, etc.
    
    # Add temporal features last
    df = add_temporal_features(df)
    
    # Handle missing values more gracefully
    df = df.fillna(method='ffill')  # Forward fill
    df = df.fillna(method='bfill')  # Back fill any remaining NaNs
    
    return df