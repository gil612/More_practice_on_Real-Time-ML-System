import talib
import pandas as pd

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
    # Remove duplicate calculations that exist in other specialized functions
    
    # Keep unique indicators not present in other functions
    df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

    return df

def add_technical_indicators_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the features.
    """
    df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    
    
    
    
    return df

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:  
    """
    Add momentum indicators to the features.
    """
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=14)
    df['APO'] = talib.APO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    aroon_up, aroon_down = talib.AROON(df['high'], df['low'], timeperiod=14)
    df['AROON_UP'] = aroon_up
    df['AROON_DOWN'] = aroon_down
    df['AROONOSC'] = talib.AROONOSC(df['high'], df['low'], timeperiod=14)
    df['BOP'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
    df['CMO'] = talib.CMO(df['close'], timeperiod=14)
    df['DX'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Fix MACD assignments
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_SIGNAL'] = macd_signal
    df['MACD_HIST'] = macd_hist
    
    # Fix MACDEXT parameters - removed matype
    macdext, macdext_signal, macdext_hist = talib.MACDEXT(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACDEXT'] = macdext
    df['MACDEXT_SIGNAL'] = macdext_signal
    df['MACDEXT_HIST'] = macdext_hist
    
    macdfix, macdfix_signal, macdfix_hist = talib.MACDFIX(df['close'], signalperiod=9)
    df['MACDFIX'] = macdfix
    df['MACDFIX_SIGNAL'] = macdfix_signal
    df['MACDFIX_HIST'] = macdfix_hist
    
    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(df['high'], df['low'], timeperiod=14)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(df['high'], df['low'], timeperiod=14)
    df['PPO'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)
    df['ROCP'] = talib.ROCP(df['close'], timeperiod=10)
    df['ROCR'] = talib.ROCR(df['close'], timeperiod=10)
    df['ROCR100'] = talib.ROCR100(df['close'], timeperiod=10)
    
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], 
                              fastk_period=14, slowk_period=3, 
                              slowk_matype=0, slowd_period=3, 
                              slowd_matype=0)
    df['STOCH_K'] = slowk
    df['STOCH_D'] = slowd
    
    fastk, fastd = talib.STOCHF(df['high'], df['low'], df['close'], 
                               fastk_period=14, fastd_period=3, 
                               fastd_matype=0)
    df['STOCHF_K'] = fastk
    df['STOCHF_D'] = fastd
    
    stochrsi_k, stochrsi_d = talib.STOCHRSI(df['close'], timeperiod=14, 
                                           fastk_period=5, fastd_period=3, 
                                           fastd_matype=0)
    df['STOCHRSI_K'] = stochrsi_k
    df['STOCHRSI_D'] = stochrsi_d
    
    df['TRIX'] = talib.TRIX(df['close'], timeperiod=14)
    df['ULTOSC'] = talib.ULTOSC(df['high'], df['low'], df['close'], 
                               timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    
    return df

def add_technical_indicators_volatility(df: pd.DataFrame) -> pd.DataFrame:
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

        which are:
        ATR                  Average True Range
        NATR                 Normalized Average True Range
        TRANGE               True Range
    """
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])
    return df

def add_technical_indicators_overlap_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the features.
    """
    # Fix BBANDS assignment
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2)
    df['BB_UPPER'] = upper
    df['BB_MIDDLE'] = middle
    df['BB_LOWER'] = lower
    
    df['DEMA'] = talib.DEMA(df['close'], timeperiod=30)
    df['EMA'] = talib.EMA(df['close'], timeperiod=30)
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close'])
    df['KAMA'] = talib.KAMA(df['close'], timeperiod=30)
    df['MA'] = talib.MA(df['close'], timeperiod=30)
    
    mama, fama = talib.MAMA(df['close'])
    df['MAMA'] = mama
    df['FAMA'] = fama
    
    df['MAVP'] = talib.MAVP(df['close'], df['volume'], minperiod=2, maxperiod=30)
    df['MIDPOINT'] = talib.MIDPOINT(df['close'], timeperiod=14)
    df['MIDPRICE'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=14)
    
    # Fix SAR parameters - using correct parameter names
    df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
    
    # Remove SAREXT for now as it's causing issues and might not be essential
    # We can add it back later if needed with correct parameters
    
    df['SMA'] = talib.SMA(df['close'], timeperiod=30)
    df['T3'] = talib.T3(df['close'], timeperiod=5, vfactor=0)
    df['TEMA'] = talib.TEMA(df['close'], timeperiod=30)
    df['TRIMA'] = talib.TRIMA(df['close'], timeperiod=30)
    df['WMA'] = talib.WMA(df['close'], timeperiod=30)
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
    
    # Remove any remaining columns with NaN values
    df = df.dropna(axis=1, how='any')
    
    return df