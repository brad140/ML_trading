import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Comprehensive feature engineering framework with 200+ technical indicators"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set with 200+ indicators"""
        df = data.copy()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Technical indicators using ta library
        df = self._add_technical_indicators(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Trend features
        df = self._add_trend_features(df)
        
        # Cycle features
        df = self._add_cycle_features(df)
        
        # Pattern recognition features
        df = self._add_pattern_features(df)
        
        # Cross-asset features (if multiple symbols)
        df = self._add_cross_asset_features(df)
        
        # Clean and validate features
        df = self._clean_features(df)
        
        return df
    
    def _add_price_features(self, df):
        """Add price-based features"""
        # Returns
        for period in [1, 2, 3, 5, 10, 20, 50, 100]:
            df[f'return_{period}d'] = df['Close'].pct_change(period)
            df[f'log_return_{period}d'] = np.log(df['Close'] / df['Close'].shift(period))
        
        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['high_close_ratio'] = df['High'] / df['Close']
        df['low_close_ratio'] = df['Low'] / df['Close']
        
        # Price position within daily range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Gap features
        df['gap_up'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_down'] = (df['Close'].shift(1) - df['Open']) / df['Close'].shift(1)
        
        return df
    
    def _add_volume_features(self, df):
        """Add volume-based features"""
        # Volume ratios
        for period in [5, 10, 20, 50]:
            df[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']
        
        # Volume-price features
        df['volume_price_trend'] = df['Volume'] * df['Close'].pct_change()
        df['volume_weighted_price'] = (df['Volume'] * df['Close']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        
        # On-Balance Volume
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # Volume Rate of Change
        for period in [5, 10, 20]:
            df[f'volume_roc_{period}'] = df['Volume'].pct_change(period)
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add comprehensive technical indicators using ta library"""
        try:
            # Add all ta features
            df = add_all_ta_features(df, open="Open", high="High", low="Low", 
                                   close="Close", volume="Volume", fillna=True)
        except Exception as e:
            print(f"Warning: Could not add all ta features: {e}")
            # Add individual indicators manually
            df = self._add_individual_indicators(df)
        
        # Add additional custom indicators to reach 200+
        df = self._add_custom_indicators(df)
        
        return df
    
    def _add_individual_indicators(self, df):
        """Add individual technical indicators"""
        # Momentum indicators
        df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['Close'], window=21).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Trend indicators
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        return df
    
    def _add_statistical_features(self, df):
        """Add statistical features"""
        # Rolling statistics
        for period in [5, 10, 20, 50]:
            df[f'std_{period}'] = df['Close'].rolling(period).std()
            df[f'skew_{period}'] = df['Close'].rolling(period).skew()
            df[f'kurt_{period}'] = df['Close'].rolling(period).apply(lambda x: x.kurtosis())
            df[f'quantile_25_{period}'] = df['Close'].rolling(period).quantile(0.25)
            df[f'quantile_75_{period}'] = df['Close'].rolling(period).quantile(0.75)
        
        # Z-scores
        for period in [20, 50]:
            mean = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['Close'] - mean) / std
        
        return df
    
    def _add_time_features(self, df):
        """Add time-based features"""
        df['day_of_week'] = df.index.dayofweek
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_microstructure_features(self, df):
        """Add market microstructure features"""
        # Bid-ask spread proxy
        df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
        
        # Price impact
        df['price_impact'] = df['Close'].pct_change() / df['Volume'].pct_change()
        
        # Tick direction
        df['tick_direction'] = np.where(df['Close'] > df['Open'], 1, -1)
        
        return df
    
    def _add_volatility_features(self, df):
        """Add volatility features"""
        # Historical volatility
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt(1/(4*np.log(2)) * np.log(df['High']/df['Low'])**2)
        
        # Garman-Klass volatility
        df['gk_vol'] = 0.5 * np.log(df['High']/df['Low'])**2 - (2*np.log(2)-1) * np.log(df['Close']/df['Open'])**2
        
        return df
    
    def _add_momentum_features(self, df):
        """Add momentum features"""
        # Rate of Change
        for period in [5, 10, 20, 50]:
            df[f'roc_{period}'] = df['Close'].pct_change(period)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        return df
    
    def _add_trend_features(self, df):
        """Add trend features"""
        # Trend strength
        for period in [20, 50, 100]:
            sma = df['Close'].rolling(period).mean()
            df[f'trend_strength_{period}'] = df['Close'] / sma
            df[f'trend_direction_{period}'] = np.where(df['Close'] > sma, 1, -1)
        
        # ADX
        try:
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
        except:
            pass
        
        return df
    
    def _add_cycle_features(self, df):
        """Add cycle features"""
        # Detrended price
        for period in [20, 50]:
            sma = df['Close'].rolling(period).mean()
            df[f'detrended_price_{period}'] = df['Close'] - sma
        
        return df
    
    def _add_pattern_features(self, df):
        """Add pattern recognition features"""
        # Doji pattern
        df['doji'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
        
        # Hammer pattern
        body = np.abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        df['hammer'] = (lower_shadow > 2 * body) & (upper_shadow < body)
        
        return df
    
    def _add_custom_indicators(self, df):
        """Add custom technical indicators to reach 200+ features"""
        
        # Additional RSI variations
        for period in [7, 9, 14, 21, 28]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
            df[f'rsi_{period}_change'] = df[f'rsi_{period}'].diff()
            df[f'rsi_{period}_ma'] = df[f'rsi_{period}'].rolling(5).mean()
        
        # Additional MACD variations
        for fast in [8, 12, 16]:
            for slow in [21, 26, 30]:
                macd = ta.trend.MACD(df['Close'], window_fast=fast, window_slow=slow)
                df[f'macd_{fast}_{slow}'] = macd.macd()
                df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
                df[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()
        
        # Additional Moving Averages
        for period in [3, 7, 13, 17, 21, 34, 55, 89, 144, 233]:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(df['Close'], window=period).sma_indicator()
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['Close'], window=period).ema_indicator()
            df[f'wma_{period}'] = df['Close'].rolling(period).apply(lambda x: np.average(x, weights=range(1, period+1)))
            
            # Price position relative to MA
            df[f'price_above_sma_{period}'] = (df['Close'] > df[f'sma_{period}']).astype(int)
            df[f'price_above_ema_{period}'] = (df['Close'] > df[f'ema_{period}']).astype(int)
            
            # MA slope
            df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(5)
            df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(5)
        
        # Bollinger Bands variations
        for period in [10, 15, 20, 25, 30]:
            for std in [1.5, 2.0, 2.5]:
                bb = ta.volatility.BollingerBands(df['Close'], window=period, window_dev=std)
                df[f'bb_upper_{period}_{std}'] = bb.bollinger_hband()
                df[f'bb_lower_{period}_{std}'] = bb.bollinger_lband()
                df[f'bb_middle_{period}_{std}'] = bb.bollinger_mavg()
                df[f'bb_width_{period}_{std}'] = (df[f'bb_upper_{period}_{std}'] - df[f'bb_lower_{period}_{std}']) / df[f'bb_middle_{period}_{std}']
                df[f'bb_position_{period}_{std}'] = (df['Close'] - df[f'bb_lower_{period}_{std}']) / (df[f'bb_upper_{period}_{std}'] - df[f'bb_lower_{period}_{std}'])
        
        # Stochastic variations
        for k_period in [5, 9, 14, 21]:
            for d_period in [3, 5, 9]:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                       window=k_period, smooth_window=d_period)
                df[f'stoch_k_{k_period}_{d_period}'] = stoch.stoch()
                df[f'stoch_d_{k_period}_{d_period}'] = stoch.stoch_signal()
        
        # Williams %R variations
        for period in [7, 14, 21, 28]:
            df[f'williams_r_{period}'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=period).williams_r()
        
        # Commodity Channel Index
        for period in [10, 14, 20, 30]:
            df[f'cci_{period}'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=period).cci()
        
        # Average True Range
        for period in [7, 14, 21, 30]:
            df[f'atr_{period}'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=period).average_true_range()
            df[f'atr_{period}_ratio'] = df[f'atr_{period}'] / df['Close']
        
        # Donchian Channel
        for period in [10, 20, 30, 50]:
            df[f'donchian_upper_{period}'] = df['High'].rolling(period).max()
            df[f'donchian_lower_{period}'] = df['Low'].rolling(period).min()
            df[f'donchian_middle_{period}'] = (df[f'donchian_upper_{period}'] + df[f'donchian_lower_{period}']) / 2
            df[f'donchian_position_{period}'] = (df['Close'] - df[f'donchian_lower_{period}']) / (df[f'donchian_upper_{period}'] - df[f'donchian_lower_{period}'])
        
        # Keltner Channel
        for period in [10, 20, 30]:
            for multiplier in [1.5, 2.0, 2.5]:
                kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=period, window_atr=period, fillna=True, multiplier=multiplier)
                df[f'keltner_upper_{period}_{multiplier}'] = kc.keltner_channel_hband()
                df[f'keltner_lower_{period}_{multiplier}'] = kc.keltner_channel_lband()
                df[f'keltner_middle_{period}_{multiplier}'] = kc.keltner_channel_mband()
        
        # Ichimoku Cloud
        try:
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['ichimoku_base'] = ichimoku.ichimoku_base_line()
            df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
            df['ichimoku_span'] = df['ichimoku_a'] - df['ichimoku_b']
        except:
            pass
        
        # Parabolic SAR
        for step in [0.02, 0.04, 0.06]:
            for maximum in [0.1, 0.2, 0.3]:
                try:
                    psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'], step=step, max_step=maximum)
                    df[f'psar_{int(step*100)}_{int(maximum*100)}'] = psar.psar()
                except:
                    pass
        
        # Additional volume indicators
        for period in [5, 10, 20, 30]:
            # Volume SMA
            df[f'volume_sma_{period}'] = df['Volume'].rolling(period).mean()
            
            # Volume EMA
            df[f'volume_ema_{period}'] = df['Volume'].ewm(span=period).mean()
            
            # Volume Rate of Change
            df[f'volume_roc_{period}'] = df['Volume'].pct_change(period)
            
            # Volume Price Trend
            df[f'vpt_{period}'] = (df['Volume'] * df['Close'].pct_change()).rolling(period).sum()
        
        # Money Flow Index
        for period in [7, 14, 21, 30]:
            df[f'mfi_{period}'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=period).money_flow_index()
        
        # Force Index
        for period in [1, 5, 10, 20]:
            df[f'force_index_{period}'] = ta.volume.ForceIndexIndicator(df['Close'], df['Volume'], window=period).force_index()
        
        # Ease of Movement
        for period in [5, 10, 20]:
            df[f'eom_{period}'] = ta.volume.EaseOfMovementIndicator(df['High'], df['Low'], df['Volume'], window=period).ease_of_movement()
        
        # Volume Weighted Average Price
        for period in [5, 10, 20, 30]:
            df[f'vwap_{period}'] = (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()
        
        # Additional momentum indicators
        for period in [5, 10, 20, 30]:
            # Rate of Change
            df[f'roc_{period}'] = df['Close'].pct_change(period)
            
            # Momentum
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            
            # Price Rate of Change
            df[f'proc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
        
        # Ultimate Oscillator
        for short in [7, 14]:
            for medium in [14, 21]:
                for long in [28, 42]:
                    try:
                        uo = ta.momentum.UltimateOscillator(df['High'], df['Low'], df['Close'], 
                                                          window1=short, window2=medium, window3=long)
                        df[f'ultimate_osc_{short}_{medium}_{long}'] = uo.ultimate_oscillator()
                    except:
                        pass
        
        # Awesome Oscillator
        df['awesome_osc'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
        
        # Percentage Price Oscillator
        for fast in [6, 12, 18]:
            for slow in [26, 35, 50]:
                try:
                    ppo = ta.momentum.PercentagePriceOscillator(df['Close'], window_slow=slow, window_fast=fast)
                    df[f'ppo_{fast}_{slow}'] = ppo.ppo()
                    df[f'ppo_signal_{fast}_{slow}'] = ppo.ppo_signal()
                    df[f'ppo_hist_{fast}_{slow}'] = ppo.ppo_hist()
                except:
                    pass
        
        # Additional volatility indicators
        for period in [5, 10, 20, 30]:
            # Historical Volatility
            df[f'hist_vol_{period}'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
            
            # Average Deviation
            df[f'avg_dev_{period}'] = df['Close'].rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            # Coefficient of Variation
            df[f'cv_{period}'] = df['Close'].rolling(period).std() / df['Close'].rolling(period).mean()
        
        # Additional pattern recognition
        # Doji variations
        df['doji_strong'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.05
        df['doji_weak'] = np.abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.15
        
        # Hammer variations
        body = np.abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_shadow = np.minimum(df['Open'], df['Close']) - df['Low']
        
        df['hammer'] = (lower_shadow > 2 * body) & (upper_shadow < body)
        df['hanging_man'] = (lower_shadow > 2 * body) & (upper_shadow < body) & (df['Close'] < df['Open'])
        df['shooting_star'] = (upper_shadow > 2 * body) & (lower_shadow < body)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & \
                                 (df['Open'] < df['Close'].shift(1)) & (df['Close'] > df['Open'].shift(1))
        df['bearish_engulfing'] = (df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & \
                                 (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1))
        
        # Additional cross-asset features (placeholder)
        df = self._add_cross_asset_features(df)
        
        return df
    
    def _add_cross_asset_features(self, df):
        """Add cross-asset features (placeholder for multi-asset support)"""
        # This will be expanded when we add multi-asset support
        return df
    
    def _clean_features(self, df):
        """Clean and validate features"""
        # Replace inf and -inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        
        return df
    
    def get_feature_names(self):
        """Get list of engineered feature names"""
        return self.feature_names

# Backward compatibility
def create_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Legacy function for backward compatibility"""
    engineer = AdvancedFeatureEngineer()
    return engineer.create_all_features(data)
