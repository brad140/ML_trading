import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MultiAssetDataProcessor:
    """Process data for 500+ equities efficiently"""
    
    def __init__(self, 
                 symbols: List[str],
                 start_date: str = "2010-01-01",
                 end_date: Optional[str] = None,
                 max_workers: int = 10):
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.max_workers = max_workers
        self.data_cache = {}
        
    def fetch_all_data(self, 
                      interval: str = "1d",
                      progress_bar: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols in parallel"""
        
        logger.info(f"Fetching data for {len(self.symbols)} symbols...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._fetch_single_symbol, symbol, interval): symbol 
                for symbol in self.symbols
            }
            
            # Collect results
            results = {}
            if progress_bar:
                futures = tqdm(concurrent.futures.as_completed(future_to_symbol), 
                             total=len(self.symbols), desc="Fetching data")
            else:
                futures = concurrent.futures.as_completed(future_to_symbol)
            
            for future in futures:
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[symbol] = data
                        self.data_cache[symbol] = data
                    else:
                        logger.warning(f"No data available for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
        
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def _fetch_single_symbol(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval=interval
            )
            
            if data.empty:
                return None
            
            # Clean data
            data = data.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
            data = data.dropna()
            
            # Add symbol column
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def create_features_for_all_assets(self, 
                                     feature_engineer,
                                     data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create features for all assets in parallel"""
        
        logger.info(f"Creating features for {len(data_dict)} assets...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._create_features_single_asset, symbol, data, feature_engineer): symbol
                for symbol, data in data_dict.items()
            }
            
            # Collect results
            results = {}
            futures = tqdm(concurrent.futures.as_completed(future_to_symbol), 
                          total=len(data_dict), desc="Creating features")
            
            for future in futures:
                symbol = future_to_symbol[future]
                try:
                    features = future.result()
                    if features is not None and not features.empty:
                        results[symbol] = features
                    else:
                        logger.warning(f"Failed to create features for {symbol}")
                except Exception as e:
                    logger.error(f"Error creating features for {symbol}: {e}")
        
        logger.info(f"Successfully created features for {len(results)} assets")
        return results
    
    def _create_features_single_asset(self, 
                                    symbol: str, 
                                    data: pd.DataFrame, 
                                    feature_engineer) -> Optional[pd.DataFrame]:
        """Create features for a single asset"""
        try:
            features = feature_engineer.create_all_features(data)
            return features
        except Exception as e:
            logger.error(f"Error creating features for {symbol}: {e}")
            return None
    
    def create_targets_for_all_assets(self, 
                                    data_dict: Dict[str, pd.DataFrame],
                                    target_type: str = 'directional',
                                    horizon: int = 1) -> Dict[str, pd.Series]:
        """Create targets for all assets"""
        
        targets = {}
        
        for symbol, data in data_dict.items():
            try:
                if target_type == 'directional':
                    # Directional target (up/down)
                    target = (data['Close'].pct_change(horizon).shift(-horizon) > 0).astype(int)
                elif target_type == 'regression':
                    # Regression target (actual return)
                    target = data['Close'].pct_change(horizon).shift(-horizon)
                elif target_type == 'volatility':
                    # Volatility target
                    target = data['Close'].pct_change().rolling(horizon).std().shift(-horizon)
                else:
                    raise ValueError(f"Unknown target type: {target_type}")
                
                targets[symbol] = target
                
            except Exception as e:
                logger.error(f"Error creating target for {symbol}: {e}")
                targets[symbol] = pd.Series(dtype=float)
        
        return targets
    
    def filter_assets_by_quality(self, 
                                data_dict: Dict[str, pd.DataFrame],
                                min_data_points: int = 1000,
                                max_missing_ratio: float = 0.1,
                                min_volume: float = 1000) -> Dict[str, pd.DataFrame]:
        """Filter assets based on data quality criteria"""
        
        filtered_data = {}
        
        for symbol, data in data_dict.items():
            try:
                # Check data length
                if len(data) < min_data_points:
                    logger.warning(f"{symbol}: Insufficient data points ({len(data)} < {min_data_points})")
                    continue
                
                # Check missing data ratio
                missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
                if missing_ratio > max_missing_ratio:
                    logger.warning(f"{symbol}: Too many missing values ({missing_ratio:.2%} > {max_missing_ratio:.2%})")
                    continue
                
                # Check volume
                if 'Volume' in data.columns:
                    avg_volume = data['Volume'].mean()
                    if avg_volume < min_volume:
                        logger.warning(f"{symbol}: Low average volume ({avg_volume:.0f} < {min_volume})")
                        continue
                
                filtered_data[symbol] = data
                
            except Exception as e:
                logger.error(f"Error filtering {symbol}: {e}")
        
        logger.info(f"Filtered {len(data_dict)} assets to {len(filtered_data)} high-quality assets")
        return filtered_data
    
    def create_cross_asset_features(self, 
                                  data_dict: Dict[str, pd.DataFrame],
                                  market_index: str = 'SPY') -> Dict[str, pd.DataFrame]:
        """Create cross-asset features using market index"""
        
        if market_index not in data_dict:
            logger.warning(f"Market index {market_index} not found in data")
            return data_dict
        
        market_data = data_dict[market_index]
        enhanced_data = {}
        
        for symbol, data in data_dict.items():
            try:
                # Align data with market index
                aligned_data = data.reindex(market_data.index, method='ffill')
                
                # Add market features
                aligned_data['market_return'] = market_data['Close'].pct_change()
                aligned_data['market_volatility'] = market_data['Close'].pct_change().rolling(20).std()
                
                # Relative performance
                aligned_data['relative_performance'] = (
                    aligned_data['Close'].pct_change() - market_data['Close'].pct_change()
                )
                
                # Beta calculation (simplified)
                returns = aligned_data['Close'].pct_change().dropna()
                market_returns = market_data['Close'].pct_change().dropna()
                
                if len(returns) > 20 and len(market_returns) > 20:
                    # Align lengths
                    min_len = min(len(returns), len(market_returns))
                    returns = returns.iloc[-min_len:]
                    market_returns = market_returns.iloc[-min_len:]
                    
                    if returns.std() > 0 and market_returns.std() > 0:
                        beta = returns.corr(market_returns) * (returns.std() / market_returns.std())
                        aligned_data['beta'] = beta
                    else:
                        aligned_data['beta'] = 1.0
                else:
                    aligned_data['beta'] = 1.0
                
                enhanced_data[symbol] = aligned_data
                
            except Exception as e:
                logger.error(f"Error creating cross-asset features for {symbol}: {e}")
                enhanced_data[symbol] = data
        
        return enhanced_data

class AssetUniverseManager:
    """Manage universe of 500+ equities"""
    
    def __init__(self):
        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'CB'],
            'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
            'Industrial': ['BA', 'CAT', 'GE', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'MMM', 'EMR'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'WMB', 'OKE', 'PSX'],
            'Utilities': ['NEE', 'DUK', 'SO', 'AEP', 'EXC', 'XEL', 'SRE', 'PPL', 'WEC', 'ES'],
            'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'PPG', 'NEM', 'FCX', 'NUE'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'UDR'],
            'Communication': ['VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'DISH', 'FOX', 'VIAC']
        }
        
        # Get S&P 500 symbols (simplified list)
        self.sp500_symbols = self._get_sp500_symbols()
    
    def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols (simplified version)"""
        # In practice, you'd fetch this from a reliable source
        # For now, return a representative sample
        symbols = []
        for sector_symbols in self.sectors.values():
            symbols.extend(sector_symbols)
        
        # Add more symbols to reach 500+
        additional_symbols = [
            'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MCHP',
            'ADI', 'MRVL', 'SWKS', 'QRVO', 'TER', 'SNPS', 'CDNS', 'ANSS', 'KEYS', 'FTNT',
            'PANW', 'CRWD', 'OKTA', 'ZS', 'NET', 'DDOG', 'SNOW', 'PLTR', 'RBLX', 'U',
            'SHOP', 'SQ', 'PYPL', 'V', 'MA', 'AXP', 'COF', 'DFS', 'FISV', 'FIS',
            'GPN', 'JKHY', 'NDAQ', 'TSS', 'WU', 'ZION', 'CFG', 'FITB', 'HBAN', 'KEY',
            'MTB', 'PNC', 'RF', 'STI', 'TFC', 'USB', 'WFC', 'ZION', 'AFL', 'ALL',
            'AON', 'AIG', 'BRO', 'CBOE', 'CINF', 'CL', 'GL', 'HIG', 'L', 'MKL',
            'MMC', 'PGR', 'PRU', 'RE', 'SPGI', 'TRV', 'WRB', 'XL', 'AEP', 'AWK',
            'CNP', 'CMS', 'D', 'DTE', 'DUK', 'ED', 'EIX', 'ES', 'ETR', 'EXC',
            'FE', 'LNT', 'NEE', 'NI', 'NRG', 'PCG', 'PEG', 'PNW', 'PPL', 'SRE',
            'SO', 'WEC', 'XEL', 'A', 'ABT', 'ALGN', 'ALXN', 'AMGN', 'BAX', 'BDX',
            'BIIB', 'BMY', 'BSX', 'CERN', 'CI', 'COO', 'CVS', 'DHR', 'DVA', 'EW',
            'GILD', 'HCA', 'HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'ISRG', 'JNJ',
            'LH', 'LLY', 'MDT', 'MRNA', 'MRK', 'MTD', 'PFE', 'PKI', 'REGN', 'SYK',
            'TMO', 'UNH', 'VRTX', 'WAT', 'ZBH', 'ZTS'
        ]
        
        symbols.extend(additional_symbols)
        return symbols[:500]  # Limit to 500 symbols
    
    def get_sector_symbols(self, sector: str) -> List[str]:
        """Get symbols for a specific sector"""
        return self.sectors.get(sector, [])
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in the universe"""
        return self.sp500_symbols
    
    def get_symbols_by_criteria(self, 
                               min_market_cap: float = 1e9,
                               max_market_cap: float = 1e12,
                               sectors: List[str] = None) -> List[str]:
        """Get symbols filtered by criteria (placeholder)"""
        # In practice, you'd filter by actual market cap data
        # For now, return all symbols
        symbols = self.get_all_symbols()
        
        if sectors:
            sector_symbols = []
            for sector in sectors:
                sector_symbols.extend(self.get_sector_symbols(sector))
            symbols = [s for s in symbols if s in sector_symbols]
        
        return symbols

class MultiAssetEnsembleTrainer:
    """Train ensemble models for multiple assets"""
    
    def __init__(self, 
                 feature_engineer,
                 ensemble_params: Dict = None):
        
        self.feature_engineer = feature_engineer
        self.ensemble_params = ensemble_params or {}
        self.asset_models = {}
        self.asset_performance = {}
        
    def train_all_assets(self, 
                        data_dict: Dict[str, pd.DataFrame],
                        targets_dict: Dict[str, pd.Series],
                        validation_split: float = 0.2) -> Dict[str, Dict]:
        """Train ensemble models for all assets"""
        
        logger.info(f"Training ensemble models for {len(data_dict)} assets...")
        
        results = {}
        
        for symbol in tqdm(data_dict.keys(), desc="Training models"):
            try:
                # Get data and targets
                data = data_dict[symbol]
                targets = targets_dict[symbol]
                
                # Align data and targets
                common_index = data.index.intersection(targets.index)
                data = data.loc[common_index]
                targets = targets.loc[common_index]
                
                if len(data) < 100:  # Minimum data requirement
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} points")
                    continue
                
                # Create features
                features = self.feature_engineer.create_all_features(data)
                
                # Prepare features and targets
                feature_cols = [col for col in features.columns 
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
                X = features[feature_cols].fillna(0)
                y = targets.fillna(0)
                
                # Remove rows with NaN targets
                valid_mask = ~y.isna()
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) < 50:
                    logger.warning(f"Insufficient valid data for {symbol}: {len(X)} points")
                    continue
                
                # Train ensemble model
                from .ensemble_models import EnsemblePredictor
                ensemble = EnsemblePredictor(**self.ensemble_params)
                
                # Split data
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train model
                ensemble.fit(X_train, y_train)
                
                # Evaluate model
                if len(X_val) > 0:
                    metrics = ensemble.evaluate(X_val, y_val)
                    self.asset_performance[symbol] = metrics
                else:
                    metrics = {'ensemble': {'accuracy': 0.5}}
                    self.asset_performance[symbol] = metrics
                
                # Store model
                self.asset_models[symbol] = ensemble
                
                results[symbol] = {
                    'model': ensemble,
                    'metrics': metrics,
                    'feature_count': len(feature_cols),
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val)
                }
                
                logger.info(f"Trained {symbol}: Accuracy={metrics['ensemble']['accuracy']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        logger.info(f"Successfully trained models for {len(self.asset_models)} assets")
        return results
    
    def get_asset_rankings(self, metric: str = 'accuracy') -> pd.DataFrame:
        """Get asset rankings by performance metric"""
        rankings = []
        
        for symbol, performance in self.asset_performance.items():
            if 'ensemble' in performance and metric in performance['ensemble']:
                rankings.append({
                    'symbol': symbol,
                    'metric': performance['ensemble'][metric],
                    'sharpe_ratio': performance['ensemble'].get('sharpe_ratio', 0),
                    'precision': performance['ensemble'].get('precision', 0),
                    'recall': performance['ensemble'].get('recall', 0)
                })
        
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('metric', ascending=False)
        
        return rankings_df
    
    def predict_all_assets(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """Make predictions for all assets"""
        predictions = {}
        
        for symbol, data in data_dict.items():
            if symbol in self.asset_models:
                try:
                    # Create features
                    features = self.feature_engineer.create_all_features(data)
                    feature_cols = [col for col in features.columns 
                                  if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
                    X = features[feature_cols].fillna(0)
                    
                    # Make prediction
                    pred_proba = self.asset_models[symbol].predict_proba(X)
                    predictions[symbol] = pred_proba[:, 1]  # Probability of positive class
                    
                except Exception as e:
                    logger.error(f"Error predicting {symbol}: {e}")
                    predictions[symbol] = np.full(len(data), 0.5)  # Neutral prediction
        
        return predictions
