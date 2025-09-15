import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our new modules
from features import AdvancedFeatureEngineer
from ensemble_models import EnsemblePredictor, MultiAssetEnsemble
from sentiment_analysis import SentimentFeatureEngineer
from backtesting import BacktestEngine, WalkForwardBacktest
from multi_asset_processor import MultiAssetDataProcessor, AssetUniverseManager, MultiAssetEnsembleTrainer
from kafka_integration import KafkaDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_asset_ensemble(symbol="SPY", start_date="2010-01-01", end_date=None):
    """Run ensemble model for a single asset with comprehensive features"""
    
    logger.info(f"Running ensemble model for {symbol}")
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer()
    sentiment_engineer = SentimentFeatureEngineer()
    ensemble = EnsemblePredictor()
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    df = df.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
    
    # Create technical features
    df = feature_engineer.create_all_features(df)
    
    # Create sentiment features
    sentiment_features = sentiment_engineer.create_sentiment_features_for_symbol(symbol, df)
    df = pd.concat([df, sentiment_features], axis=1)
    
    # Create target
    df['target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    if len(df) < 100:
        logger.error(f"Insufficient data for {symbol}: {len(df)} points")
        return None
    
    # Prepare features
    feature_cols = [col for col in df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'symbol']]
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    metrics = ensemble.evaluate(X_val, y_val)
    
    # Run backtest
    backtest_engine = BacktestEngine()
    backtest_results = backtest_engine.run_backtest(
        df[['Open', 'High', 'Low', 'Close', 'Volume']],
        ensemble.predict_proba(X_val)[:, 1]
    )
    
    logger.info(f"Results for {symbol}:")
    logger.info(f"Accuracy: {metrics['ensemble']['accuracy']:.3f}")
    logger.info(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.3f}")
    logger.info(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']:.3f}")
    
    return {
        'symbol': symbol,
        'metrics': metrics,
        'backtest': backtest_results,
        'model': ensemble,
        'feature_count': len(feature_cols)
    }

def run_multi_asset_ensemble(symbols=None, start_date="2010-01-01", end_date=None):
    """Run ensemble models for 500+ equities"""
    
    logger.info("Running multi-asset ensemble training")
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer()
    sentiment_engineer = SentimentFeatureEngineer()
    
    # Get symbol universe
    if symbols is None:
        universe_manager = AssetUniverseManager()
        symbols = universe_manager.get_all_symbols()[:100]  # Limit for demo
    
    logger.info(f"Processing {len(symbols)} symbols")
    
    # Process data for all assets
    processor = MultiAssetDataProcessor(symbols, start_date, end_date)
    data_dict = processor.fetch_all_data()
    
    # Filter by quality
    data_dict = processor.filter_assets_by_quality(data_dict)
    
    # Create cross-asset features
    data_dict = processor.create_cross_asset_features(data_dict)
    
    # Create features for all assets
    features_dict = processor.create_features_for_all_assets(feature_engineer, data_dict)
    
    # Add sentiment features
    for symbol in features_dict:
        try:
            sentiment_features = sentiment_engineer.create_sentiment_features_for_symbol(
                symbol, data_dict[symbol]
            )
            features_dict[symbol] = pd.concat([features_dict[symbol], sentiment_features], axis=1)
        except Exception as e:
            logger.warning(f"Could not add sentiment features for {symbol}: {e}")
    
    # Create targets
    targets_dict = processor.create_targets_for_all_assets(features_dict)
    
    # Train ensemble models
    trainer = MultiAssetEnsembleTrainer(feature_engineer)
    results = trainer.train_all_assets(features_dict, targets_dict)
    
    # Get performance rankings
    rankings = trainer.get_asset_rankings('accuracy')
    
    logger.info("Top 10 performing assets:")
    logger.info(rankings.head(10).to_string())
    
    # Calculate aggregate metrics
    successful_models = [r for r in results.values() if 'error' not in r]
    avg_accuracy = np.mean([r['metrics']['ensemble']['accuracy'] for r in successful_models])
    
    logger.info(f"Average accuracy across {len(successful_models)} models: {avg_accuracy:.3f}")
    
    return {
        'results': results,
        'rankings': rankings,
        'successful_models': len(successful_models),
        'avg_accuracy': avg_accuracy
    }

def run_walk_forward_validation(symbol="SPY", start_date="2010-01-01", n_splits=5):
    """Run walk-forward validation with ensemble models"""
    
    logger.info(f"Running walk-forward validation for {symbol}")
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer()
    sentiment_engineer = SentimentFeatureEngineer()
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, interval="1d")
    df = df.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
    
    # Create features
    df = feature_engineer.create_all_features(df)
    sentiment_features = sentiment_engineer.create_sentiment_features_for_symbol(symbol, df)
    df = pd.concat([df, sentiment_features], axis=1)
    
    # Create target
    df['target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)
    
    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Prepare features
    feature_cols = [col for col in df.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'symbol']]
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    # Run walk-forward validation
    walk_forward = WalkForwardBacktest(train_period=252, test_period=63, step_size=21)
    
    # Create a simple ensemble for walk-forward
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42)
    ensemble = VotingClassifier([('rf', rf), ('lr', lr)], voting='soft')
    
    results = walk_forward.run_walk_forward(df, ensemble, feature_cols)
    
    # Calculate aggregate metrics
    metrics_df = pd.DataFrame([r['metrics'] for r in results])
    avg_accuracy = metrics_df['directional_accuracy'].mean()
    avg_sharpe = metrics_df['sharpe_ratio'].mean()
    
    logger.info(f"Walk-forward results for {symbol}:")
    logger.info(f"Average Accuracy: {avg_accuracy:.3f}")
    logger.info(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
    
    return {
        'results': results,
        'avg_accuracy': avg_accuracy,
        'avg_sharpe': avg_sharpe,
        'total_periods': len(results)
    }

def run_real_time_pipeline(symbols=None, bootstrap_servers=['localhost:9092']):
    """Run real-time data pipeline with Kafka"""
    
    logger.info("Starting real-time data pipeline")
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer()
    sentiment_engineer = SentimentFeatureEngineer()
    
    # Get symbols
    if symbols is None:
        universe_manager = AssetUniverseManager()
        symbols = universe_manager.get_all_symbols()[:50]  # Limit for demo
    
    # Initialize Kafka manager
    kafka_manager = KafkaDataManager(symbols, bootstrap_servers)
    
    # Create a simple ensemble model for real-time predictions
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train on historical data first
    logger.info("Training initial model on historical data...")
    sample_data = yf.Ticker('SPY').history(period="1y")
    sample_features = feature_engineer.create_all_features(sample_data)
    sample_target = (sample_data['Close'].pct_change().shift(-1) > 0).astype(int)
    
    feature_cols = [col for col in sample_features.columns 
                   if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    X = sample_features[feature_cols].fillna(0)
    y = sample_target.fillna(0)
    
    model.fit(X, y)
    
    # Start real-time pipeline
    kafka_manager.start_data_pipeline(feature_engineer, model, source='yfinance')
    
    logger.info("Real-time pipeline started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
            stats = kafka_manager.get_data_stats()
            if stats['total_messages_processed'] % 100 == 0:
                logger.info(f"Processed {stats['total_messages_processed']} messages")
    except KeyboardInterrupt:
        logger.info("Stopping real-time pipeline...")
        kafka_manager.stop_data_pipeline()

# Legacy function for backward compatibility
def run_walk_forward(symbol="SPY", start_date="2010-01-01", n_splits=5):
    """Legacy function - now uses ensemble models"""
    logger.warning("Using legacy function. Consider using run_single_asset_ensemble() for better performance.")
    
    # Initialize components
    feature_engineer = AdvancedFeatureEngineer()
    
    # Fetch data
    df = yf.Ticker(symbol).history(start=start_date, interval="1d")
    df = feature_engineer.create_all_features(df)
    df['target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)
    df = df.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        X_train, y_train = df.iloc[train_idx].drop('target', axis=1), df.iloc[train_idx]['target']
        X_val, y_val = df.iloc[val_idx].drop('target', axis=1), df.iloc[val_idx]['target']

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.02,
            depth=4,
            l2_leaf_reg=5,
            verbose=0,
            random_seed=42,
            eval_metric='Accuracy'
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        results.append(acc)
        print(f"Fold {fold+1}/{n_splits}: Accuracy={acc:.2%}")

    print("\nFinal Results:", np.mean(results))
    return results
