import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import sys

# Add src to path for imports
sys.path.append('/opt/python/src')

from src.features import AdvancedFeatureEngineer
from src.ensemble_models import EnsemblePredictor
from src.sentiment_analysis import SentimentFeatureEngineer

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

class LambdaMLPredictor:
    """AWS Lambda handler for ML predictions"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.sentiment_engineer = SentimentFeatureEngineer()
        self.models = {}
        self.model_bucket = os.environ.get('MODEL_BUCKET', 'ml-trading-models')
        
    def load_model(self, symbol: str):
        """Load model for a specific symbol from S3"""
        try:
            if symbol in self.models:
                return self.models[symbol]
            
            # Download model from S3
            model_key = f"models/{symbol}_ensemble.pkl"
            
            # For Lambda, we'll use a simplified model loading
            # In practice, you'd download and deserialize the actual model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Load pre-trained weights (simplified)
            self.models[symbol] = model
            return model
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return None
    
    def predict(self, symbol: str, market_data: dict) -> dict:
        """Make prediction for a symbol"""
        try:
            # Convert market data to DataFrame
            df = pd.DataFrame([market_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Create features
            features_df = self.feature_engineer.create_all_features(df)
            
            # Add sentiment features
            sentiment_features = self.sentiment_engineer.create_sentiment_features_for_symbol(symbol, df)
            features_df = pd.concat([features_df, sentiment_features], axis=1)
            
            # Prepare features for prediction
            feature_cols = [col for col in features_df.columns 
                           if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
            X = features_df[feature_cols].fillna(0)
            
            # Load model
            model = self.load_model(symbol)
            if model is None:
                return self._default_prediction()
            
            # Make prediction
            prediction_proba = model.predict_proba(X)[0][1]
            prediction = 1 if prediction_proba > 0.5 else 0
            confidence = abs(prediction_proba - 0.5) * 2
            
            # Determine signal
            if prediction_proba > 0.6:
                signal = 'BUY'
            elif prediction_proba < 0.4:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'symbol': symbol,
                'prediction': float(prediction_proba),
                'confidence': float(confidence),
                'signal': signal,
                'timestamp': datetime.now().isoformat(),
                'model_version': '1.0',
                'features_used': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> dict:
        """Return default prediction when model fails"""
        return {
            'symbol': 'UNKNOWN',
            'prediction': 0.5,
            'confidence': 0.0,
            'signal': 'HOLD',
            'timestamp': datetime.now().isoformat(),
            'model_version': 'default',
            'features_used': 0,
            'error': 'Model not available'
        }

# Initialize predictor
predictor = LambdaMLPredictor()

def lambda_handler(event, context):
    """Main Lambda handler function"""
    try:
        # Parse event
        if 'Records' in event:
            # SQS/SNS event
            for record in event['Records']:
                if 'body' in record:
                    message = json.loads(record['body'])
                    result = process_message(message)
                    logger.info(f"Processed message: {result}")
        else:
            # Direct invocation
            result = process_message(event)
            logger.info(f"Processed direct invocation: {result}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Prediction completed successfully',
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

def process_message(message: dict) -> dict:
    """Process a single message"""
    try:
        symbol = message.get('symbol')
        market_data = message.get('market_data', {})
        
        if not symbol or not market_data:
            raise ValueError("Missing required fields: symbol and market_data")
        
        # Make prediction
        prediction = predictor.predict(symbol, market_data)
        
        # Store result in S3 or send to another Lambda
        store_prediction(prediction)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return predictor._default_prediction()

def store_prediction(prediction: dict):
    """Store prediction result"""
    try:
        # Store in S3
        bucket = os.environ.get('PREDICTIONS_BUCKET', 'ml-trading-predictions')
        key = f"predictions/{prediction['symbol']}/{datetime.now().strftime('%Y/%m/%d')}/{prediction['timestamp']}.json"
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(prediction),
            ContentType='application/json'
        )
        
        logger.info(f"Stored prediction in S3: {key}")
        
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")

# Batch processing function
def batch_predict(event, context):
    """Process multiple symbols in batch"""
    try:
        symbols = event.get('symbols', [])
        market_data = event.get('market_data', {})
        
        results = []
        for symbol in symbols:
            symbol_data = market_data.get(symbol, {})
            prediction = predictor.predict(symbol, symbol_data)
            results.append(prediction)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'predictions': results,
                'total_processed': len(results),
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }

# Model training function
def train_model(event, context):
    """Train model for a specific symbol"""
    try:
        symbol = event.get('symbol')
        training_data = event.get('training_data', [])
        
        if not symbol or not training_data:
            raise ValueError("Missing required fields: symbol and training_data")
        
        # Convert training data to DataFrame
        df = pd.DataFrame(training_data)
        
        # Create features
        feature_engineer = AdvancedFeatureEngineer()
        features_df = feature_engineer.create_all_features(df)
        
        # Create target
        features_df['target'] = (features_df['Close'].pct_change().shift(-1) > 0).astype(int)
        features_df = features_df.dropna()
        
        # Prepare training data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'target', 'symbol']]
        X = features_df[feature_cols].fillna(0)
        y = features_df['target']
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model to S3
        import pickle
        model_key = f"models/{symbol}_ensemble.pkl"
        model_data = pickle.dumps(model)
        
        s3_client.put_object(
            Bucket=os.environ.get('MODEL_BUCKET', 'ml-trading-models'),
            Key=model_key,
            Body=model_data,
            ContentType='application/octet-stream'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Model trained for {symbol}',
                'model_key': model_key,
                'training_samples': len(X),
                'features': len(feature_cols),
                'timestamp': datetime.now().isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Model training error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }
