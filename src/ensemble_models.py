import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """Ensemble model combining Random Forest, XGBoost, and LSTM for price movement prediction"""
    
    def __init__(self, n_estimators_rf=100, n_estimators_xgb=100, lstm_units=50, 
                 lstm_dropout=0.2, learning_rate=0.01, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.feature_importance = {}
        self.is_fitted = False
        
        # Initialize models
        self._initialize_models(n_estimators_rf, n_estimators_xgb, lstm_units, 
                              lstm_dropout, learning_rate)
    
    def _initialize_models(self, n_estimators_rf, n_estimators_xgb, lstm_units, 
                          lstm_dropout, learning_rate):
        """Initialize all ensemble models"""
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=n_estimators_xgb,
            max_depth=6,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # CatBoost
        self.models['catboost'] = CatBoostClassifier(
            iterations=1000,
            learning_rate=learning_rate,
            depth=6,
            l2_leaf_reg=3,
            random_seed=self.random_state,
            verbose=False
        )
        
        # LSTM (will be built dynamically based on input shape)
        self.lstm_params = {
            'units': lstm_units,
            'dropout': lstm_dropout,
            'learning_rate': learning_rate
        }
    
    def _build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_params['units'], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.lstm_params['dropout']),
            BatchNormalization(),
            
            LSTM(self.lstm_params['units'] // 2, 
                 return_sequences=False),
            Dropout(self.lstm_params['dropout']),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.lstm_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _prepare_lstm_data(self, X, y=None, sequence_length=60):
        """Prepare data for LSTM (create sequences)"""
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, 
            sequence_length=60, early_stopping_patience=10):
        """Train all ensemble models"""
        
        # Prepare data for different models
        X_rf = X.copy()
        X_xgb = X.copy()
        X_cat = X.copy()
        
        # For LSTM, we need to create sequences
        X_lstm, y_lstm = self._prepare_lstm_data(X.values, y.values, sequence_length)
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        
        # Random Forest
        print("Training Random Forest...")
        self.models['random_forest'].fit(X_rf[:split_idx], y[:split_idx])
        
        # XGBoost
        print("Training XGBoost...")
        self.models['xgboost'].fit(
            X_xgb[:split_idx], y[:split_idx],
            eval_set=[(X_xgb[split_idx:], y[split_idx:])],
            verbose=False
        )
        
        # CatBoost
        print("Training CatBoost...")
        self.models['catboost'].fit(
            X_cat[:split_idx], y[:split_idx],
            eval_set=(X_cat[split_idx:], y[split_idx:]),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # LSTM
        print("Training LSTM...")
        lstm_split_idx = int(len(X_lstm) * (1 - validation_split))
        
        self.models['lstm'] = self._build_lstm_model((sequence_length, X.shape[1]))
        
        callbacks = [
            EarlyStopping(patience=early_stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.models['lstm'].fit(
            X_lstm[:lstm_split_idx], y_lstm[:lstm_split_idx],
            validation_data=(X_lstm[lstm_split_idx:], y_lstm[lstm_split_idx:]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Calculate model weights based on validation performance
        self._calculate_model_weights(X, y, split_idx, sequence_length)
        
        # Store feature importance
        self._calculate_feature_importance(X.columns)
        
        self.is_fitted = True
        print("All models trained successfully!")
    
    def _calculate_model_weights(self, X, y, split_idx, sequence_length):
        """Calculate ensemble weights based on validation performance"""
        weights = {}
        
        # Random Forest
        rf_pred = self.models['random_forest'].predict(X[split_idx:])
        weights['random_forest'] = accuracy_score(y[split_idx:], rf_pred)
        
        # XGBoost
        xgb_pred = self.models['xgboost'].predict(X[split_idx:])
        weights['xgboost'] = accuracy_score(y[split_idx:], xgb_pred)
        
        # CatBoost
        cat_pred = self.models['catboost'].predict(X[split_idx:])
        weights['catboost'] = accuracy_score(y[split_idx:], cat_pred)
        
        # LSTM
        X_lstm_val, y_lstm_val = self._prepare_lstm_data(
            X[split_idx:].values, y[split_idx:].values, sequence_length
        )
        if len(X_lstm_val) > 0:
            lstm_pred = (self.models['lstm'].predict(X_lstm_val) > 0.5).astype(int).flatten()
            weights['lstm'] = accuracy_score(y_lstm_val, lstm_pred)
        else:
            weights['lstm'] = 0.5
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}
        
        print(f"Model weights: {self.weights}")
    
    def _calculate_feature_importance(self, feature_names):
        """Calculate and store feature importance from tree-based models"""
        self.feature_importance = {}
        
        # Random Forest importance
        if hasattr(self.models['random_forest'], 'feature_importances_'):
            self.feature_importance['random_forest'] = dict(
                zip(feature_names, self.models['random_forest'].feature_importances_)
            )
        
        # XGBoost importance
        if hasattr(self.models['xgboost'], 'feature_importances_'):
            self.feature_importance['xgboost'] = dict(
                zip(feature_names, self.models['xgboost'].feature_importances_)
            )
        
        # CatBoost importance
        if hasattr(self.models['catboost'], 'feature_importances_'):
            self.feature_importance['catboost'] = dict(
                zip(feature_names, self.models['catboost'].feature_importances_)
            )
    
    def predict(self, X, sequence_length=60):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        
        # Random Forest
        predictions['random_forest'] = self.models['random_forest'].predict_proba(X)[:, 1]
        
        # XGBoost
        predictions['xgboost'] = self.models['xgboost'].predict_proba(X)[:, 1]
        
        # CatBoost
        predictions['catboost'] = self.models['catboost'].predict_proba(X)[:, 1]
        
        # LSTM
        X_lstm, _ = self._prepare_lstm_data(X.values, sequence_length=sequence_length)
        if len(X_lstm) > 0:
            lstm_pred = self.models['lstm'].predict(X_lstm).flatten()
            # Pad with zeros for the first sequence_length predictions
            lstm_padded = np.zeros(len(X))
            lstm_padded[sequence_length:] = lstm_pred
            predictions['lstm'] = lstm_padded
        else:
            predictions['lstm'] = np.zeros(len(X))
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        return ensemble_pred, predictions
    
    def predict_proba(self, X, sequence_length=60):
        """Return ensemble prediction probabilities"""
        ensemble_pred, _ = self.predict(X, sequence_length)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def evaluate(self, X, y, sequence_length=60):
        """Evaluate ensemble performance"""
        ensemble_pred, individual_preds = self.predict(X, sequence_length)
        binary_pred = (ensemble_pred > 0.5).astype(int)
        
        metrics = {
            'ensemble': {
                'accuracy': accuracy_score(y, binary_pred),
                'precision': precision_score(y, binary_pred, zero_division=0),
                'recall': recall_score(y, binary_pred, zero_division=0),
                'f1': f1_score(y, binary_pred, zero_division=0)
            }
        }
        
        # Individual model metrics
        for model_name, pred in individual_preds.items():
            binary_pred_model = (pred > 0.5).astype(int)
            metrics[model_name] = {
                'accuracy': accuracy_score(y, binary_pred_model),
                'precision': precision_score(y, binary_pred_model, zero_division=0),
                'recall': recall_score(y, binary_pred_model, zero_division=0),
                'f1': f1_score(y, binary_pred_model, zero_division=0)
            }
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """Get top N most important features across all models"""
        if not self.feature_importance:
            return {}
        
        # Average importance across models
        all_features = set()
        for model_importance in self.feature_importance.values():
            all_features.update(model_importance.keys())
        
        avg_importance = {}
        for feature in all_features:
            importances = []
            for model_importance in self.feature_importance.values():
                if feature in model_importance:
                    importances.append(model_importance[feature])
            avg_importance[feature] = np.mean(importances)
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])

class MultiAssetEnsemble:
    """Ensemble predictor for multiple assets (500+ equities)"""
    
    def __init__(self, asset_list, **ensemble_params):
        self.asset_list = asset_list
        self.ensemble_params = ensemble_params
        self.asset_models = {}
        self.asset_weights = {}
    
    def fit_asset(self, symbol, X, y, **fit_params):
        """Fit ensemble model for a specific asset"""
        print(f"Training ensemble for {symbol}...")
        ensemble = EnsemblePredictor(**self.ensemble_params)
        ensemble.fit(X, y, **fit_params)
        self.asset_models[symbol] = ensemble
        return ensemble
    
    def fit_all_assets(self, data_dict, **fit_params):
        """Fit ensemble models for all assets"""
        for symbol, (X, y) in data_dict.items():
            self.fit_asset(symbol, X, y, **fit_params)
    
    def predict_asset(self, symbol, X, **predict_params):
        """Predict for a specific asset"""
        if symbol not in self.asset_models:
            raise ValueError(f"No model found for asset {symbol}")
        
        return self.asset_models[symbol].predict(X, **predict_params)
    
    def predict_all_assets(self, data_dict, **predict_params):
        """Predict for all assets"""
        predictions = {}
        for symbol, X in data_dict.items():
            predictions[symbol] = self.predict_asset(symbol, X, **predict_params)
        return predictions
    
    def get_asset_performance(self, symbol, X, y, **eval_params):
        """Get performance metrics for a specific asset"""
        if symbol not in self.asset_models:
            raise ValueError(f"No model found for asset {symbol}")
        
        return self.asset_models[symbol].evaluate(X, y, **eval_params)
