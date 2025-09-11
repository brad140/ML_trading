import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier
from .features import create_advanced_features

def run_walk_forward(symbol="SPY", start_date="2010-01-01", n_splits=5):
    df = yf.Ticker(symbol).history(start=start_date, interval="1d")
    df = create_advanced_features(df)
    df['target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)
    df = df.drop(['Dividends', 'Stock Splits'], axis=1)
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
