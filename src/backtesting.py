import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    """Comprehensive backtesting framework with performance metrics"""
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = {}
        
    def run_backtest(self, 
                    price_data: pd.DataFrame,
                    predictions: pd.Series,
                    strategy_type: str = 'long_short',
                    confidence_threshold: float = 0.6,
                    rebalance_frequency: str = 'daily',
                    max_position_size: float = 0.1) -> Dict:
        """
        Run comprehensive backtest
        
        Args:
            price_data: DataFrame with OHLCV data
            predictions: Series with prediction probabilities
            strategy_type: 'long_only', 'short_only', 'long_short'
            confidence_threshold: Minimum confidence for trades
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            max_position_size: Maximum position size as fraction of capital
        """
        
        # Prepare data
        backtest_data = self._prepare_backtest_data(price_data, predictions)
        
        # Generate signals
        signals = self._generate_signals(
            backtest_data, strategy_type, confidence_threshold
        )
        
        # Calculate returns
        returns = self._calculate_returns(backtest_data, signals)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(returns, backtest_data)
        
        # Store results
        self.results = {
            'backtest_data': backtest_data,
            'signals': signals,
            'returns': returns,
            'metrics': metrics,
            'strategy_type': strategy_type,
            'confidence_threshold': confidence_threshold
        }
        
        return self.results
    
    def _prepare_backtest_data(self, price_data: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
        """Prepare data for backtesting"""
        data = price_data.copy()
        
        # Ensure predictions are aligned with price data
        if isinstance(predictions, pd.Series):
            data['prediction'] = predictions.reindex(data.index, method='ffill')
        else:
            data['prediction'] = predictions
        
        # Calculate daily returns
        data['returns'] = data['Close'].pct_change()
        
        # Calculate log returns
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate volatility
        data['volatility'] = data['returns'].rolling(20).std() * np.sqrt(252)
        
        # Calculate moving averages
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['sma_50'] = data['Close'].rolling(50).mean()
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def _generate_signals(self, data: pd.DataFrame, strategy_type: str, 
                         confidence_threshold: float) -> pd.DataFrame:
        """Generate trading signals based on predictions"""
        signals = pd.DataFrame(index=data.index)
        
        if strategy_type == 'long_only':
            signals['position'] = np.where(
                data['prediction'] > confidence_threshold, 1, 0
            )
        elif strategy_type == 'short_only':
            signals['position'] = np.where(
                data['prediction'] < (1 - confidence_threshold), -1, 0
            )
        elif strategy_type == 'long_short':
            signals['position'] = np.where(
                data['prediction'] > confidence_threshold, 1,
                np.where(data['prediction'] < (1 - confidence_threshold), -1, 0)
            )
        
        # Calculate position changes
        signals['position_change'] = signals['position'].diff()
        
        # Calculate trade costs
        signals['trade_cost'] = np.abs(signals['position_change']) * self.commission
        
        return signals
    
    def _calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns"""
        returns = pd.DataFrame(index=data.index)
        
        # Strategy returns
        returns['strategy_returns'] = signals['position'].shift(1) * data['returns']
        
        # Apply transaction costs
        returns['strategy_returns'] -= signals['trade_cost']
        
        # Apply slippage
        returns['strategy_returns'] -= np.abs(signals['position_change']) * self.slippage
        
        # Cumulative returns
        returns['cumulative_returns'] = (1 + returns['strategy_returns']).cumprod()
        returns['cumulative_returns'] *= self.initial_capital
        
        # Benchmark returns (buy and hold)
        returns['benchmark_returns'] = data['returns']
        returns['benchmark_cumulative'] = (1 + returns['benchmark_returns']).cumprod()
        returns['benchmark_cumulative'] *= self.initial_capital
        
        # Excess returns
        returns['excess_returns'] = returns['strategy_returns'] - returns['benchmark_returns']
        
        return returns
    
    def _calculate_performance_metrics(self, returns: pd.DataFrame, 
                                     data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        total_return = returns['cumulative_returns'].iloc[-1] / self.initial_capital - 1
        benchmark_return = returns['benchmark_cumulative'].iloc[-1] / self.initial_capital - 1
        
        metrics['total_return'] = total_return
        metrics['benchmark_return'] = benchmark_return
        metrics['excess_return'] = total_return - benchmark_return
        
        # Risk metrics
        strategy_vol = returns['strategy_returns'].std() * np.sqrt(252)
        benchmark_vol = returns['benchmark_returns'].std() * np.sqrt(252)
        
        metrics['strategy_volatility'] = strategy_vol
        metrics['benchmark_volatility'] = benchmark_vol
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = (total_return - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino ratio
        downside_returns = returns['strategy_returns'][returns['strategy_returns'] < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (total_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        metrics['sortino_ratio'] = sortino_ratio
        
        # Maximum drawdown
        cumulative = returns['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = max_drawdown
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics['calmar_ratio'] = calmar_ratio
        
        # Win rate and average win/loss
        strategy_returns = returns['strategy_returns'].dropna()
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        
        metrics['win_rate'] = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
        metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
        metrics['profit_factor'] = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0
        
        # Information ratio
        excess_returns = returns['excess_returns'].dropna()
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        metrics['information_ratio'] = information_ratio
        
        # Accuracy metrics (for classification)
        if 'prediction' in data.columns:
            # Calculate directional accuracy
            actual_direction = np.where(data['returns'] > 0, 1, 0)
            predicted_direction = np.where(data['prediction'] > 0.5, 1, 0)
            
            # Align lengths
            min_len = min(len(actual_direction), len(predicted_direction))
            actual_direction = actual_direction[:min_len]
            predicted_direction = predicted_direction[:min_len]
            
            accuracy = np.mean(actual_direction == predicted_direction)
            metrics['directional_accuracy'] = accuracy
            
            # Precision and recall
            true_positives = np.sum((actual_direction == 1) & (predicted_direction == 1))
            false_positives = np.sum((actual_direction == 0) & (predicted_direction == 1))
            false_negatives = np.sum((actual_direction == 1) & (predicted_direction == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1_score'] = f1_score
        
        # Additional metrics
        metrics['total_trades'] = len(strategy_returns)
        metrics['profitable_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        
        # VaR and CVaR
        var_95 = np.percentile(strategy_returns, 5)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        metrics['var_95'] = var_95
        metrics['cvar_95'] = cvar_95
        
        return metrics
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results"""
        if not self.results:
            raise ValueError("No backtest results available. Run backtest first.")
        
        data = self.results['backtest_data']
        returns = self.results['returns']
        signals = self.results['signals']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price and signals
        axes[0, 0].plot(data.index, data['Close'], label='Price', alpha=0.7)
        
        # Plot buy signals
        buy_signals = data[signals['position'] == 1]
        if not buy_signals.empty:
            axes[0, 0].scatter(buy_signals.index, buy_signals['Close'], 
                             color='green', marker='^', s=50, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = data[signals['position'] == -1]
        if not sell_signals.empty:
            axes[0, 0].scatter(sell_signals.index, sell_signals['Close'], 
                             color='red', marker='v', s=50, label='Sell Signal')
        
        axes[0, 0].set_title('Price and Trading Signals')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Cumulative returns
        axes[0, 1].plot(returns.index, returns['cumulative_returns'], 
                       label='Strategy', linewidth=2)
        axes[0, 1].plot(returns.index, returns['benchmark_cumulative'], 
                       label='Benchmark', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Drawdown
        cumulative = returns['cumulative_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[1, 0].fill_between(returns.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].plot(returns.index, drawdown, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].grid(True)
        
        # Returns distribution
        axes[1, 1].hist(returns['strategy_returns'].dropna(), bins=50, alpha=0.7, 
                       label='Strategy', density=True)
        axes[1, 1].hist(returns['benchmark_returns'].dropna(), bins=50, alpha=0.7, 
                       label='Benchmark', density=True)
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_summary_report(self) -> str:
        """Generate a summary report of backtest results"""
        if not self.results:
            return "No backtest results available."
        
        metrics = self.results['metrics']
        
        report = f"""
        BACKTEST SUMMARY REPORT
        =======================
        
        Strategy: {self.results['strategy_type']}
        Confidence Threshold: {self.results['confidence_threshold']}
        
        PERFORMANCE METRICS
        -------------------
        Total Return: {metrics['total_return']:.2%}
        Benchmark Return: {metrics['benchmark_return']:.2%}
        Excess Return: {metrics['excess_return']:.2%}
        
        RISK METRICS
        ------------
        Strategy Volatility: {metrics['strategy_volatility']:.2%}
        Benchmark Volatility: {metrics['benchmark_volatility']:.2%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Sortino Ratio: {metrics['sortino_ratio']:.2f}
        Maximum Drawdown: {metrics['max_drawdown']:.2%}
        Calmar Ratio: {metrics['calmar_ratio']:.2f}
        
        TRADING METRICS
        ---------------
        Total Trades: {metrics['total_trades']}
        Win Rate: {metrics['win_rate']:.2%}
        Average Win: {metrics['avg_win']:.2%}
        Average Loss: {metrics['avg_loss']:.2%}
        Profit Factor: {metrics['profit_factor']:.2f}
        
        ACCURACY METRICS
        ----------------
        Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}
        Precision: {metrics.get('precision', 0):.2%}
        Recall: {metrics.get('recall', 0):.2%}
        F1 Score: {metrics.get('f1_score', 0):.2f}
        
        RISK MEASURES
        -------------
        VaR (95%): {metrics['var_95']:.2%}
        CVaR (95%): {metrics['cvar_95']:.2%}
        Information Ratio: {metrics['information_ratio']:.2f}
        """
        
        return report

class WalkForwardBacktest:
    """Walk-forward backtesting for time series data"""
    
    def __init__(self, train_period=252, test_period=63, step_size=21):
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.results = []
    
    def run_walk_forward(self, data: pd.DataFrame, model, feature_columns: List[str]):
        """Run walk-forward backtest"""
        results = []
        
        for i in range(self.train_period, len(data) - self.test_period, self.step_size):
            # Define train and test periods
            train_start = i - self.train_period
            train_end = i
            test_start = i
            test_end = min(i + self.test_period, len(data))
            
            # Prepare data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            X_train = train_data[feature_columns]
            y_train = train_data['target'] if 'target' in train_data.columns else train_data['returns']
            
            model.fit(X_train, y_train)
            
            # Make predictions
            X_test = test_data[feature_columns]
            predictions = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            
            # Run backtest
            backtest_engine = BacktestEngine()
            backtest_results = backtest_engine.run_backtest(
                test_data[['Open', 'High', 'Low', 'Close', 'Volume']],
                pd.Series(predictions, index=test_data.index)
            )
            
            # Store results
            results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'metrics': backtest_results['metrics']
            })
        
        self.results = results
        return results
    
    def get_aggregated_metrics(self) -> Dict:
        """Get aggregated metrics across all walk-forward periods"""
        if not self.results:
            return {}
        
        # Aggregate metrics
        metrics_df = pd.DataFrame([r['metrics'] for r in self.results])
        
        aggregated = {
            'avg_total_return': metrics_df['total_return'].mean(),
            'avg_sharpe_ratio': metrics_df['sharpe_ratio'].mean(),
            'avg_max_drawdown': metrics_df['max_drawdown'].mean(),
            'avg_directional_accuracy': metrics_df.get('directional_accuracy', pd.Series([0])).mean(),
            'consistency_score': (metrics_df['total_return'] > 0).mean(),
            'total_periods': len(self.results)
        }
        
        return aggregated
