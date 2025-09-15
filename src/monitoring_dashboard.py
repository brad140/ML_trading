from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import redis
import psycopg2
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import plotly.graph_objs as go
import plotly.utils
from kafka import KafkaConsumer
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
REDIS_URL = "redis://localhost:6379"
DATABASE_URL = "postgresql://trading_user:trading_password@localhost:5432/trading_db"
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']

class MonitoringDashboard:
    """Real-time monitoring dashboard for ML trading system"""
    
    def __init__(self):
        self.redis_client = redis.from_url(REDIS_URL)
        self.db_connection = None
        self.kafka_consumer = None
        self.metrics_cache = {}
        self.prediction_cache = {}
        self.performance_metrics = {}
        
        # Initialize connections
        self._init_database()
        self._init_kafka_consumer()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_database(self):
        """Initialize database connection"""
        try:
            self.db_connection = psycopg2.connect(DATABASE_URL)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
    
    def _init_kafka_consumer(self):
        """Initialize Kafka consumer for real-time data"""
        try:
            self.kafka_consumer = KafkaConsumer(
                'predictions',
                'market_data',
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info("Kafka consumer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks for data processing"""
        # Start Kafka consumer thread
        kafka_thread = threading.Thread(target=self._consume_kafka_messages)
        kafka_thread.daemon = True
        kafka_thread.start()
        
        # Start metrics calculation thread
        metrics_thread = threading.Thread(target=self._calculate_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
    
    def _consume_kafka_messages(self):
        """Consume messages from Kafka"""
        if not self.kafka_consumer:
            return
        
        for message in self.kafka_consumer:
            try:
                topic = message.topic
                data = message.value
                
                if topic == 'predictions':
                    self._process_prediction(data)
                elif topic == 'market_data':
                    self._process_market_data(data)
                    
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")
    
    def _process_prediction(self, data: Dict):
        """Process prediction data"""
        symbol = data.get('symbol')
        if symbol:
            self.prediction_cache[symbol] = {
                'timestamp': data.get('timestamp'),
                'prediction': data.get('prediction'),
                'confidence': data.get('confidence'),
                'signal': data.get('signal')
            }
            
            # Store in Redis for quick access
            self.redis_client.setex(
                f"prediction:{symbol}", 
                300,  # 5 minutes TTL
                json.dumps(self.prediction_cache[symbol])
            )
    
    def _process_market_data(self, data: Dict):
        """Process market data"""
        symbol = data.get('symbol')
        if symbol:
            # Store latest market data
            self.redis_client.setex(
                f"market_data:{symbol}",
                60,  # 1 minute TTL
                json.dumps(data)
            )
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        while True:
            try:
                # Calculate accuracy metrics
                self._calculate_accuracy_metrics()
                
                # Calculate performance metrics
                self._calculate_performance_metrics()
                
                # Calculate system metrics
                self._calculate_system_metrics()
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                time.sleep(60)
    
    def _calculate_accuracy_metrics(self):
        """Calculate prediction accuracy metrics"""
        try:
            # Get recent predictions from database
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT symbol, prediction, actual_return, timestamp
                    FROM predictions 
                    WHERE timestamp > NOW() - INTERVAL '1 day'
                    ORDER BY timestamp DESC
                """)
                
                results = cursor.fetchall()
                
                if results:
                    df = pd.DataFrame(results, columns=['symbol', 'prediction', 'actual_return', 'timestamp'])
                    df['correct'] = (df['prediction'] > 0.5) == (df['actual_return'] > 0)
                    
                    accuracy_by_symbol = df.groupby('symbol')['correct'].mean()
                    overall_accuracy = df['correct'].mean()
                    
                    self.metrics_cache['accuracy'] = {
                        'overall': float(overall_accuracy),
                        'by_symbol': accuracy_by_symbol.to_dict(),
                        'total_predictions': len(df)
                    }
                
                cursor.close()
                
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
    
    def _calculate_performance_metrics(self):
        """Calculate trading performance metrics"""
        try:
            # Get portfolio performance from Redis or database
            portfolio_data = self.redis_client.get('portfolio_performance')
            if portfolio_data:
                self.performance_metrics = json.loads(portfolio_data)
            else:
                # Mock performance data
                self.performance_metrics = {
                    'total_return': 0.15,
                    'sharpe_ratio': 2.3,
                    'max_drawdown': -0.08,
                    'win_rate': 0.67,
                    'total_trades': 1250,
                    'profit_factor': 1.85
                }
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    def _calculate_system_metrics(self):
        """Calculate system performance metrics"""
        try:
            # Get system metrics from Redis
            system_metrics = {
                'kafka_messages_processed': int(self.redis_client.get('kafka_messages_processed') or 0),
                'active_models': int(self.redis_client.get('active_models') or 0),
                'prediction_latency': float(self.redis_client.get('prediction_latency') or 0),
                'memory_usage': float(self.redis_client.get('memory_usage') or 0),
                'cpu_usage': float(self.redis_client.get('cpu_usage') or 0)
            }
            
            self.metrics_cache['system'] = system_metrics
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")
    
    def get_dashboard_data(self) -> Dict:
        """Get all dashboard data"""
        return {
            'predictions': self.prediction_cache,
            'metrics': self.metrics_cache,
            'performance': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_prediction_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get prediction history for a symbol"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    SELECT timestamp, prediction, confidence, signal
                    FROM predictions 
                    WHERE symbol = %s AND timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                """, (symbol, hours))
                
                results = cursor.fetchall()
                cursor.close()
                
                return [
                    {
                        'timestamp': row[0].isoformat(),
                        'prediction': float(row[1]),
                        'confidence': float(row[2]),
                        'signal': row[3]
                    }
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
        
        return []
    
    def get_performance_chart_data(self, days: int = 30) -> Dict:
        """Get performance chart data"""
        try:
            # Mock performance data - in practice, get from database
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            # Generate realistic performance data
            np.random.seed(42)
            returns = np.random.normal(0.001, 0.02, days)
            cumulative_returns = np.cumprod(1 + returns) - 1
            
            # Generate benchmark data
            benchmark_returns = np.random.normal(0.0008, 0.015, days)
            benchmark_cumulative = np.cumprod(1 + benchmark_returns) - 1
            
            return {
                'dates': [d.isoformat() for d in dates],
                'strategy_returns': cumulative_returns.tolist(),
                'benchmark_returns': benchmark_cumulative.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance chart data: {e}")
            return {'dates': [], 'strategy_returns': [], 'benchmark_returns': []}

# Initialize dashboard
dashboard = MonitoringDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def get_dashboard():
    """Get dashboard data API"""
    return jsonify(dashboard.get_dashboard_data())

@app.route('/api/predictions/<symbol>')
def get_predictions(symbol):
    """Get predictions for a specific symbol"""
    hours = request.args.get('hours', 24, type=int)
    history = dashboard.get_prediction_history(symbol, hours)
    return jsonify(history)

@app.route('/api/performance')
def get_performance():
    """Get performance data"""
    days = request.args.get('days', 30, type=int)
    chart_data = dashboard.get_performance_chart_data(days)
    return jsonify(chart_data)

@app.route('/api/metrics')
def get_metrics():
    """Get system metrics"""
    return jsonify(dashboard.metrics_cache)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'redis': dashboard.redis_client.ping(),
            'database': dashboard.db_connection is not None,
            'kafka': dashboard.kafka_consumer is not None
        }
    })

@app.route('/api/symbols')
def get_symbols():
    """Get list of available symbols"""
    symbols = list(dashboard.prediction_cache.keys())
    return jsonify(symbols)

@app.route('/api/predictions/current')
def get_current_predictions():
    """Get current predictions for all symbols"""
    return jsonify(dashboard.prediction_cache)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
