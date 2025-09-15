import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import yfinance as yf
import requests
import logging
from dataclasses import dataclass, asdict
import queue
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure for Kafka messages"""
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str = "yfinance"
    
    def to_dict(self):
        return asdict(self)

class KafkaDataProducer:
    """Kafka producer for real-time market data ingestion"""
    
    def __init__(self, 
                 bootstrap_servers: List[str] = ['localhost:9092'],
                 topic: str = 'market_data',
                 batch_size: int = 1000,
                 linger_ms: int = 100):
        
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.batch_size = batch_size
        self.producer = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=10000)
        
        # Initialize producer
        self._init_producer()
    
    def _init_producer(self):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                batch_size=self.batch_size,
                linger_ms=linger_ms,
                acks='all',
                retries=3,
                retry_backoff_ms=100
            )
            logger.info(f"Kafka producer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def add_data(self, market_data: MarketData):
        """Add market data to the queue"""
        try:
            self.data_queue.put_nowait(market_data.to_dict())
        except queue.Full:
            logger.warning("Data queue is full, dropping oldest data")
            try:
                self.data_queue.get_nowait()  # Remove oldest
                self.data_queue.put_nowait(market_data.to_dict())
            except queue.Empty:
                pass
    
    def start_producer(self):
        """Start the producer thread"""
        if self.running:
            return
        
        self.running = True
        self.producer_thread = threading.Thread(target=self._producer_loop)
        self.producer_thread.daemon = True
        self.producer_thread.start()
        logger.info("Kafka producer started")
    
    def stop_producer(self):
        """Stop the producer thread"""
        self.running = False
        if hasattr(self, 'producer_thread'):
            self.producer_thread.join()
        logger.info("Kafka producer stopped")
    
    def _producer_loop(self):
        """Main producer loop"""
        batch = []
        
        while self.running:
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        data = self.data_queue.get(timeout=1)
                        batch.append(data)
                    except queue.Empty:
                        break
                
                # Send batch
                if batch:
                    self._send_batch(batch)
                    batch = []
                
            except Exception as e:
                logger.error(f"Error in producer loop: {e}")
                time.sleep(1)
    
    def _send_batch(self, batch: List[Dict]):
        """Send a batch of data to Kafka"""
        try:
            for data in batch:
                self.producer.send(
                    self.topic,
                    value=data,
                    key=data['symbol'].encode('utf-8')
                )
            
            self.producer.flush()
            logger.info(f"Sent batch of {len(batch)} messages to {self.topic}")
            
        except Exception as e:
            logger.error(f"Failed to send batch: {e}")

class MarketDataCollector:
    """Collect market data from various sources and send to Kafka"""
    
    def __init__(self, 
                 symbols: List[str],
                 producer: KafkaDataProducer,
                 update_frequency: int = 60):  # seconds
        
        self.symbols = symbols
        self.producer = producer
        self.update_frequency = update_frequency
        self.running = False
        self.collector_threads = []
        
        # Data sources
        self.data_sources = {
            'yfinance': self._collect_yfinance_data,
            'alpha_vantage': self._collect_alpha_vantage_data,
            'mock_data': self._generate_mock_data
        }
    
    def start_collection(self, source: str = 'yfinance'):
        """Start data collection from specified source"""
        if self.running:
            return
        
        self.running = True
        
        # Start producer
        self.producer.start_producer()
        
        # Start collector threads for each symbol
        for symbol in self.symbols:
            thread = threading.Thread(
                target=self._collect_loop,
                args=(symbol, source)
            )
            thread.daemon = True
            thread.start()
            self.collector_threads.append(thread)
        
        logger.info(f"Started data collection for {len(self.symbols)} symbols from {source}")
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.collector_threads:
            thread.join(timeout=5)
        
        # Stop producer
        self.producer.stop_producer()
        
        logger.info("Data collection stopped")
    
    def _collect_loop(self, symbol: str, source: str):
        """Main collection loop for a symbol"""
        while self.running:
            try:
                data = self.data_sources[source](symbol)
                if data:
                    self.producer.add_data(data)
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                time.sleep(self.update_frequency)
    
    def _collect_yfinance_data(self, symbol: str) -> Optional[MarketData]:
        """Collect data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                open=float(latest['Open']),
                high=float(latest['High']),
                low=float(latest['Low']),
                close=float(latest['Close']),
                volume=int(latest['Volume']),
                source="yfinance"
            )
            
        except Exception as e:
            logger.error(f"Error collecting yfinance data for {symbol}: {e}")
            return None
    
    def _collect_alpha_vantage_data(self, symbol: str) -> Optional[MarketData]:
        """Collect data from Alpha Vantage (placeholder)"""
        # This would require Alpha Vantage API key
        # For now, return None
        return None
    
    def _generate_mock_data(self, symbol: str) -> MarketData:
        """Generate mock market data for testing"""
        # Generate realistic price movement
        base_price = 100 + hash(symbol) % 1000
        price_change = np.random.normal(0, 0.02) * base_price
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            open=base_price,
            high=base_price + abs(price_change) + np.random.uniform(0, 5),
            low=base_price - abs(price_change) - np.random.uniform(0, 5),
            close=base_price + price_change,
            volume=np.random.randint(1000, 100000),
            source="mock"
        )

class KafkaDataConsumer:
    """Kafka consumer for processing real-time market data"""
    
    def __init__(self,
                 bootstrap_servers: List[str] = ['localhost:9092'],
                 topics: List[str] = ['market_data'],
                 group_id: str = 'market_data_processor',
                 auto_offset_reset: str = 'latest'):
        
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer = None
        self.running = False
        self.processors = []
        
        # Initialize consumer
        self._init_consumer()
    
    def _init_consumer(self):
        """Initialize Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            logger.info(f"Kafka consumer initialized for topics: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def add_processor(self, processor: Callable[[Dict], None]):
        """Add a data processor function"""
        self.processors.append(processor)
    
    def start_consuming(self):
        """Start consuming messages"""
        if self.running:
            return
        
        self.running = True
        self.consumer_thread = threading.Thread(target=self._consume_loop)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        logger.info("Kafka consumer started")
    
    def stop_consuming(self):
        """Stop consuming messages"""
        self.running = False
        if hasattr(self, 'consumer_thread'):
            self.consumer_thread.join()
        logger.info("Kafka consumer stopped")
    
    def _consume_loop(self):
        """Main consumption loop"""
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    data = message.value
                    self._process_message(data)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
    
    def _process_message(self, data: Dict):
        """Process a single message"""
        for processor in self.processors:
            try:
                processor(data)
            except Exception as e:
                logger.error(f"Error in processor: {e}")

class RealTimeDataProcessor:
    """Process real-time market data for ML predictions"""
    
    def __init__(self, 
                 feature_engineer,
                 model,
                 prediction_topic: str = 'predictions',
                 producer: Optional[KafkaDataProducer] = None):
        
        self.feature_engineer = feature_engineer
        self.model = model
        self.prediction_topic = prediction_topic
        self.producer = producer
        
        # Data buffers for each symbol
        self.data_buffers = {}
        self.buffer_size = 1000  # Keep last 1000 data points per symbol
    
    def process_market_data(self, data: Dict):
        """Process incoming market data"""
        symbol = data['symbol']
        
        # Add to buffer
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = []
        
        self.data_buffers[symbol].append(data)
        
        # Keep only recent data
        if len(self.data_buffers[symbol]) > self.buffer_size:
            self.data_buffers[symbol] = self.data_buffers[symbol][-self.buffer_size:]
        
        # Generate features and prediction if we have enough data
        if len(self.data_buffers[symbol]) >= 50:  # Minimum data for features
            self._generate_prediction(symbol)
    
    def _generate_prediction(self, symbol: str):
        """Generate prediction for a symbol"""
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffers[symbol])
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
            
            # Generate features
            features_df = self.feature_engineer.create_all_features(df)
            
            # Get latest features
            latest_features = features_df.iloc[-1:].drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            
            # Make prediction
            prediction = self.model.predict_proba(latest_features)[0][1]
            
            # Create prediction message
            prediction_data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'prediction': float(prediction),
                'confidence': float(abs(prediction - 0.5) * 2),
                'signal': 'BUY' if prediction > 0.6 else 'SELL' if prediction < 0.4 else 'HOLD'
            }
            
            # Send prediction to Kafka
            if self.producer:
                self.producer.add_data(prediction_data)
            
            logger.info(f"Generated prediction for {symbol}: {prediction_data['signal']} (confidence: {prediction_data['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")

class KafkaDataManager:
    """Main class for managing Kafka data flow"""
    
    def __init__(self, 
                 symbols: List[str],
                 bootstrap_servers: List[str] = ['localhost:9092'],
                 market_data_topic: str = 'market_data',
                 prediction_topic: str = 'predictions'):
        
        self.symbols = symbols
        self.bootstrap_servers = bootstrap_servers
        
        # Initialize producers and consumers
        self.market_data_producer = KafkaDataProducer(
            bootstrap_servers=bootstrap_servers,
            topic=market_data_topic
        )
        
        self.prediction_producer = KafkaDataProducer(
            bootstrap_servers=bootstrap_servers,
            topic=prediction_topic
        )
        
        self.consumer = KafkaDataConsumer(
            bootstrap_servers=bootstrap_servers,
            topics=[market_data_topic]
        )
        
        # Data collector
        self.collector = MarketDataCollector(
            symbols=symbols,
            producer=self.market_data_producer
        )
    
    def start_data_pipeline(self, feature_engineer, model, source: str = 'yfinance'):
        """Start the complete data pipeline"""
        # Initialize real-time processor
        processor = RealTimeDataProcessor(
            feature_engineer=feature_engineer,
            model=model,
            producer=self.prediction_producer
        )
        
        # Add processor to consumer
        self.consumer.add_processor(processor.process_market_data)
        
        # Start all components
        self.collector.start_collection(source=source)
        self.consumer.start_consuming()
        
        logger.info("Data pipeline started")
    
    def stop_data_pipeline(self):
        """Stop the complete data pipeline"""
        self.collector.stop_collection()
        self.consumer.stop_consuming()
        logger.info("Data pipeline stopped")
    
    def get_data_stats(self) -> Dict:
        """Get statistics about data flow"""
        stats = {
            'symbols': len(self.symbols),
            'buffer_sizes': {symbol: len(buffer) for symbol, buffer in 
                           self.collector.data_buffers.items()},
            'total_messages_processed': sum(len(buffer) for buffer in 
                                          self.collector.data_buffers.values())
        }
        return stats
