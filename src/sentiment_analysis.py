import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Comprehensive sentiment analysis from news and social media sources"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'https://feeds.reuters.com/news/wealth',
            'https://feeds.bloomberg.com/markets/news.rss'
        ]
        self.sentiment_cache = {}
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a single text using multiple methods"""
        if not text or pd.isna(text):
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0,
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'sentiment_score': 0.0
            }
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Combined sentiment score (weighted average)
        sentiment_score = (
            0.6 * vader_scores['compound'] + 
            0.4 * textblob_polarity
        )
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'sentiment_score': sentiment_score
        }
    
    def get_news_sentiment(self, symbol, days_back=30):
        """Get news sentiment for a specific symbol"""
        cache_key = f"{symbol}_{days_back}_{datetime.now().strftime('%Y%m%d')}"
        
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        all_news = []
        
        # Yahoo Finance news
        try:
            yahoo_news = self._get_yahoo_news(symbol, days_back)
            all_news.extend(yahoo_news)
        except Exception as e:
            print(f"Error fetching Yahoo news for {symbol}: {e}")
        
        # MarketWatch news
        try:
            mw_news = self._get_marketwatch_news(symbol, days_back)
            all_news.extend(mw_news)
        except Exception as e:
            print(f"Error fetching MarketWatch news for {symbol}: {e}")
        
        # Analyze sentiment for all news
        sentiment_data = []
        for news_item in all_news:
            sentiment = self.analyze_text_sentiment(news_item['title'] + ' ' + news_item.get('summary', ''))
            sentiment['date'] = news_item['date']
            sentiment['source'] = news_item['source']
            sentiment_data.append(sentiment)
        
        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiment_data)
        
        if len(sentiment_df) > 0:
            # Aggregate by date
            daily_sentiment = sentiment_df.groupby('date').agg({
                'vader_compound': 'mean',
                'vader_positive': 'mean',
                'vader_negative': 'mean',
                'vader_neutral': 'mean',
                'textblob_polarity': 'mean',
                'textblob_subjectivity': 'mean',
                'sentiment_score': 'mean'
            }).reset_index()
            
            # Add additional metrics
            daily_sentiment['news_count'] = sentiment_df.groupby('date').size().values
            daily_sentiment['sentiment_volatility'] = sentiment_df.groupby('date')['sentiment_score'].std().values
            daily_sentiment['positive_ratio'] = (sentiment_df['sentiment_score'] > 0.1).groupby(sentiment_df['date']).mean().values
            daily_sentiment['negative_ratio'] = (sentiment_df['sentiment_score'] < -0.1).groupby(sentiment_df['date']).mean().values
        else:
            # Return empty DataFrame with proper structure
            daily_sentiment = pd.DataFrame(columns=[
                'date', 'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
                'textblob_polarity', 'textblob_subjectivity', 'sentiment_score',
                'news_count', 'sentiment_volatility', 'positive_ratio', 'negative_ratio'
            ])
        
        # Cache the result
        self.sentiment_cache[cache_key] = daily_sentiment
        
        return daily_sentiment
    
    def _get_yahoo_news(self, symbol, days_back):
        """Fetch news from Yahoo Finance"""
        news_data = []
        
        try:
            # Get news from yfinance
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in news:
                if 'providerPublishTime' in item:
                    news_date = datetime.fromtimestamp(item['providerPublishTime'])
                    if news_date >= cutoff_date:
                        news_data.append({
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'date': news_date.date(),
                            'source': 'Yahoo Finance'
                        })
        except Exception as e:
            print(f"Error fetching Yahoo news: {e}")
        
        return news_data
    
    def _get_marketwatch_news(self, symbol, days_back):
        """Fetch news from MarketWatch (simplified)"""
        news_data = []
        
        try:
            # This is a simplified implementation
            # In practice, you'd use MarketWatch's API or web scraping
            url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract news headlines (this is simplified)
                headlines = soup.find_all('h3', class_='article__headline')
                
                for headline in headlines[:10]:  # Limit to 10 headlines
                    title = headline.get_text(strip=True)
                    if title:
                        news_data.append({
                            'title': title,
                            'summary': '',
                            'date': datetime.now().date(),
                            'source': 'MarketWatch'
                        })
        except Exception as e:
            print(f"Error fetching MarketWatch news: {e}")
        
        return news_data
    
    def get_social_media_sentiment(self, symbol, platform='twitter', days_back=7):
        """Get social media sentiment (placeholder for real implementation)"""
        # This is a placeholder - in practice, you'd integrate with:
        # - Twitter API
        # - Reddit API
        # - StockTwits API
        # - Discord/Telegram channels
        
        # For now, return mock data
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        
        mock_sentiment = pd.DataFrame({
            'date': dates.date,
            'social_sentiment_score': np.random.normal(0, 0.3, days_back),
            'social_mention_count': np.random.poisson(50, days_back),
            'social_engagement': np.random.uniform(0.1, 0.9, days_back)
        })
        
        return mock_sentiment
    
    def create_sentiment_features(self, symbol, days_back=30):
        """Create comprehensive sentiment features for a symbol"""
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(symbol, days_back)
        
        # Get social media sentiment
        social_sentiment = self.get_social_media_sentiment(symbol, days_back=days_back)
        
        # Merge sentiment data
        sentiment_df = pd.merge(
            news_sentiment, 
            social_sentiment, 
            on='date', 
            how='outer'
        ).fillna(0)
        
        # Create additional features
        sentiment_df['combined_sentiment'] = (
            0.7 * sentiment_df['sentiment_score'] + 
            0.3 * sentiment_df['social_sentiment_score']
        )
        
        sentiment_df['sentiment_momentum'] = sentiment_df['combined_sentiment'].rolling(5).mean()
        sentiment_df['sentiment_acceleration'] = sentiment_df['sentiment_momentum'].diff()
        
        sentiment_df['news_sentiment_volatility'] = sentiment_df['sentiment_score'].rolling(10).std()
        sentiment_df['social_sentiment_volatility'] = sentiment_df['social_sentiment_score'].rolling(10).std()
        
        # Sentiment regime classification
        sentiment_df['sentiment_regime'] = pd.cut(
            sentiment_df['combined_sentiment'],
            bins=[-np.inf, -0.2, 0.2, np.inf],
            labels=['negative', 'neutral', 'positive']
        )
        
        return sentiment_df
    
    def get_market_sentiment_indicators(self, symbols, days_back=30):
        """Get market-wide sentiment indicators"""
        all_sentiment = []
        
        for symbol in symbols[:10]:  # Limit to first 10 symbols for demo
            try:
                sentiment = self.create_sentiment_features(symbol, days_back)
                sentiment['symbol'] = symbol
                all_sentiment.append(sentiment)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error processing sentiment for {symbol}: {e}")
        
        if all_sentiment:
            market_sentiment = pd.concat(all_sentiment, ignore_index=True)
            
            # Aggregate market-wide metrics
            market_metrics = market_sentiment.groupby('date').agg({
                'combined_sentiment': ['mean', 'std'],
                'news_count': 'sum',
                'social_mention_count': 'sum',
                'positive_ratio': 'mean',
                'negative_ratio': 'mean'
            }).reset_index()
            
            # Flatten column names
            market_metrics.columns = ['_'.join(col).strip() for col in market_metrics.columns]
            market_metrics = market_metrics.rename(columns={'date_': 'date'})
            
            return market_metrics
        else:
            return pd.DataFrame()

class SentimentFeatureEngineer:
    """Feature engineering specifically for sentiment data"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def create_sentiment_features_for_symbol(self, symbol, price_data, days_back=30):
        """Create sentiment features aligned with price data"""
        # Get sentiment data
        sentiment_data = self.sentiment_analyzer.create_sentiment_features(symbol, days_back)
        
        if sentiment_data.empty:
            # Return empty features if no sentiment data
            return pd.DataFrame(index=price_data.index)
        
        # Align sentiment data with price data
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        sentiment_data = sentiment_data.set_index('date')
        
        # Ensure timezone consistency
        if price_data.index.tz is not None:
            sentiment_data.index = sentiment_data.index.tz_localize('UTC')
        if sentiment_data.index.tz is not None and price_data.index.tz is None:
            price_data.index = price_data.index.tz_localize('UTC')
        
        # Reindex to match price data frequency
        sentiment_aligned = sentiment_data.reindex(price_data.index, method='ffill')
        
        # Create additional sentiment features
        features = pd.DataFrame(index=price_data.index)
        
        # Basic sentiment features
        sentiment_cols = [
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'textblob_polarity', 'textblob_subjectivity', 'sentiment_score',
            'social_sentiment_score', 'combined_sentiment'
        ]
        
        for col in sentiment_cols:
            if col in sentiment_aligned.columns:
                features[f'sentiment_{col}'] = sentiment_aligned[col]
                
                # Add rolling statistics
                for window in [5, 10, 20]:
                    features[f'sentiment_{col}_ma_{window}'] = sentiment_aligned[col].rolling(window).mean()
                    features[f'sentiment_{col}_std_{window}'] = sentiment_aligned[col].rolling(window).std()
        
        # Sentiment momentum and acceleration
        if 'combined_sentiment' in sentiment_aligned.columns:
            features['sentiment_momentum_5d'] = sentiment_aligned['combined_sentiment'].rolling(5).mean()
            features['sentiment_acceleration'] = features['sentiment_momentum_5d'].diff()
            
            # Sentiment regime features
            features['sentiment_regime_positive'] = (sentiment_aligned['combined_sentiment'] > 0.1).astype(int)
            features['sentiment_regime_negative'] = (sentiment_aligned['combined_sentiment'] < -0.1).astype(int)
        
        # News and social media activity
        activity_cols = ['news_count', 'social_mention_count', 'positive_ratio', 'negative_ratio']
        for col in activity_cols:
            if col in sentiment_aligned.columns:
                features[f'sentiment_{col}'] = sentiment_aligned[col]
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
