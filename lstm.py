import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import schedule
import time

# Configuration
OANDA_API_KEY = 'your_oanda_api_key'
NEWS_API_KEY = 'your_newsapi_key'
INSTRUMENT = 'EUR_USD'
TIMEFRAME = 'M1'  # 1-minute timeframe

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def fetch_price_data():
    """Fetch high-frequency EUR/USD data from OANDA API"""
    url = f'https://api-fxpractice.oanda.com/v3/instruments/{INSTRUMENT}/candles'
    headers = {'Authorization': f'Bearer {OANDA_API_KEY}'}
    params = {
        'granularity': TIMEFRAME,
        'count': 500  # Get last 500 data points
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()['candles']
    
    df = pd.DataFrame([{
        'time': c['time'],
        'open': float(c['mid']['o']),
        'high': float(c['mid']['h']),
        'low': float(c['mid']['l']),
        'close': float(c['mid']['c'])
    } for c in data])
    
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')

def fetch_sentiment():
    """Fetch news sentiment data"""
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'EUR/USD OR Euro OR ECB',
        'apiKey': NEWS_API_KEY,
        'pageSize': 100,
        'sortBy': 'publishedAt'
    }
    
    response = requests.get(url, params=params)
    articles = response.json().get('articles', [])
    
    sentiments = []
    for article in articles:
        text = f"{article['title']} {article['description']}"
        score = sia.polarity_scores(text)['compound']  # VADER sentiment score
        sentiments.append({
            'time': pd.to_datetime(article['publishedAt']),
            'sentiment': score
        })
    
    return pd.DataFrame(sentiments).set_index('time')

def preprocess_data(price_df, sentiment_df):
    """Merge and preprocess data"""
    # Resample sentiment data to 1-minute intervals
    sentiment_df = sentiment_df.resample('1T').mean().ffill()
    
    # Merge datasets
    merged_df = price_df.join(sentiment_df, how='left').ffill()
    
    # Feature engineering
    merged_df['returns'] = merged_df['close'].pct_change()
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_df[['close', 'sentiment']])
    
    return scaler, scaled_data

def create_sequences(data, window_size=60):
    """Create LSTM sequences"""
    X, y = [], []
    for i in range(len(data)-window_size-1):
        X.append(data[i:(i+window_size)])
        y.append(data[i+window_size, 0])  # Predict next close price
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X_train, y_train):
    """Train LSTM model"""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model

def trading_strategy(prediction, current_price):
    """Simple trading strategy"""
    if prediction > current_price * 1.0005:  # 0.05% threshold
        return 'BUY'
    elif prediction < current_price * 0.9995:
        return 'SELL'
    else:
        return 'HOLD'

def main():
    # Fetch data
    price_data = fetch_price_data()
    sentiment_data = fetch_sentiment()
    
    # Preprocess data
    scaler, processed_data = preprocess_data(price_data, sentiment_data)
    
    # Prepare LSTM data
    window_size = 60
    X, y = create_sequences(processed_data, window_size)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make prediction
    last_sequence = processed_data[-window_size:]
    last_sequence = last_sequence.reshape(1, window_size, 2)
    predicted_scaled = model.predict(last_sequence)
    
    # Inverse transform prediction
    predicted_price = scaler.inverse_transform(
        np.concatenate([predicted_scaled, np.zeros((1, 1))], axis=1)
    )[:, 0][0]
    
    # Get current price
    current_price = price_data['close'].iloc[-1]
    
    # Generate trading signal
    signal = trading_strategy(predicted_price, current_price)
    print(f"Predicted Price: {predicted_price:.5f}")
    print(f"Current Price: {current_price:.5f}")
    print(f"Signal: {signal}")

if __name__ == '__main__':
    # Schedule to run every minute
    schedule.every(1).minutes.do(main)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
