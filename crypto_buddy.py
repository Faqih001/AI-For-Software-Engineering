#!/usr/bin/env python3
# CryptoBuddy Chatbot with CoinGecko API and NLTK Integration
# A rule-based cryptocurrency advisor with NLP for natural user queries.

import sys
import time
import traceback
import random
import json
import os
import datetime
import re
from threading import Thread

# Try importing required modules with better error handling
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet
    from nltk.tag import pos_tag
    from pycoingecko import CoinGeckoAPI
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you've installed the required packages using:")
    print("pip install nltk pycoingecko")
    sys.exit(1)

class PriceAlert:
    """A class to manage cryptocurrency price alerts for users."""
    
    def __init__(self, coin_id, target_price, alert_type='above', username='default'):
        """Initialize a price alert.
        
        Args:
            coin_id (str): The CoinGecko ID of the cryptocurrency
            target_price (float): The target price for the alert
            alert_type (str): 'above' or 'below' to trigger when price crosses target
            username (str): The username associated with this alert
        """
        self.coin_id = coin_id
        self.target_price = float(target_price)
        self.alert_type = alert_type
        self.username = username
        self.created_at = datetime.datetime.now().isoformat()
        self.triggered = False
        self.triggered_at = None
        
    def check_condition(self, current_price):
        """Check if the alert condition is met.
        
        Args:
            current_price (float): The current price of the cryptocurrency
            
        Returns:
            bool: True if the alert condition is met, False otherwise
        """
        if not self.triggered:
            if self.alert_type == 'above' and current_price >= self.target_price:
                return True
            elif self.alert_type == 'below' and current_price <= self.target_price:
                return True
        return False
        
    def trigger(self):
        """Mark the alert as triggered."""
        self.triggered = True
        self.triggered_at = datetime.datetime.now().isoformat()
        
    def to_dict(self):
        """Convert the alert to a dictionary for serialization.
        
        Returns:
            dict: A dictionary representation of the alert
        """
        return {
            'coin_id': self.coin_id,
            'target_price': self.target_price,
            'alert_type': self.alert_type,
            'username': self.username,
            'created_at': self.created_at,
            'triggered': self.triggered,
            'triggered_at': self.triggered_at
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create an alert from a dictionary.
        
        Args:
            data (dict): A dictionary representation of the alert
            
        Returns:
            PriceAlert: A PriceAlert instance
        """
        alert = cls(
            data['coin_id'],
            data['target_price'],
            data['alert_type'],
            data['username']
        )
        alert.created_at = data['created_at']
        alert.triggered = data['triggered']
        alert.triggered_at = data['triggered_at']
        return alert

class AlertManager:
    """A class to manage multiple price alerts."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.alerts = []
        self.running = False
        self.check_interval = 60  # seconds between price checks
        self.last_price_check = {}  # cache of last price check
        
    def add_alert(self, alert):
        """Add a price alert.
        
        Args:
            alert (PriceAlert): The alert to add
        """
        self.alerts.append(alert)
        self.save_alerts()
        
    def remove_alert(self, alert_index):
        """Remove a price alert.
        
        Args:
            alert_index (int): The index of the alert to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            del self.alerts[alert_index]
            self.save_alerts()
            return True
        except IndexError:
            return False
            
    def get_alerts_by_username(self, username):
        """Get all alerts for a specific user.
        
        Args:
            username (str): The username to filter by
            
        Returns:
            list: A list of alerts for the user
        """
        return [alert for alert in self.alerts if alert.username == username]
    
    def save_alerts(self):
        """Save all alerts to a file."""
        try:
            os.makedirs('alerts', exist_ok=True)
            
            alerts_data = [alert.to_dict() for alert in self.alerts]
            
            with open('alerts/alerts.json', 'w') as f:
                json.dump(alerts_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving alerts: {e}")
            return False
    
    def load_alerts(self):
        """Load alerts from a file."""
        try:
            if not os.path.exists('alerts/alerts.json'):
                return
                
            with open('alerts/alerts.json', 'r') as f:
                alerts_data = json.load(f)
                
            self.alerts = [PriceAlert.from_dict(data) for data in alerts_data]
        except Exception as e:
            print(f"Error loading alerts: {e}")
    
    def check_alerts(self):
        """Check all active alerts against current prices."""
        try:
            # Only check prices for coins with active alerts
            coins_to_check = set(alert.coin_id for alert in self.alerts if not alert.triggered)
            if not coins_to_check:
                return []
                
            cg = CoinGeckoAPI()
            prices = cg.get_price(ids=list(coins_to_check), vs_currencies='usd')
            
            triggered_alerts = []
            
            for i, alert in enumerate(self.alerts):
                if alert.triggered:
                    continue
                    
                if alert.coin_id in prices:
                    current_price = prices[alert.coin_id]['usd']
                    self.last_price_check[alert.coin_id] = current_price
                    
                    if alert.check_condition(current_price):
                        alert.trigger()
                        triggered_alerts.append((i, alert, current_price))
            
            if triggered_alerts:
                self.save_alerts()
                
            return triggered_alerts
        except Exception as e:
            print(f"Error checking alerts: {e}")
            return []
    
    def start_monitoring(self):
        """Start the alert monitoring process in a background thread."""
        if self.running:
            return False
            
        self.running = True
        self.monitor_thread = Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return True
        
    def stop_monitoring(self):
        """Stop the alert monitoring process."""
        self.running = False
        
    def _monitoring_loop(self):
        """The main monitoring loop that runs in a background thread."""
        while self.running:
            triggered = self.check_alerts()
            
            for _, alert, price in triggered:
                direction = "above" if alert.alert_type == "above" else "below"
                print(f"🔔 ALERT: {alert.coin_id.capitalize()} price is now {direction} ${alert.target_price} (Current: ${price})")
                
            time.sleep(self.check_interval)

def setup_dependencies():
    """Install and download required dependencies."""
    try:
        # Import required modules, install if missing
        try:
            import pycoingecko
            import nltk
        except ImportError:
            import subprocess
            print("Installing required packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycoingecko", "nltk"])
            
        # Set up the NLTK data path to use a local directory
        import os
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)
        
        # Download required NLTK data with SSL workaround
        print("Downloading NLTK data...")
        try:
            # First attempt - normal download
            nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
            nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=nltk_data_dir)
            nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
        except Exception as ssl_error:
            print("SSL verification issue detected. Trying alternative download method...")
            import ssl
            
            # Create an unverified SSL context
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Try the download again with the modified SSL context
            nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
            nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=nltk_data_dir)
            nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
        
        # Verify that NLTK data files are available
        nltk_data_files = {
            'punkt': os.path.join(nltk_data_dir, 'tokenizers', 'punkt'),
            'averaged_perceptron_tagger': os.path.join(nltk_data_dir, 'taggers', 'averaged_perceptron_tagger'),
            'wordnet': os.path.join(nltk_data_dir, 'corpora', 'wordnet')
        }
        
        all_available = True
        for name, path in nltk_data_files.items():
            if not os.path.exists(path):
                print(f"Warning: {name} data not available at {path}")
                all_available = False
        
        if all_available:
            print("All NLTK data files successfully downloaded!")
        else:
            print("Some NLTK data files may be missing. Continuing with degraded functionality.")
            
        print("Setup complete!")
    except Exception as e:
        print(f"Setup failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        sys.exit(1)

def fetch_crypto_data():
    """Fetch real-time cryptocurrency data from CoinGecko API."""
    # Initialize CoinGecko API client
    cg = CoinGeckoAPI()
    
    # Fetch real-time data for an expanded list of cryptocurrencies
    coins = [
        'bitcoin', 'ethereum', 'cardano',  # Original coins
        'solana', 'polkadot', 'ripple', 'dogecoin', 'avalanche-2',  # Additional major coins
        'chainlink', 'polygon', 'near'  # More options
    ]
    try:
        data = cg.get_price(
            ids=coins,
            vs_currencies='usd',
            include_market_cap=True,
            include_24hr_change=True
        )
        
        # Enhanced sustainability data with research-based estimates
        # Note: These are still simulated but more comprehensive
        sustainability_data = {
            'bitcoin': {'energy_use': 'high', 'sustainability_score': 3/10, 'consensus': 'proof-of-work'},
            'ethereum': {'energy_use': 'medium', 'sustainability_score': 6/10, 'consensus': 'proof-of-stake'},
            'cardano': {'energy_use': 'low', 'sustainability_score': 8/10, 'consensus': 'proof-of-stake'},
            'solana': {'energy_use': 'low', 'sustainability_score': 7.5/10, 'consensus': 'proof-of-stake'},
            'polkadot': {'energy_use': 'low', 'sustainability_score': 7.2/10, 'consensus': 'nominated proof-of-stake'},
            'ripple': {'energy_use': 'very low', 'sustainability_score': 8.2/10, 'consensus': 'federated consensus'},
            'dogecoin': {'energy_use': 'high', 'sustainability_score': 3.5/10, 'consensus': 'proof-of-work'},
            'avalanche-2': {'energy_use': 'low', 'sustainability_score': 7.8/10, 'consensus': 'proof-of-stake'},
            'chainlink': {'energy_use': 'medium', 'sustainability_score': 6.5/10, 'consensus': 'hybrid'},
            'polygon': {'energy_use': 'low', 'sustainability_score': 7.3/10, 'consensus': 'proof-of-stake'},
            'near': {'energy_use': 'low', 'sustainability_score': 7.9/10, 'consensus': 'proof-of-stake'}
        }
        
        # Construct crypto database
        crypto_db = {}
        for coin in coins:
            if coin in data:
                price_change = data[coin]['usd_24h_change']
                market_cap = data[coin]['usd_market_cap']
                crypto_db[coin.capitalize()] = {
                    'price_trend': 'rising' if price_change > 1 else 'stable' if -1 <= price_change <= 1 else 'falling',
                    'market_cap': 'high' if market_cap > 100_000_000_000 else 'medium' if market_cap > 10_000_000_000 else 'low',
                    'energy_use': sustainability_data[coin]['energy_use'],
                    'sustainability_score': sustainability_data[coin]['sustainability_score']
                }
        return crypto_db
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")
        return None

def fetch_historical_data(coin_id, days=30):
    """Fetch historical price data for a cryptocurrency.
    
    Args:
        coin_id (str): The CoinGecko ID of the cryptocurrency
        days (int): Number of days of historical data to retrieve
        
    Returns:
        dict: Historical data including prices, market caps, and volumes
    """
    try:
        cg = CoinGeckoAPI()
        historical_data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days
        )
        return historical_data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def calculate_technical_indicators(historical_data):
    """Calculate technical indicators from historical price data.
    
    Args:
        historical_data (dict): Historical price data from CoinGecko
        
    Returns:
        dict: Calculated technical indicators
    """
    if not historical_data or 'prices' not in historical_data:
        return None
    
    # Extract price data
    prices = [price[1] for price in historical_data['prices']]
    
    # Calculate simple moving averages
    sma_7 = sum(prices[-7:]) / 7 if len(prices) >= 7 else None
    sma_30 = sum(prices[-30:]) / 30 if len(prices) >= 30 else None
    
    # Calculate exponential moving averages (EMA)
    def calculate_ema(data, period, smoothing=2):
        ema = [data[0]]  # Start with first data point
        multiplier = smoothing / (1 + period)
        
        for price in data[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
            
        return ema
    
    # Calculate EMA if we have enough data points
    ema_12 = None
    ema_26 = None
    if len(prices) >= 26:
        ema_12_full = calculate_ema(prices, 12)
        ema_26_full = calculate_ema(prices, 26)
        ema_12 = ema_12_full[-1]
        ema_26 = ema_26_full[-1]
    
    # Calculate Relative Strength Index (RSI) - Enhanced version
    rsi_period = 14
    if len(prices) >= rsi_period + 1:
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(0, change) for change in changes]
        losses = [abs(min(0, change)) for change in changes]
        
        # Calculate initial average gain and loss
        avg_gain = sum(gains[:rsi_period]) / rsi_period
        avg_loss = sum(losses[:rsi_period]) / rsi_period
        
        # Use Wilder's smoothing method for subsequent values
        for i in range(rsi_period, len(changes)):
            avg_gain = (avg_gain * (rsi_period - 1) + gains[i]) / rsi_period
            avg_loss = (avg_loss * (rsi_period - 1) + losses[i]) / rsi_period
        
        # Calculate RS and RSI
        if avg_loss == 0:
            rsi = 100  # Prevent division by zero
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    else:
        rsi = None
    
    # Calculate price volatility (standard deviation)
    if len(prices) > 1:
        mean_price = sum(prices) / len(prices)
        squared_diffs = [(price - mean_price) ** 2 for price in prices]
        variance = sum(squared_diffs) / len(prices)
        volatility = variance ** 0.5
        volatility_percent = (volatility / mean_price) * 100
    else:
        volatility_percent = None
    
    # Calculate MACD (Moving Average Convergence Divergence) - Enhanced
    macd = None
    macd_signal = None
    macd_histogram = None
    
    if ema_12 is not None and ema_26 is not None:
        macd = ema_12 - ema_26
        
        # Calculate MACD signal line (9-day EMA of MACD)
        if len(prices) >= 35:  # Need at least 26 + 9 data points
            macd_line = [ema_12_full[i] - ema_26_full[i] for i in range(len(ema_26_full))]
            macd_signal_line = calculate_ema(macd_line, 9)
            macd_signal = macd_signal_line[-1]
            macd_histogram = macd - macd_signal
    
    # Calculate Bollinger Bands
    bollinger_period = 20
    if len(prices) >= bollinger_period:
        sma_20 = sum(prices[-bollinger_period:]) / bollinger_period
        
        # Calculate standard deviation for the period
        squared_diffs = [(price - sma_20) ** 2 for price in prices[-bollinger_period:]]
        std_dev = (sum(squared_diffs) / bollinger_period) ** 0.5
        
        upper_band = sma_20 + (2 * std_dev)
        lower_band = sma_20 - (2 * std_dev)
        
        # Determine if price is near upper or lower band (potential reversal)
        current_price = prices[-1]
        band_position = None
        
        if current_price > upper_band * 0.95:
            band_position = 'upper'
        elif current_price < lower_band * 1.05:
            band_position = 'lower'
        else:
            band_position = 'middle'
    else:
        upper_band = None
        lower_band = None
        band_position = None
    
    # Price momentum (percentage change)
    momentum_periods = [1, 7, 14, 30]
    momentum = {}
    
    for period in momentum_periods:
        if len(prices) >= period + 1:
            momentum[f'{period}d'] = ((prices[-1] / prices[-(period+1)]) - 1) * 100
    
    # Return enhanced technical indicators
    return {
        'current_price': prices[-1] if prices else None,
        'sma_7': sma_7,
        'sma_30': sma_30,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'rsi': rsi,
        'volatility': volatility_percent,
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram,
        'bollinger': {
            'upper': upper_band,
            'lower': lower_band,
            'position': band_position
        },
        'momentum': momentum,
        'trend_signal': get_trend_signal(sma_7, sma_30, rsi, macd) if all(x is not None for x in [sma_7, sma_30, rsi, macd]) else 'neutral'
    }

def get_trend_signal(sma_7, sma_30, rsi, macd, indicators=None):
    """Determine trend signal based on technical indicators.
    
    Args:
        sma_7 (float): 7-day simple moving average
        sma_30 (float): 30-day simple moving average
        rsi (float): Relative Strength Index
        macd (float): Moving Average Convergence Divergence
        indicators (dict, optional): Full technical indicators dictionary for additional signals
        
    Returns:
        dict: Contains overall trend signal and confidence score along with individual signals
    """
    signals = []
    signal_details = {}
    
    # SMA signal
    if sma_7 > sma_30:
        signals.append('bullish')
        signal_details['sma_crossover'] = 'bullish'
    elif sma_7 < sma_30:
        signals.append('bearish')
        signal_details['sma_crossover'] = 'bearish'
    else:
        signals.append('neutral')
        signal_details['sma_crossover'] = 'neutral'
    
    # RSI signal
    if rsi is not None:
        if rsi < 30:
            signals.append('bullish')  # Oversold
            signal_details['rsi'] = 'bullish (oversold)'
        elif rsi > 70:
            signals.append('bearish')  # Overbought
            signal_details['rsi'] = 'bearish (overbought)'
        else:
            signals.append('neutral')
            signal_details['rsi'] = 'neutral'
    
    # MACD signal
    if macd is not None:
        if macd > 0:
            signals.append('bullish')
            signal_details['macd'] = 'bullish'
        elif macd < 0:
            signals.append('bearish')
            signal_details['macd'] = 'bearish'
        else:
            signals.append('neutral')
            signal_details['macd'] = 'neutral'
    
    # Additional signals from the full indicators dictionary if provided
    if indicators:
        # Bollinger Bands signal
        if 'bollinger' in indicators and indicators['bollinger']['position']:
            if indicators['bollinger']['position'] == 'lower':
                signals.append('bullish')  # Price near lower band suggests potential upward reversal
                signal_details['bollinger'] = 'bullish (near lower band)'
            elif indicators['bollinger']['position'] == 'upper':
                signals.append('bearish')  # Price near upper band suggests potential downward reversal
                signal_details['bollinger'] = 'bearish (near upper band)'
            else:
                signals.append('neutral')
                signal_details['bollinger'] = 'neutral (middle band)'
        
        # MACD histogram signal (momentum)
        if 'macd_histogram' in indicators and indicators['macd_histogram'] is not None:
            if indicators['macd_histogram'] > 0 and indicators['macd_histogram'] > indicators.get('macd', 0):
                signals.append('bullish')
                signal_details['macd_histogram'] = 'bullish (increasing)'
            elif indicators['macd_histogram'] < 0 and indicators['macd_histogram'] < indicators.get('macd', 0):
                signals.append('bearish')
                signal_details['macd_histogram'] = 'bearish (decreasing)'
            else:
                signals.append('neutral')
                signal_details['macd_histogram'] = 'neutral'
        
        # Recent momentum signal
        if 'momentum' in indicators and '7d' in indicators['momentum']:
            momentum_7d = indicators['momentum']['7d']
            if momentum_7d > 5:  # 5% gain in 7 days
                signals.append('bullish')
                signal_details['momentum'] = f'bullish ({momentum_7d:.2f}% in 7d)'
            elif momentum_7d < -5:  # 5% loss in 7 days
                signals.append('bearish')
                signal_details['momentum'] = f'bearish ({momentum_7d:.2f}% in 7d)'
            else:
                signals.append('neutral')
                signal_details['momentum'] = f'neutral ({momentum_7d:.2f}% in 7d)'
    
    # Count signals
    bullish_count = signals.count('bullish')
    bearish_count = signals.count('bearish')
    neutral_count = signals.count('neutral')
    total_count = len(signals)
    
    # Calculate confidence score (0-100%)
    if total_count > 0:
        if bullish_count > bearish_count:
            confidence = (bullish_count / total_count) * 100
            overall_signal = 'bullish'
        elif bearish_count > bullish_count:
            confidence = (bearish_count / total_count) * 100
            overall_signal = 'bearish'
        else:
            confidence = (neutral_count / total_count) * 100
            overall_signal = 'neutral'
    else:
        confidence = 0
        overall_signal = 'neutral'
    
    return {
        'signal': overall_signal,
        'confidence': confidence,
        'details': signal_details,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'neutral_count': neutral_count
    }
    
def get_synonyms(word):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
    except LookupError:
        # WordNet data not available, return empty set
        pass
    return synonyms

def get_greeting():
    """Return a random greeting template."""
    greetings = [
        "Hey there! I'm CryptoBuddy, your crypto sidekick! 🌟\n",
        "Hello crypto explorer! CryptoBuddy at your service! 💰\n",
        "Greetings, future crypto mogul! CryptoBuddy here to help! 🚀\n",
        "Welcome to the crypto universe! I'm CryptoBuddy, your guide! ✨\n",
        "Hi there! CryptoBuddy reporting for duty! 📊\n",
        "Yo! CryptoBuddy in the house! Ready to talk crypto! 🤑\n",
        "Howdy partner! CryptoBuddy here for all your crypto questions! 🤠\n",
        "Salutations, crypto enthusiast! CryptoBuddy at your disposal! 🧠\n",
        "G'day mate! CryptoBuddy ready to dive into the crypto world! 🏄\n",
        "What's up? CryptoBuddy here to demystify crypto for you! 🔍\n"
    ]
    return random.choice(greetings)

def get_disclaimer():
    """Return a random disclaimer template."""
    disclaimers = [
        "⚠️ Crypto is risky—always do your own research!\n",
        "⚠️ Remember: all crypto investments come with risks. Never invest what you can't afford to lose!\n",
        "⚠️ Disclaimer: This is not financial advice. Always research before investing!\n",
        "⚠️ The crypto market is volatile—proceed with caution and your own due diligence!\n",
        "⚠️ Warning: Cryptocurrency values can dramatically fluctuate. Invest wisely!\n",
        "⚠️ Note: I'm just a friendly AI, not a financial advisor. Make informed decisions!\n",
        "⚠️ Friendly reminder: The crypto space is high-risk! Do thorough research!\n",
        "⚠️ Important: Crypto investments can result in substantial gains or losses. Be careful!\n",
        "⚠️ PSA: Always verify information and consult professionals before crypto investments!\n",
        "⚠️ Caution: The crypto market never sleeps and can be unpredictable. Stay informed!\n"
    ]
    return random.choice(disclaimers)

def interpret_query(user_query):
    """Interpret user query intent using NLP."""
    # Check if NLTK resources are available
    nltk_available = True
    try:
        # Try using NLTK functions
        tokens = word_tokenize(user_query.lower())
        tagged = pos_tag(tokens)
    except LookupError:
        # NLTK data not available, use simple fallback
        nltk_available = False
        tokens = user_query.lower().split()
        tagged = [(token, 'UNKNOWN') for token in tokens]
        print("Notice: Using simplified query analysis (NLTK data unavailable)")
    
    # Define expanded keywords and their synonyms for better intent recognition
    trend_keywords = {
        'trend', 'trending', 'rise', 'rising', 'up', 'growth', 'profitable', 'profit',
        'hot', 'booming', 'surging', 'bullish', 'mooning', 'pumping', 'momentum', 'gains',
        'climbing', 'increasing', 'appreciating', 'growing', 'popular', 'hype', 'buzz'
    }
    
    sustain_keywords = {
        'sustainable', 'sustainability', 'eco', 'ecofriendly', 'green', 'environment',
        'clean', 'renewable', 'carbon', 'footprint', 'climate', 'earth', 'planet',
        'ecological', 'responsible', 'ethical', 'energy', 'efficient', 'conscious'
    }
    
    longterm_keywords = {
        'longterm', 'long', 'future', 'growth', 'potential', 'investment', 'hold',
        'hodl', 'stable', 'stability', 'lasting', 'enduring', 'portfolio', 'retirement',
        'years', 'decade', 'permanent', 'horizon', 'prospect', 'tomorrow'
    }
    
    general_keywords = {
        'which', 'what', 'recommend', 'suggest', 'best', 'good', 'better',
        'advice', 'help', 'guide', 'opinion', 'think', 'consider', 'thoughts',
        'view', 'guidance', 'suggestion', 'recommendation', 'insight', 'preference'
    }
    
    # Expand keywords with synonyms if NLTK is available
    trend_synonyms = set(trend_keywords)
    sustain_synonyms = set(sustain_keywords)
    longterm_synonyms = set(longterm_keywords)
    
    if nltk_available:
        for word in trend_keywords:
            trend_synonyms.update(get_synonyms(word))
        for word in sustain_keywords:
            sustain_synonyms.update(get_synonyms(word))
        for word in longterm_keywords:
            longterm_synonyms.update(get_synonyms(word))
    
    # Analyze query intent
    intent = 'general'
    for token, pos in tagged:
        token = token.lower()
        if token in trend_synonyms or token in trend_keywords:
            intent = 'trending'
            break
        elif token in sustain_synonyms or token in sustain_keywords:
            intent = 'sustainable'
            break
        elif token in longterm_synonyms or token in longterm_keywords:
            intent = 'longterm'
            break
        elif token in general_keywords:
            intent = 'general'
    
    return intent

def get_trending_response(coin_name, trend, market_cap):
    """Return a varied response for trending crypto recommendations."""
    responses = [
        f"For profitability, I recommend {coin_name}! 🚀 It's {trend} up with a {market_cap} market cap.\n",
        f"If you're chasing gains, {coin_name} is your best bet! It's currently {trend} with substantial market presence.\n",
        f"Looking at momentum plays? {coin_name} stands out with its {trend} trajectory and {market_cap} capitalization! 📈\n",
        f"For maximum profit potential, {coin_name} is showing strong bullish signals with its {trend} pattern.\n",
        f"Hot pick alert! {coin_name} is demonstrating remarkable upward mobility in the current market! 🔥\n",
        f"Investors seeking short-term gains are flocking to {coin_name} due to its {trend} momentum and {market_cap} market cap! 💸\n",
        f"{coin_name} is currently the star performer, with impressive {trend} patterns that technical analysts are excited about! ⭐\n",
        f"The numbers don't lie - {coin_name} is outperforming with its {trend} trend and robust {market_cap} capitalization! 📊\n",
        f"My analysis suggests {coin_name} for those seeking profit opportunities in this volatile market! 🧠\n",
        f"Traders are buzzing about {coin_name} right now - it's showing strong {trend} signals with substantial market backing! 🐝\n"
    ]
    return random.choice(responses)

def get_no_trending_response():
    """Return a response when no trending coins are found."""
    responses = [
        "Hmm, no coins match both rising trends and high market cap right now. Try Cardano for a rising star! 🌱\n",
        "The market's a bit cautious today - no clear winners with both upward momentum and market dominance. Consider Cardano as an alternative! 🔍\n",
        "I don't see any cryptocurrencies that satisfy both trending status and market cap criteria at the moment. Cardano might be worth investigating! 🧐\n",
        "Market conditions are complex right now. No cryptos meet my strict criteria for both trend and capitalization, but Cardano shows promise! 🌊\n",
        "It's a tricky market cycle - there aren't any clear winners meeting both my trend and market cap thresholds. Take a look at Cardano! 🔄\n",
        "The data isn't showing any cryptocurrencies with the perfect combination of upward momentum and market strength. Cardano remains interesting though! 📱\n",
        "Today's market isn't giving us any cryptos with both bullish trends and substantial market presence. Cardano has potential worth exploring! 🌠\n",
        "Market indicators aren't aligning for any single cryptocurrency right now. Consider Cardano for its growing potential! 📶\n",
        "The crypto markets are consolidating - no assets currently display both strong upward movement and significant market share. Cardano shows positive signs! 📉📈\n",
        "I can't confidently recommend any crypto for pure momentum trading right now. Cardano offers an interesting alternative with its unique approach! 🧩\n"
    ]
    return random.choice(responses)

def get_sustainable_response(coin_name, score):
    """Return a varied response for sustainable crypto recommendations."""
    formatted_score = score * 10
    responses = [
        f"Invest in {coin_name}! 🌱 It's eco-friendly with a sustainability score of {formatted_score}/10!\n",
        f"For the environmentally conscious investor, {coin_name} leads the pack with an impressive sustainability score of {formatted_score}/10! 🌍\n",
        f"Looking to go green? {coin_name} stands out with its remarkable {formatted_score}/10 sustainability rating! 🌿\n",
        f"Eco-warriors rejoice! {coin_name} offers both investment potential and environmental responsibility with a {formatted_score}/10 sustainability score! 🌳\n",
        f"The planet will thank you for investing in {coin_name} - it boasts a stellar {formatted_score}/10 on the sustainability scale! 🌎\n",
        f"For minimal carbon footprint with maximum potential, {coin_name} delivers with its {formatted_score}/10 sustainability rating! ♻️\n",
        f"Climate-conscious crypto enthusiasts are rallying behind {coin_name} thanks to its impressive {formatted_score}/10 sustainability metrics! 🌦️\n",
        f"{coin_name} is leading the charge in the green crypto revolution with a sustainability score of {formatted_score}/10! 🚀\n",
        f"Sustainable investing made simple: {coin_name} scores a remarkable {formatted_score}/10 for environmental responsibility! 📊\n",
        f"Merge your financial goals with environmental values by considering {coin_name} - sustainability score: {formatted_score}/10! 💚\n"
    ]
    return random.choice(responses)

def get_less_sustainable_response(coin_name, score):
    """Return a response for less sustainable crypto options."""
    formatted_score = score * 10
    responses = [
        f"{coin_name} is the most sustainable with a score of {formatted_score}/10, but explore more options for greener choices! 🌍\n",
        f"While {coin_name} leads available options with a {formatted_score}/10 sustainability rating, the crypto industry is still evolving toward greener solutions! 🌱\n",
        f"{coin_name} scores {formatted_score}/10 on sustainability - moderate, but the best among current options. Keep an eye out for emerging greener alternatives! 🔍\n",
        f"With a {formatted_score}/10 sustainability score, {coin_name} is leading the pack, though the industry has room for improvement in eco-friendly practices! 📈\n",
        f"{coin_name} offers the best environmental credentials ({formatted_score}/10) among current options, though truly green crypto remains an evolving goal! 🌿\n",
        f"At {formatted_score}/10 for sustainability, {coin_name} represents the current best option, though the crypto space is still working toward truly green solutions! 🌤️\n",
        f"The most eco-conscious choice available is {coin_name} with a {formatted_score}/10 sustainability score. The crypto industry continues to work on reducing its footprint! ♻️\n",
        f"{coin_name} leads with a {formatted_score}/10 sustainability score, but remember that the benchmark for truly sustainable crypto is still developing! 🌳\n",
        f"Among current options, {coin_name} is most sustainable at {formatted_score}/10 - a modest but leading score as the industry moves toward greener technologies! 🌎\n",
        f"With environmental concerns in mind, {coin_name} offers the best current balance at {formatted_score}/10 for sustainability. Keep watching as greener options emerge! 🔭\n"
    ]
    return random.choice(responses)

def get_longterm_response(coin_name, trend, score):
    """Return a varied response for long-term crypto recommendations."""
    formatted_score = score * 10
    responses = [
        f"For long-term growth, go with {coin_name}! 🚀 It's {trend} up and has a sustainability score of {formatted_score}/10!\n",
        f"Looking years ahead? {coin_name} presents an intriguing balance of current {trend} momentum and future-proof sustainability ({formatted_score}/10)! 🔮\n",
        f"Strategic investors should consider {coin_name} for their portfolios - it combines {trend} performance with sustainable practices rated {formatted_score}/10! 📈\n",
        f"For the patient investor thinking long-term, {coin_name} offers both {trend} trends and environmental consciousness ({formatted_score}/10)! ⏳\n",
        f"My long-range analysis points to {coin_name} as a solid option with its {trend} technical indicators and sustainability rating of {formatted_score}/10! 📊\n",
        f"Those building wealth over time might appreciate {coin_name}'s combination of {trend} market behavior and eco-friendly approach ({formatted_score}/10)! 💰\n",
        f"The long-term thesis for {coin_name} looks compelling - {trend} market performance plus sustainable practices scoring {formatted_score}/10! 📝\n",
        f"If you're playing the long game, {coin_name} deserves attention with its {trend} trajectory and impressive sustainability metrics ({formatted_score}/10)! 🎯\n",
        f"Retirement portfolio material? Consider {coin_name} with its {trend} patterns and forward-thinking sustainability score of {formatted_score}/10! 🏦\n",
        f"{coin_name} stands out for horizon investors - showing {trend} momentum now and future-readiness with a sustainability rating of {formatted_score}/10! 🌅\n"
    ]
    return random.choice(responses)

def get_no_longterm_response():
    """Return a response when no ideal long-term coins are found."""
    responses = [
        "Cardano looks promising for long-term growth with its rising trend and eco-friendly vibe! 🌱\n",
        "For long-term investing, I'd suggest exploring Cardano - its proof-of-stake consensus mechanism offers sustainable growth potential! ♻️\n",
        "While no coins meet my strict long-term criteria right now, Cardano stands out with its forward-thinking architecture and environmental consciousness! 🌿\n",
        "The horizon investor might appreciate Cardano's unique combination of technological innovation and energy efficiency! 🔋\n",
        "Looking years ahead? Cardano's approach to sustainability while maintaining growth potential makes it worth investigating! 🔭\n",
        "For the patient investor, Cardano represents an intriguing long-term prospect with its environmentally responsible blockchain design! 🌳\n",
        "My analysis suggests Cardano for those thinking beyond market cycles - its sustainable foundation supports long-range potential! 📈\n",
        "Strategic portfolio building might include Cardano for its balance of current development and future-focused sustainability! 🧩\n",
        "The long view favors projects like Cardano that prioritize both technological advancement and environmental responsibility! 🌎\n",
        "No perfect candidates for long-term investment right now, but Cardano's trajectory and eco-credentials make it worth consideration! 🌠\n"
    ]
    return random.choice(responses)

def get_general_response(coin_name, trend, score):
    """Return a varied general recommendation response."""
    formatted_score = score * 10
    responses = [
        f"I'd suggest {coin_name}! It balances profitability (trend: {trend}) and sustainability (score: {formatted_score}/10). 🚀🌱\n",
        f"Based on current data, {coin_name} offers a compelling balance of {trend} market behavior and environmental responsibility ({formatted_score}/10)! 📊\n",
        f"My analysis points to {coin_name} as a well-rounded option with its {trend} price action and sustainability rating of {formatted_score}/10! 🔍\n",
        f"Looking at the overall picture, {coin_name} stands out with its {trend} momentum and eco-conscious approach (scoring {formatted_score}/10)! 🌟\n",
        f"For a balanced investment approach, consider {coin_name} - it's showing {trend} performance while maintaining a {formatted_score}/10 sustainability score! ⚖️\n",
        f"Taking all factors into account, {coin_name} emerges as a strong contender with its {trend} market position and {formatted_score}/10 environmental rating! 🏆\n",
        f"The data suggests {coin_name} as your best all-around option, combining {trend} financial indicators with responsible practices ({formatted_score}/10)! 📱\n",
        f"My comprehensive evaluation favors {coin_name}, which balances {trend} market metrics with sustainable operations rated {formatted_score}/10! 🧠\n",
        f"For investors seeking balance, {coin_name} delivers with its {trend} trajectory and commitment to sustainability (scoring {formatted_score}/10)! 🎯\n",
        f"When weighing all criteria, {coin_name} comes out ahead by combining {trend} market performance with environmental consciousness ({formatted_score}/10)! 💯\n"
    ]
    return random.choice(responses)

class UserProfile:
    """A class to store and manage user cryptocurrency preferences."""
    
    def __init__(self, username='default'):
        """Initialize a user profile.
        
        Args:
            username (str): The name of the user profile
        """
        self.username = username
        self.favorite_coins = []
        self.risk_tolerance = 'medium'  # 'low', 'medium', 'high'
        self.sustainability_preference = 'medium'  # 'low', 'medium', 'high'
        self.investment_horizon = 'medium'  # 'short', 'medium', 'long'
        
    def add_favorite_coin(self, coin):
        """Add a cryptocurrency to favorites.
        
        Args:
            coin (str): The name/id of the cryptocurrency
        """
        if coin not in self.favorite_coins:
            self.favorite_coins.append(coin)
            
    def remove_favorite_coin(self, coin):
        """Remove a cryptocurrency from favorites.
        
        Args:
            coin (str): The name/id of the cryptocurrency
        """
        if coin in self.favorite_coins:
            self.favorite_coins.remove(coin)
            
    def set_risk_tolerance(self, level):
        """Set the user's risk tolerance level.
        
        Args:
            level (str): 'low', 'medium', or 'high'
        """
        if level in ['low', 'medium', 'high']:
            self.risk_tolerance = level
            
    def set_sustainability_preference(self, level):
        """Set the user's preference for sustainable cryptocurrencies.
        
        Args:
            level (str): 'low', 'medium', or 'high'
        """
        if level in ['low', 'medium', 'high']:
            self.sustainability_preference = level
            
    def set_investment_horizon(self, horizon):
        """Set the user's investment time horizon.
        
        Args:
            horizon (str): 'short', 'medium', or 'long'
        """
        if horizon in ['short', 'medium', 'long']:
            self.investment_horizon = horizon
    
    def save_profile(self):
        """Save the user profile to a file."""
        try:
            import json
            import os
            from datetime import datetime
            
            # Create profiles directory if it doesn't exist
            os.makedirs('profiles', exist_ok=True)
            
            # Save profile with additional metadata
            profile_data = {
                'username': self.username,
                'favorite_coins': self.favorite_coins,
                'risk_tolerance': self.risk_tolerance,
                'sustainability_preference': self.sustainability_preference,
                'investment_horizon': self.investment_horizon,
                'last_updated': datetime.now().isoformat(),
                'version': '1.1'  # Version tracking for future compatibility
            }
            
            # Add price alerts if they exist
            try:
                alert_manager = AlertManager()
                alert_manager.load_alerts()
                user_alerts = alert_manager.get_alerts_by_username(self.username)
                profile_data['active_alerts_count'] = len([a for a in user_alerts if not a.triggered])
            except Exception:
                profile_data['active_alerts_count'] = 0
            
            # Save the file with proper error handling
            profile_path = f'profiles/{self.username}.json'
            
            # Create backup of existing file if it exists
            if os.path.exists(profile_path):
                try:
                    backup_path = f'profiles/{self.username}.backup.json'
                    with open(profile_path, 'r') as src:
                        with open(backup_path, 'w') as dst:
                            dst.write(src.read())
                except Exception as backup_error:
                    print(f"Warning: Failed to create profile backup: {backup_error}")
            
            # Write the new profile
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
                
            print(f"Profile saved successfully for {self.username}")
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False
    
    @classmethod
    def load_profile(cls, username):
        """Load a user profile from a file.
        
        Args:
            username (str): The name of the profile to load
            
        Returns:
            UserProfile: The loaded user profile or a new default profile
        """
        try:
            import json
            import os
            from datetime import datetime
            
            profile_path = f'profiles/{username}.json'
            
            # Create new profile if file doesn't exist
            if not os.path.exists(profile_path):
                print(f"Creating new profile for {username}")
                new_profile = cls(username)
                new_profile.save_profile()  # Save the new profile
                return new_profile
            
            # Try to load the profile
            try:
                with open(profile_path, 'r') as f:
                    profile_data = json.load(f)
                
                profile = cls(username)
                
                # Load basic profile data
                profile.favorite_coins = profile_data.get('favorite_coins', [])
                profile.risk_tolerance = profile_data.get('risk_tolerance', 'medium')
                profile.sustainability_preference = profile_data.get('sustainability_preference', 'medium')
                profile.investment_horizon = profile_data.get('investment_horizon', 'medium')
                
                # Check profile version for future compatibility
                profile_version = profile_data.get('version', '1.0')
                current_version = '1.1'  # Update this when the profile structure changes
                
                if profile_version != current_version:
                    print(f"Notice: Profile version {profile_version} found, current version is {current_version}.")
                    print("The profile will be updated to the current version format.")
                    profile.save_profile()  # Update to current version
                
                print(f"Profile loaded successfully for {username}")
                
                # Print last login time if available
                if 'last_updated' in profile_data:
                    try:
                        last_updated = datetime.fromisoformat(profile_data['last_updated'])
                        time_diff = datetime.now() - last_updated
                        days = time_diff.days
                        
                        if days == 0:
                            time_msg = "today"
                        elif days == 1:
                            time_msg = "yesterday"
                        else:
                            time_msg = f"{days} days ago"
                        
                        print(f"Last login: {time_msg}")
                    except:
                        pass
                
                return profile
                
            except json.JSONDecodeError:
                print(f"Error: Profile file for {username} is corrupted.")
                
                # Try to restore from backup
                backup_path = f'profiles/{username}.backup.json'
                if os.path.exists(backup_path):
                    print("Attempting to restore from backup...")
                    try:
                        with open(backup_path, 'r') as f:
                            profile_data = json.load(f)
                        
                        profile = cls(username)
                        profile.favorite_coins = profile_data.get('favorite_coins', [])
                        profile.risk_tolerance = profile_data.get('risk_tolerance', 'medium')
                        profile.sustainability_preference = profile_data.get('sustainability_preference', 'medium')
                        profile.investment_horizon = profile_data.get('investment_horizon', 'medium')
                        
                        # Immediately save to restore the primary profile file
                        profile.save_profile()
                        print("Profile restored from backup.")
                        return profile
                    except Exception:
                        print("Backup restoration failed.")
                
                # Creating new profile as fallback
                print(f"Creating new profile for {username}")
                return cls(username)
                
        except Exception as e:
            print(f"Error loading profile: {e}")
            return cls(username)
    
    def get_personalized_recommendations(self, crypto_db):
        """Get personalized cryptocurrency recommendations based on user preferences.
        
        Args:
            crypto_db (dict): Database of cryptocurrency information
            
        Returns:
            list: Sorted list of recommended cryptocurrencies with scores
        """
        recommendations = []
        
        for coin, data in crypto_db.items():
            score = 0
            
            # Favorite coins get a boost
            if coin.lower() in [c.lower() for c in self.favorite_coins]:
                score += 2
            
            # Risk tolerance factor (based on volatility, which we don't have yet)
            if self.risk_tolerance == 'high' and data['market_cap'] == 'low':
                score += 1
            elif self.risk_tolerance == 'medium' and data['market_cap'] == 'medium':
                score += 1
            elif self.risk_tolerance == 'low' and data['market_cap'] == 'high':
                score += 1
                
            # Sustainability preference
            if self.sustainability_preference == 'high':
                if data['sustainability_score'] > 7/10:
                    score += 2
                elif data['sustainability_score'] > 5/10:
                    score += 1
            elif self.sustainability_preference == 'medium':
                if data['sustainability_score'] > 5/10:
                    score += 1
            
            # Investment horizon
            if self.investment_horizon == 'short' and data['price_trend'] == 'rising':
                score += 1
            elif self.investment_horizon == 'medium' and data['market_cap'] != 'low':
                score += 1
            elif self.investment_horizon == 'long' and data['sustainability_score'] > 6/10:
                score += 1
                
            recommendations.append((coin, score))
        
        # Sort by score, highest first
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations

class PortfolioManager:
    """A class to manage cryptocurrency portfolio suggestions."""
    
    def __init__(self):
        """Initialize the portfolio manager."""
        # Portfolio templates with risk profiles
        self.portfolio_templates = {
            'conservative': {
                'bitcoin': 0.50,  # 50% allocation to Bitcoin
                'ethereum': 0.30,  # 30% allocation to Ethereum
                'cardano': 0.10,  # 10% allocation to Cardano
                'solana': 0.05,   # 5% allocation to Solana
                'polkadot': 0.05  # 5% allocation to Polkadot
            },
            'balanced': {
                'bitcoin': 0.35,
                'ethereum': 0.25,
                'cardano': 0.15,
                'solana': 0.10,
                'polkadot': 0.05,
                'chainlink': 0.05,
                'avalanche-2': 0.05
            },
            'aggressive': {
                'bitcoin': 0.20,
                'ethereum': 0.20,
                'cardano': 0.15,
                'solana': 0.15,
                'polkadot': 0.10,
                'chainlink': 0.05,
                'avalanche-2': 0.05,
                'dogecoin': 0.05,
                'near': 0.05
            },
            'eco_friendly': {
                'ethereum': 0.30,
                'cardano': 0.25,
                'solana': 0.15,
                'polkadot': 0.10,
                'avalanche-2': 0.10,
                'near': 0.10
            },
            'high_cap': {
                'bitcoin': 0.50,
                'ethereum': 0.30,
                'solana': 0.10,
                'cardano': 0.05,
                'ripple': 0.05
            }
        }
    
    def get_portfolio_suggestion(self, risk_profile, investment_amount=1000):
        """Get a portfolio suggestion based on risk profile.
        
        Args:
            risk_profile (str): 'conservative', 'balanced', 'aggressive', 'eco_friendly', or 'high_cap'
            investment_amount (float): The total investment amount
            
        Returns:
            dict: A dictionary with coin allocations and amounts
        """
        if risk_profile not in self.portfolio_templates:
            risk_profile = 'balanced'  # Default to balanced if risk profile not found
            
        template = self.portfolio_templates[risk_profile]
        
        # Get current prices
        try:
            cg = CoinGeckoAPI()
            prices = cg.get_price(ids=list(template.keys()), vs_currencies='usd')
            
            portfolio = {
                'risk_profile': risk_profile,
                'total_amount': investment_amount,
                'date_created': datetime.datetime.now().isoformat(),
                'allocations': []
            }
            
            for coin, allocation in template.items():
                if coin in prices:
                    amount = investment_amount * allocation
                    coin_amount = amount / prices[coin]['usd']
                    
                    portfolio['allocations'].append({
                        'coin': coin,
                        'allocation_percentage': allocation * 100,
                        'fiat_amount': amount,
                        'coin_amount': coin_amount,
                        'current_price': prices[coin]['usd']
                    })
            
            return portfolio
        except Exception as e:
            print(f"Error generating portfolio: {e}")
            return None
    
    def get_portfolio_recommendation_based_on_profile(self, user_profile):
        """Get a portfolio recommendation based on user profile.
        
        Args:
            user_profile (UserProfile): The user's profile
            
        Returns:
            dict: A portfolio suggestion
        """
        # Map user preferences to portfolio risk profile
        risk_mapping = {
            'low': 'conservative',
            'medium': 'balanced',
            'high': 'aggressive'
        }
        
        # If user has high sustainability preference, recommend eco_friendly portfolio
        if user_profile.sustainability_preference == 'high':
            risk_profile = 'eco_friendly'
        else:
            risk_profile = risk_mapping.get(user_profile.risk_tolerance, 'balanced')
        
        return self.get_portfolio_suggestion(risk_profile)
    
    def get_portfolio_performance(self, portfolio, days=30):
        """Calculate the historical performance of a portfolio.
        
        Args:
            portfolio (dict): A portfolio suggestion
            days (int): Number of days to look back
            
        Returns:
            dict: Performance metrics for the portfolio
        """
        try:
            cg = CoinGeckoAPI()
            performance = {
                'initial_investment': portfolio['total_amount'],
                'current_value': 0,
                'percent_change': 0,
                'coin_performance': []
            }
            
            for allocation in portfolio['allocations']:
                coin = allocation['coin']
                
                # Get historical data
                historical_data = fetch_historical_data(coin, days)
                if not historical_data or 'prices' not in historical_data:
                    continue
                
                # Get current and past prices
                current_price = historical_data['prices'][-1][1]
                past_price = historical_data['prices'][0][1]
                
                # Calculate performance
                initial_value = allocation['fiat_amount']
                current_value = allocation['coin_amount'] * current_price
                percent_change = ((current_price / past_price) - 1) * 100
                
                performance['current_value'] += current_value
                
                performance['coin_performance'].append({
                    'coin': coin,
                    'initial_value': initial_value,
                    'current_value': current_value,
                    'percent_change': percent_change
                })
            
            # Calculate overall portfolio performance
            if performance['initial_investment'] > 0:
                performance['percent_change'] = ((performance['current_value'] / performance['initial_investment']) - 1) * 100
                
            return performance
        except Exception as e:
            print(f"Error calculating portfolio performance: {e}")
            return None

def crypto_buddy_response(user_query, user_profile=None):
    """Generate chatbot response based on user query.
    
    Args:
        user_query (str): The user's query text
        user_profile (UserProfile, optional): User profile for personalized responses
        
    Returns:
        str: Chatbot response
    """
    # Fetch real-time data
    crypto_db = fetch_crypto_data()
    if not crypto_db:
        return "Oops! Couldn't fetch data from CoinGecko. Try again later! 😅"

    # Interpret query intent using NLP
    intent = interpret_query(user_query)
    
    # Check for special commands in the query
    query_lower = user_query.lower()
    
    # Handle portfolio suggestions
    if "portfolio" in query_lower or "diversify" in query_lower or "allocation" in query_lower:
        portfolio_manager = PortfolioManager()
        
        # Determine risk profile from the query or user preferences
        risk_profile = 'balanced'  # Default
        
        if "conservative" in query_lower or "safe" in query_lower or "low risk" in query_lower:
            risk_profile = 'conservative'
        elif "aggressive" in query_lower or "high risk" in query_lower or "growth" in query_lower:
            risk_profile = 'aggressive'
        elif "eco" in query_lower or "green" in query_lower or "sustainable" in query_lower:
            risk_profile = 'eco_friendly'
        elif "blue chip" in query_lower or "established" in query_lower:
            risk_profile = 'high_cap'
        elif user_profile:
            # Use user preferences if available
            if user_profile.sustainability_preference == 'high':
                risk_profile = 'eco_friendly'
            elif user_profile.risk_tolerance == 'low':
                risk_profile = 'conservative'
            elif user_profile.risk_tolerance == 'high':
                risk_profile = 'aggressive'
        
        # Get investment amount from query if mentioned
        investment_amount = 1000  # Default
        import re
        amount_matches = re.findall(r'(\$?\d+(?:,\d+)*(?:\.\d+)?)\s*(?:dollars|usd)?', query_lower)
        if amount_matches:
            try:
                # Remove $ and commas, then convert to float
                amount_str = amount_matches[0].replace('$', '').replace(',', '')
                investment_amount = float(amount_str)
            except ValueError:
                pass  # Use default if conversion fails
        
        # Generate portfolio suggestion
        portfolio = portfolio_manager.get_portfolio_suggestion(risk_profile, investment_amount)
        
        if portfolio:
            # Create a nicely formatted portfolio response
            greeting = get_greeting()
            disclaimer = get_disclaimer()
            
            response = greeting + disclaimer
            response += f"Here's a {risk_profile} portfolio allocation for ${investment_amount:.2f}:\n\n"
            
            for coin in portfolio['allocations']:
                coin_name = coin['coin'].capitalize()
                percentage = coin['allocation_percentage']
                usd_amount = coin['fiat_amount']
                coin_amount = coin['coin_amount']
                response += f"• {coin_name}: {percentage:.1f}% (${usd_amount:.2f} ≈ {coin_amount:.6f} {coin_name})\n"
            
            response += "\nThis allocation balances risk and potential returns based on current market conditions! 📊\n"
            response += "Remember to periodically rebalance your portfolio to maintain your target allocation. 🔄"
            
            return response
    
    # Handle price alerts
    elif "alert" in query_lower or "notify" in query_lower or "when price" in query_lower:
        if not user_profile:
            return "You need to be logged in to set price alerts! Please log in or create a profile first."
        
        # Extract the coin and target price from the query
        coin_matches = re.findall(r'(bitcoin|btc|ethereum|eth|cardano|ada|solana|sol|polkadot|dot|ripple|xrp|dogecoin|doge|avalanche|avax|chainlink|link|polygon|matic|near)', query_lower)
        price_matches = re.findall(r'(\$?\d+(?:,\d+)*(?:\.\d+)?)', query_lower)
        
        if coin_matches and price_matches:
            # Map common abbreviations to full coin IDs
            coin_map = {
                'btc': 'bitcoin', 'eth': 'ethereum', 'ada': 'cardano',
                'sol': 'solana', 'dot': 'polkadot', 'xrp': 'ripple',
                'doge': 'dogecoin', 'avax': 'avalanche-2', 'link': 'chainlink',
                'matic': 'polygon', 'near': 'near'
            }
            
            # Get the coin ID
            coin = coin_matches[0].lower()
            coin_id = coin_map.get(coin, coin)  # Use the mapping or keep as is
            
            # Get the target price
            target_price = price_matches[0].replace('$', '').replace(',', '')
            try:
                target_price = float(target_price)
            except ValueError:
                return "I couldn't understand the target price. Please try again with a clear price value."
            
            # Determine if this is an 'above' or 'below' alert
            alert_type = 'above'
            if 'below' in query_lower or 'under' in query_lower or 'drops' in query_lower or 'fall' in query_lower:
                alert_type = 'below'
            
            # Create the alert
            alert = PriceAlert(coin_id, target_price, alert_type, user_profile.username)
            
            # Add to alert manager
            alert_manager = AlertManager()
            alert_manager.load_alerts()
            alert_manager.add_alert(alert)
            
            # Start monitoring if it's not already running
            alert_manager.start_monitoring()
            
            # Format a nice response
            direction = "rises above" if alert_type == 'above' else "falls below"
            coin_name = coin_id.capitalize()
            response = f"🔔 Alert set! I'll notify you when {coin_name} {direction} ${target_price:.2f} USD.\n"
            response += f"You currently have {len(alert_manager.get_alerts_by_username(user_profile.username))} active alerts."
            
            return response
    
    # Handle technical analysis requests
    elif "analysis" in query_lower or "technical" in query_lower or "indicator" in query_lower:
        # Extract the coin from the query
        coin_matches = re.findall(r'(bitcoin|btc|ethereum|eth|cardano|ada|solana|sol|polkadot|dot|ripple|xrp|dogecoin|doge|avalanche|avax|chainlink|link|polygon|matic|near)', query_lower)
        
        if coin_matches:
            # Map abbreviations to full coin IDs
            coin_map = {
                'btc': 'bitcoin', 'eth': 'ethereum', 'ada': 'cardano',
                'sol': 'solana', 'dot': 'polkadot', 'xrp': 'ripple',
                'doge': 'dogecoin', 'avax': 'avalanche-2', 'link': 'chainlink',
                'matic': 'polygon', 'near': 'near'
            }
            
            # Get the coin ID
            coin = coin_matches[0].lower()
            coin_id = coin_map.get(coin, coin)
            
            # Determine the analysis period
            days = 30  # Default to 30 days
            if "week" in query_lower or "7 day" in query_lower:
                days = 7
            elif "month" in query_lower or "30 day" in query_lower:
                days = 30
            elif "quarter" in query_lower or "90 day" in query_lower:
                days = 90
            
            # Fetch historical data and calculate indicators
            historical_data = fetch_historical_data(coin_id, days)
            if historical_data:
                indicators = calculate_technical_indicators(historical_data)
                
                if indicators:
                    # Get trend signal with full indicators
                    trend_result = get_trend_signal(
                        indicators['sma_7'], 
                        indicators['sma_30'], 
                        indicators['rsi'], 
                        indicators['macd'], 
                        indicators
                    )
                    
                    # Format a nice technical analysis response
                    greeting = get_greeting()
                    disclaimer = get_disclaimer()
                    
                    response = greeting + disclaimer
                    response += f"Here's my {days}-day technical analysis for {coin_id.capitalize()}:\n\n"
                    
                    # Current price
                    current_price = indicators['current_price']
                    response += f"Current Price: ${current_price:.2f} USD\n\n"
                    
                    # Overall signal with confidence
                    signal = trend_result['signal'].capitalize()
                    confidence = trend_result['confidence']
                    response += f"📊 Overall Signal: {signal} (Confidence: {confidence:.1f}%)\n\n"
                    
                    # Key indicators summary
                    response += "Key Indicators:\n"
                    
                    # RSI
                    if indicators['rsi'] is not None:
                        rsi = indicators['rsi']
                        rsi_status = "Oversold! 📉" if rsi < 30 else "Overbought! 📈" if rsi > 70 else "Neutral ⚖️"
                        response += f"• RSI: {rsi:.1f} - {rsi_status}\n"
                    
                    # Moving Averages
                    if indicators['sma_7'] is not None and indicators['sma_30'] is not None:
                        sma_status = "Bullish ⬆️" if indicators['sma_7'] > indicators['sma_30'] else "Bearish ⬇️"
                        response += f"• Moving Average Crossover: {sma_status}\n"
                    
                    # MACD
                    if indicators['macd'] is not None:
                        macd_status = "Bullish ⬆️" if indicators['macd'] > 0 else "Bearish ⬇️"
                        response += f"• MACD: {macd_status}\n"
                    
                    # Bollinger Bands
                    if 'bollinger' in indicators and indicators['bollinger']['position']:
                        band_position = indicators['bollinger']['position']
                        band_desc = "Approaching resistance (possible reversal)" if band_position == 'upper' else "Approaching support (possible bounce)" if band_position == 'lower' else "Within trading range"
                        response += f"• Bollinger Bands: {band_desc}\n"
                    
                    # Momentum
                    if 'momentum' in indicators and '7d' in indicators['momentum']:
                        momentum = indicators['momentum']['7d']
                        response += f"• 7-Day Momentum: {momentum:.2f}%\n"
                    
                    # Additional summary and advice
                    if trend_result['signal'] == 'bullish':
                        response += "\n🔍 Analysis: Technical indicators suggest positive momentum. Consider watching for entry points."
                    elif trend_result['signal'] == 'bearish':
                        response += "\n🔍 Analysis: Technical indicators suggest downward pressure. Consider caution or hedging strategies."
                    else:
                        response += "\n🔍 Analysis: Technical indicators are mixed. The market appears to be consolidating."
                    
                    return response
                
    # Friendly greeting and ethics disclaimer
    greeting = get_greeting()
    disclaimer = get_disclaimer()
    
    # Initialize response
    response = greeting + disclaimer
    
    # Logic for handling standard user intents
    if intent == 'trending':
        # Find coins with rising price trend and high market cap
        rising_coins = [
            coin for coin in crypto_db 
            if crypto_db[coin]["price_trend"] == "rising" and crypto_db[coin]["market_cap"] == "high"
        ]
        if rising_coins:
            # Add technical analysis data for the recommended coin if available
            try:
                coin = rising_coins[0].lower()
                historical_data = fetch_historical_data(coin)
                if historical_data:
                    indicators = calculate_technical_indicators(historical_data)
                    if indicators and indicators['trend_signal'] and isinstance(indicators['trend_signal'], dict):
                        signal = indicators['trend_signal']['signal']
                        confidence = indicators['trend_signal']['confidence']
                        technical_info = f"\nTechnical analysis shows a {signal} trend with {confidence:.1f}% confidence. "
                        if 'rsi' in indicators and indicators['rsi']:
                            technical_info += f"RSI is at {indicators['rsi']:.1f}. "
                        if 'momentum' in indicators and '7d' in indicators['momentum']:
                            technical_info += f"7-day momentum is {indicators['momentum']['7d']:.2f}%."
                        response += get_trending_response(rising_coins[0], crypto_db[rising_coins[0]]["price_trend"], crypto_db[rising_coins[0]]["market_cap"]) + technical_info
                    else:
                        response += get_trending_response(rising_coins[0], crypto_db[rising_coins[0]]["price_trend"], crypto_db[rising_coins[0]]["market_cap"])
                else:
                    response += get_trending_response(rising_coins[0], crypto_db[rising_coins[0]]["price_trend"], crypto_db[rising_coins[0]]["market_cap"])
            except Exception:
                # Fallback to standard response if technical analysis fails
                response += get_trending_response(rising_coins[0], crypto_db[rising_coins[0]]["price_trend"], crypto_db[rising_coins[0]]["market_cap"])
        else:
            response += get_no_trending_response()
    
    elif intent == 'sustainable':
        # Find the most sustainable coin
        recommend = max(crypto_db, key=lambda x: crypto_db[x]["sustainability_score"])
        if crypto_db[recommend]["sustainability_score"] > 7/10:
            response += get_sustainable_response(recommend, crypto_db[recommend]['sustainability_score'])
        else:
            response += get_less_sustainable_response(recommend, crypto_db[recommend]['sustainability_score'])
    
    elif intent == 'longterm':
        # Recommend coins with rising trend and high sustainability
        growth_coins = [
            coin for coin in crypto_db 
            if crypto_db[coin]["price_trend"] == "rising" and crypto_db[coin]["sustainability_score"] > 7/10
        ]
        if growth_coins:
            response += get_longterm_response(growth_coins[0], crypto_db[growth_coins[0]]["price_trend"], crypto_db[growth_coins[0]]["sustainability_score"])
        else:
            response += get_no_longterm_response()
    
    else:  # General intent
        # If user profile exists, provide personalized recommendations
        if user_profile:
            personalized_recommendations = user_profile.get_personalized_recommendations(crypto_db)
            if personalized_recommendations:
                top_coin, _ = personalized_recommendations[0]
                response += get_general_response(top_coin, crypto_db[top_coin]['price_trend'], crypto_db[top_coin]['sustainability_score'])
                response += f"\nThis recommendation is personalized based on your profile preferences! 🎯"
            else:
                # General recommendation based on balanced criteria if personalization fails
                recommend = max(crypto_db, key=lambda x: (crypto_db[x]["sustainability_score"], 1 if crypto_db[x]["price_trend"] == "rising" else 0))
                response += get_general_response(recommend, crypto_db[recommend]['price_trend'], crypto_db[recommend]['sustainability_score'])
        else:
            # General recommendation based on balanced criteria
            recommend = max(crypto_db, key=lambda x: (crypto_db[x]["sustainability_score"], 1 if crypto_db[x]["price_trend"] == "rising" else 0))
            response += get_general_response(recommend, crypto_db[recommend]['price_trend'], crypto_db[recommend]['sustainability_score'])
    
    return response

def run_crypto_buddy():
    """Run the interactive CryptoBuddy chatbot."""
    print("\n===== CryptoBuddy - Your Crypto Advisor =====")
    print("Welcome to CryptoBuddy! Type 'exit' to quit, or ask about cryptos! 😎")
    print("Example queries:")
    print("  - What's a good cryptocurrency to invest in?")
    print("  - Which crypto is trending right now?")
    print("  - Tell me about sustainable cryptocurrencies")
    print("  - What's good for long term investment?")
    print("  - Suggest a balanced portfolio for $2000")
    print("  - Set an alert for Bitcoin when price goes above $50000")
    print("  - Show me a technical analysis of Ethereum")
    print("  - Give me historical performance data for Cardano")
    print("===============================================\n")
    
    # Load user profile
    username = input("Enter your username: ")
    user_profile = UserProfile.load_profile(username)
    print(f"Welcome back, {user_profile.username}! Your preferences have been loaded. 😊")
    
    # Initialize alert manager and start monitoring in the background
    alert_manager = AlertManager()
    alert_manager.load_alerts()
    
    # Check for existing alerts
    user_alerts = alert_manager.get_alerts_by_username(username)
    active_alerts = [a for a in user_alerts if not a.triggered]
    
    if active_alerts:
        print(f"You have {len(active_alerts)} active price alerts.")
        
    # Start monitoring
    if active_alerts:
        alert_manager.start_monitoring()
        print("Alert monitoring is active! You'll be notified when price conditions are met.")
    
    # Command help information
    help_info = """
Available commands:
- /help - Show this help message
- /profile - View and update your user profile
- /alerts - Manage your price alerts
- /portfolio - Get personalized portfolio suggestions
- /analyze <coin> - Get technical analysis for a cryptocurrency
- /exit - Exit the chatbot
"""
    
    while True:
        user_input = input("You: ")
        
        # Handle special commands
        if user_input.lower() in ["exit", "quit", "bye", "/exit"]:
            user_profile.save_profile()  # Save profile before exiting
            print("CryptoBuddy: Catch you later! Stay savvy! 👋")
            break
            
        elif user_input.lower() == "/help":
            print(help_info)
            continue
            
        elif user_input.lower() == "/profile":
            print("\n=== Your Profile ===")
            print(f"Username: {user_profile.username}")
            print(f"Favorite coins: {', '.join(user_profile.favorite_coins) if user_profile.favorite_coins else 'None'}")
            print(f"Risk tolerance: {user_profile.risk_tolerance}")
            print(f"Sustainability preference: {user_profile.sustainability_preference}")
            print(f"Investment horizon: {user_profile.investment_horizon}")
            
            update = input("\nWould you like to update your profile? (yes/no): ")
            if update.lower() in ["yes", "y"]:
                # Update risk tolerance
                risk = input("Risk tolerance (low/medium/high) [current: {}]: ".format(user_profile.risk_tolerance))
                if risk in ["low", "medium", "high"]:
                    user_profile.set_risk_tolerance(risk)
                
                # Update sustainability preference
                sustainability = input("Sustainability preference (low/medium/high) [current: {}]: ".format(user_profile.sustainability_preference))
                if sustainability in ["low", "medium", "high"]:
                    user_profile.set_sustainability_preference(sustainability)
                
                # Update investment horizon
                horizon = input("Investment horizon (short/medium/long) [current: {}]: ".format(user_profile.investment_horizon))
                if horizon in ["short", "medium", "long"]:
                    user_profile.set_investment_horizon(horizon)
                
                # Update favorite coins
                favorites = input("Favorite coins (comma-separated) [current: {}]: ".format(', '.join(user_profile.favorite_coins) if user_profile.favorite_coins else 'None'))
                if favorites.strip():
                    user_profile.favorite_coins = [coin.strip() for coin in favorites.split(',')]
                
                # Save updated profile
                if user_profile.save_profile():
                    print("Profile updated successfully! 👍")
                else:
                    print("Failed to save profile. 😢")
            continue
            
        elif user_input.lower() == "/alerts":
            user_alerts = alert_manager.get_alerts_by_username(username)
            
            if not user_alerts:
                print("You don't have any price alerts set.")
                print("Try saying something like: 'Alert me when Bitcoin goes above $50000'")
                continue
                
            print("\n=== Your Price Alerts ===")
            for i, alert in enumerate(user_alerts):
                status = "✅ TRIGGERED" if alert.triggered else "⏳ ACTIVE"
                direction = "rises above" if alert.alert_type == 'above' else "falls below"
                coin_name = alert.coin_id.capitalize()
                print(f"{i+1}. {status} - {coin_name} {direction} ${alert.target_price:.2f}")
                
                if alert.triggered and alert.triggered_at:
                    try:
                        triggered_date = datetime.datetime.fromisoformat(alert.triggered_at)
                        print(f"   Triggered on: {triggered_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        pass
            
            action = input("\nWould you like to (d)elete an alert, (c)lear all, or (b)ack? ")
            if action.lower() in ['d', 'delete']:
                num = input("Enter alert number to delete: ")
                try:
                    index = int(num) - 1
                    if 0 <= index < len(user_alerts):
                        if alert_manager.remove_alert(alert_manager.alerts.index(user_alerts[index])):
                            print("Alert deleted successfully!")
                        else:
                            print("Failed to delete the alert.")
                    else:
                        print("Invalid alert number.")
                except ValueError:
                    print("Please enter a valid number.")
            elif action.lower() in ['c', 'clear']:
                confirm = input("Are you sure you want to delete ALL your alerts? (yes/no): ")
                if confirm.lower() in ["yes", "y"]:
                    for alert in list(user_alerts):
                        alert_manager.alerts.remove(alert)
                    alert_manager.save_alerts()
                    print("All alerts deleted.")
            continue
            
        elif user_input.lower().startswith("/portfolio"):
            portfolio_manager = PortfolioManager()
            
            # Check for investment amount in command
            parts = user_input.split()
            investment_amount = 1000  # Default
            
            if len(parts) > 1:
                try:
                    amount = parts[1].replace('$', '').replace(',', '')
                    investment_amount = float(amount)
                except ValueError:
                    pass
                    
            # Get portfolio recommendation based on user profile
            portfolio = portfolio_manager.get_portfolio_recommendation_based_on_profile(user_profile)
            
            if portfolio:
                print(f"\n=== Personalized Portfolio Suggestion (${investment_amount:.2f}) ===")
                print(f"Risk profile: {portfolio['risk_profile']}")
                print("\nAllocation:")
                
                for coin in portfolio['allocations']:
                    coin_name = coin['coin'].capitalize()
                    percentage = coin['allocation_percentage']
                    usd_amount = coin['fiat_amount']
                    coin_amount = coin['coin_amount']
                    print(f"• {coin_name}: {percentage:.1f}% (${usd_amount:.2f} ≈ {coin_amount:.6f} {coin_name})")
                
                # Get historical performance
                performance = portfolio_manager.get_portfolio_performance(portfolio)
                
                if performance:
                    print("\nHistorical Performance (30 days):")
                    print(f"Initial investment: ${performance['initial_investment']:.2f}")
                    print(f"Current value: ${performance['current_value']:.2f}")
                    print(f"Overall change: {performance['percent_change']:.2f}%")
                    
                    # Show top and bottom performers
                    if performance['coin_performance']:
                        performances = sorted(performance['coin_performance'], key=lambda x: x['percent_change'], reverse=True)
                        
                        print("\nTop performer:")
                        top = performances[0]
                        print(f"{top['coin'].capitalize()}: {top['percent_change']:.2f}%")
                        
                        print("Bottom performer:")
                        bottom = performances[-1]
                        print(f"{bottom['coin'].capitalize()}: {bottom['percent_change']:.2f}%")
            else:
                print("Failed to generate portfolio recommendation.")
                
            continue
            
        elif user_input.lower().startswith("/analyze"):
            # Extract coin from command
            parts = user_input.split()
            if len(parts) > 1:
                coin = parts[1].lower()
                
                # Map abbreviations to full coin IDs
                coin_map = {
                    'btc': 'bitcoin', 'eth': 'ethereum', 'ada': 'cardano',
                    'sol': 'solana', 'dot': 'polkadot', 'xrp': 'ripple',
                    'doge': 'dogecoin', 'avax': 'avalanche-2', 'link': 'chainlink',
                    'matic': 'polygon', 'near': 'near'
                }
                
                coin_id = coin_map.get(coin, coin)
                
                # Get technical analysis
                print(f"Analyzing {coin_id.capitalize()}...")
                historical_data = fetch_historical_data(coin_id)
                
                if historical_data:
                    indicators = calculate_technical_indicators(historical_data)
                    
                    if indicators:
                        trend_result = get_trend_signal(
                            indicators['sma_7'], 
                            indicators['sma_30'], 
                            indicators['rsi'], 
                            indicators['macd'], 
                            indicators
                        )
                        
                        print(f"\n=== Technical Analysis: {coin_id.capitalize()} ===")
                        print(f"Current Price: ${indicators['current_price']:.2f} USD")
                        print(f"Signal: {trend_result['signal'].upper()} (Confidence: {trend_result['confidence']:.1f}%)")
                        
                        print("\nKey Indicators:")
                        if indicators['rsi'] is not None:
                            rsi_status = "OVERSOLD" if indicators['rsi'] < 30 else "OVERBOUGHT" if indicators['rsi'] > 70 else "NEUTRAL"
                            print(f"RSI: {indicators['rsi']:.1f} - {rsi_status}")
                            
                        if indicators['sma_7'] and indicators['sma_30']:
                            print(f"SMA 7: ${indicators['sma_7']:.2f}")
                            print(f"SMA 30: ${indicators['sma_30']:.2f}")
                            
                        if 'bollinger' in indicators and indicators['bollinger']['upper']:
                            print(f"Bollinger Upper: ${indicators['bollinger']['upper']:.2f}")
                            print(f"Bollinger Lower: ${indicators['bollinger']['lower']:.2f}")
                            
                        if 'momentum' in indicators:
                            for period, value in indicators['momentum'].items():
                                print(f"Momentum {period}: {value:.2f}%")
                    else:
                        print("Failed to calculate technical indicators.")
                else:
                    print("Failed to fetch historical data for this coin.")
            else:
                print("Please specify a coin, e.g., /analyze bitcoin")
                
            continue
            
        # Process normal queries
        response = crypto_buddy_response(user_input, user_profile)
        print("CryptoBuddy:", response)
        
        # Check for any triggered alerts after each response
        triggered_alerts = alert_manager.check_alerts()
        for _, alert, price in triggered_alerts:
            if alert.username == username:
                direction = "above" if alert.alert_type == "above" else "below"
                print(f"\n🔔 ALERT: {alert.coin_id.capitalize()} price is now {direction} ${alert.target_price} (Current: ${price})")
        
        # Respect CoinGecko's free API rate limit (30 calls/min)
        time.sleep(2)
