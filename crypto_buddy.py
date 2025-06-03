#!/usr/bin/env python3
# CryptoBuddy Chatbot with CoinGecko API and NLTK Integration
# A rule-based cryptocurrency advisor with NLP for natural user queries.

import sys
import time
import traceback
import random

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
    
    # Fetch real-time data for Bitcoin, Ethereum, and Cardano
    coins = ['bitcoin', 'ethereum', 'cardano']
    try:
        data = cg.get_price(
            ids=coins,
            vs_currencies='usd',
            include_market_cap=True,
            include_24hr_change=True
        )
        
        # Mock sustainability data (since CoinGecko doesn't provide this)
        sustainability_data = {
            'bitcoin': {'energy_use': 'high', 'sustainability_score': 3/10},
            'ethereum': {'energy_use': 'medium', 'sustainability_score': 6/10},
            'cardano': {'energy_use': 'low', 'sustainability_score': 8/10}
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

def crypto_buddy_response(user_query):
    """Generate chatbot response based on user query."""
    # Fetch real-time data
    crypto_db = fetch_crypto_data()
    if not crypto_db:
        return "Oops! Couldn't fetch data from CoinGecko. Try again later! 😅"

    # Interpret query intent using NLTK
    intent = interpret_query(user_query)
    
    # Friendly greeting and ethics disclaimer
    greeting = get_greeting()
    disclaimer = get_disclaimer()
    
    # Initialize response
    response = greeting + disclaimer
    
    # Logic for handling user intents
    if intent == 'trending':
        # Find coins with rising price trend and high market cap
        rising_coins = [
            coin for coin in crypto_db 
            if crypto_db[coin]["price_trend"] == "rising" and crypto_db[coin]["market_cap"] == "high"
        ]
        if rising_coins:
            response += f"For profitability, I recommend {rising_coins[0]}! 🚀 It's trending up with a strong market cap.\n"
        else:
            response += "Hmm, no coins match both rising trends and high market cap right now. Try Cardano for a rising star! 🌱\n"
    
    elif intent == 'sustainable':
        # Find the most sustainable coin
        recommend = max(crypto_db, key=lambda x: crypto_db[x]["sustainability_score"])
        if crypto_db[recommend]["sustainability_score"] > 7/10:
            response += f"Invest in {recommend}! 🌱 It's eco-friendly with a sustainability score of {crypto_db[recommend]['sustainability_score']*10}/10!\n"
        else:
            response += f"{recommend} is the most sustainable with a score of {crypto_db[recommend]['sustainability_score']*10}/10, but explore more options for greener choices! 🌍\n"
    
    elif intent == 'longterm':
        # Recommend coins with rising trend and high sustainability
        growth_coins = [
            coin for coin in crypto_db 
            if crypto_db[coin]["price_trend"] == "rising" and crypto_db[coin]["sustainability_score"] > 7/10
        ]
        if growth_coins:
            response += f"For long-term growth, go with {growth_coins[0]}! 🚀 It's trending up and has a sustainability score of {crypto_db[growth_coins[0]]['sustainability_score']*10}/10!\n"
        else:
            response += "Cardano looks promising for long-term growth with its rising trend and eco-friendly vibe! 🌱\n"
    
    else:  # General intent
        # General recommendation based on balanced criteria
        recommend = max(crypto_db, key=lambda x: (crypto_db[x]["sustainability_score"], 1 if crypto_db[x]["price_trend"] == "rising" else 0))
        response += f"I'd suggest {recommend}! It balances profitability (trend: {crypto_db[recommend]['price_trend']}) and sustainability (score: {crypto_db[recommend]['sustainability_score']*10}/10). 🚀🌱\n"
    
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
    print("===============================================\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("CryptoBuddy: Catch you later! Stay savvy! 👋")
            break
        response = crypto_buddy_response(user_input)
        print("CryptoBuddy:", response)
        # Respect CoinGecko's free API rate limit (30 calls/min)
        time.sleep(2)

# Entry point
if __name__ == "__main__":
    setup_dependencies()
    run_crypto_buddy()
