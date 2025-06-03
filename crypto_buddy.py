#!/usr/bin/env python3
# CryptoBuddy Chatbot with CoinGecko API and NLTK Integration
# A rule-based cryptocurrency advisor with NLP for natural user queries.

import sys
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from pycoingecko import CoinGeckoAPI

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
            
        # Download required NLTK data
        print("Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("Setup complete!")
    except Exception as e:
        print(f"Setup failed: {e}")
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
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def interpret_query(user_query):
    """Interpret user query intent using NLP."""
    # Tokenize and tag the query
    tokens = word_tokenize(user_query.lower())
    tagged = pos_tag(tokens)
    
    # Define keywords and their synonyms
    trend_keywords = {'trend', 'trending', 'rise', 'rising', 'up', 'growth', 'profitable', 'profit'}
    sustain_keywords = {'sustainable', 'sustainability', 'eco', 'ecofriendly', 'green', 'environment'}
    longterm_keywords = {'longterm', 'long', 'future', 'growth', 'potential'}
    general_keywords = {'which', 'what', 'recommend', 'suggest', 'best', 'good'}
    
    # Expand keywords with synonyms
    trend_synonyms = set()
    sustain_synonyms = set()
    longterm_synonyms = set()
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
    greeting = "Hey there! I'm CryptoBuddy, your crypto sidekick! 🌟\n⚠️ Crypto is risky—always do your own research!\n"
    
    # Initialize response
    response = greeting
    
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
