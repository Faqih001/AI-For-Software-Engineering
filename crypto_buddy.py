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
