import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from pycoingecko import CoinGeckoAPI
from PIL import Image
import datetime
import json
import os
import re
import threading
import traceback
import random
import sys
from streamlit import config, set_page_config

# Import functionality from original crypto_buddy
from crypto_buddy import (
    PriceAlert,
    AlertManager,
    UserProfile,
    PortfolioManager,
    fetch_crypto_data,
    fetch_historical_data,
    calculate_technical_indicators,
    get_trend_signal,
    get_synonyms,
    interpret_query,
    get_greeting,
    get_disclaimer,
    get_trending_response,
    get_no_trending_response,
    get_sustainable_response,
    get_less_sustainable_response,
    get_longterm_response,
    get_no_longterm_response,
    get_general_response,
    setup_dependencies
)
from chat_styles import get_chat_styling

# Set up page config
st.set_page_config(
    page_title="CryptoBuddy - Your AI Crypto Advisor",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'alert_manager' not in st.session_state:
    st.session_state.alert_manager = AlertManager()
    st.session_state.alert_manager.load_alerts()
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()
    
# Chat state for each section
if 'portfolio_chat_history' not in st.session_state:
    st.session_state.portfolio_chat_history = []
if 'alerts_chat_history' not in st.session_state:
    st.session_state.alerts_chat_history = []
if 'technical_chat_history' not in st.session_state:
    st.session_state.technical_chat_history = []
if 'buddy_chat_history' not in st.session_state:
    st.session_state.buddy_chat_history = []

def initialize_nlp():
    try:
        setup_dependencies()
    except Exception as e:
        st.error(f"Error setting up NLP dependencies: {e}")

def create_profile():
    with st.form("profile_form"):
        username = st.text_input("Choose a username")
        risk_tolerance = st.selectbox("Risk Tolerance", ["low", "medium", "high"])
        sustainability_pref = st.selectbox("Sustainability Preference", ["low", "medium", "high"])
        investment_horizon = st.selectbox("Investment Horizon", ["short", "medium", "long"])
        favorite_coins = st.multiselect("Favorite Coins", 
            ["bitcoin", "ethereum", "cardano", "solana", "polkadot", "ripple", 
             "dogecoin", "avalanche-2", "chainlink", "polygon", "near"])
        
        submitted = st.form_submit_button("Create Profile")
        if submitted and username:
            profile = UserProfile(username)
            profile.set_risk_tolerance(risk_tolerance)
            profile.set_sustainability_preference(sustainability_pref)
            profile.set_investment_horizon(investment_horizon)
            profile.favorite_coins = favorite_coins
            profile.save_profile()
            st.session_state.user_profile = profile
            st.success("Profile created successfully! 👍")
            return True
    return False

def load_profile():
    with st.form("login_form"):
        username = st.text_input("Enter your username")
        submitted = st.form_submit_button("Load Profile")
        if submitted and username:
            profile = UserProfile.load_profile(username)
            st.session_state.user_profile = profile
            st.success(f"Welcome back, {profile.username}! 😊")
            return True
    return False

def handle_portfolio_chat(message):
    # Process the chat message and get portfolio-related response
    response = None
    try:
        # Use interpret_query and related functions to process portfolio queries
        if "portfolio" in message.lower() or "investment" in message.lower():
            # Add message to chat history
            st.session_state.portfolio_chat_history.append({"role": "user", "content": message})
            
            # Generate portfolio suggestion based on query
            if st.session_state.user_profile:
                # Parse investment amount if mentioned
                amount_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)', message)
                investment_amount = 1000  # default
                if amount_match:
                    investment_amount = float(amount_match.group(1).replace(',', ''))
                
                # Determine risk profile from message
                risk_profile = 'balanced'  # default
                if any(word in message.lower() for word in ['conservative', 'safe', 'low risk']):
                    risk_profile = 'conservative'
                elif any(word in message.lower() for word in ['aggressive', 'high risk', 'risky']):
                    risk_profile = 'aggressive'
                elif any(word in message.lower() for word in ['eco', 'sustainable', 'green']):
                    risk_profile = 'eco_friendly'
                
                # Generate portfolio suggestion
                portfolio = st.session_state.portfolio_manager.get_portfolio_suggestion(
                    risk_profile,
                    investment_amount
                )
                
                if portfolio:
                    response = f"Based on your query, here's a {risk_profile} portfolio suggestion for ${investment_amount}:\n\n"
                    for alloc in portfolio['allocations']:
                        response += f"• {alloc['coin'].capitalize()}: {alloc['allocation_percentage']:.1f}% (${alloc['fiat_amount']:.2f})\n"
                    
                    performance = st.session_state.portfolio_manager.get_portfolio_performance(portfolio)
                    if performance:
                        response += f"\nProjected 30-day performance based on historical data:\n"
                        response += f"• Initial Investment: ${performance['initial_investment']:.2f}\n"
                        response += f"• Current Value: ${performance['current_value']:.2f}\n"
                        response += f"• Overall Change: {performance['percent_change']:.2f}%"
            else:
                response = "Please log in or create a profile to get personalized portfolio suggestions."
        else:
            response = "I can help you analyze portfolios and create investment strategies. Try asking something like 'Create a conservative portfolio with $5000' or 'Show me an eco-friendly investment strategy'"
        
        if response:
            st.session_state.portfolio_chat_history.append({"role": "assistant", "content": response})
            
    except Exception as e:
        response = f"I encountered an error while processing your request: {str(e)}"
        st.session_state.portfolio_chat_history.append({"role": "assistant", "content": response})

def display_portfolio_analysis():
    st.subheader("Portfolio Analysis")
    
    # Chat interface
    st.write("💬 Chat with me about portfolio strategies!")
    
    # Add custom CSS for chat styling
    st.markdown("""
        <style>
        .user-message {
            padding: 10px;
            margin: 5px;
            border-radius: 15px;
            background-color: #e6f3ff;
            max-width: 80%;
            margin-left: auto;
            margin-right: 10px;
            text-align: right;
        }
        .assistant-message {
            padding: 10px;
            margin: 5px;
            border-radius: 15px;
            background-color: #f0f0f0;
            max-width: 80%;
            margin-right: auto;
            margin-left: 10px;
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display chat history with alternating styles
    for message in st.session_state.portfolio_chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if message := st.chat_input("Ask about portfolio strategies..."):
        handle_portfolio_chat(message)
    
    st.markdown("---")
    st.write("📊 Or use the traditional interface:")
    
    # Input for investment amount
    investment_amount = st.number_input("Investment Amount (USD)", min_value=100, value=1000, step=100)
    
    # Risk profile selection
    risk_profiles = {
        'Conservative': 'conservative',
        'Balanced': 'balanced',
        'Aggressive': 'aggressive',
        'Eco-Friendly': 'eco_friendly',
        'High-Cap': 'high_cap'
    }
    selected_profile = st.selectbox("Select Risk Profile", list(risk_profiles.keys()))
    
    if st.button("Generate Portfolio"):
        with st.spinner("Generating portfolio suggestion..."):
            portfolio = st.session_state.portfolio_manager.get_portfolio_suggestion(
                risk_profiles[selected_profile], 
                investment_amount
            )
            
            if portfolio:
                # Create allocation pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=[coin['coin'].capitalize() for coin in portfolio['allocations']],
                    values=[coin['allocation_percentage'] for coin in portfolio['allocations']],
                    hole=.3
                )])
                fig.update_layout(title="Portfolio Allocation")
                st.plotly_chart(fig)
                
                # Display allocations in a table
                allocations_df = pd.DataFrame([{
                    'Coin': coin['coin'].capitalize(),
                    'Percentage': f"{coin['allocation_percentage']:.1f}%",
                    'Amount (USD)': f"${coin['fiat_amount']:.2f}",
                    'Coin Amount': f"{coin['coin_amount']:.6f}"
                } for coin in portfolio['allocations']])
                
                st.table(allocations_df)
                
                # Get historical performance
                performance = st.session_state.portfolio_manager.get_portfolio_performance(portfolio)
                if performance:
                    st.subheader("30-Day Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Initial Investment", f"${performance['initial_investment']:.2f}")
                    with col2:
                        st.metric("Current Value", f"${performance['current_value']:.2f}")
                    with col3:
                        st.metric("Overall Change", f"{performance['percent_change']:.2f}%")
            else:
                st.error("Failed to generate portfolio recommendation.")

def handle_alerts_chat(message):
    # Process the chat message and get alerts-related response
    response = None
    try:
        if "alert" in message.lower() or "notify" in message.lower():
            # Add message to chat history
            st.session_state.alerts_chat_history.append({"role": "user", "content": message})
            
            if st.session_state.user_profile:
                # Extract coin name, alert type, and price from message
                coins = ["bitcoin", "ethereum", "cardano", "solana", "polkadot", "ripple", "dogecoin", "avalanche", "chainlink", "polygon", "near"]
                coin_id = None
                for coin in coins:
                    if coin in message.lower():
                        coin_id = coin if coin != "avalanche" else "avalanche-2"
                        break
                
                # Extract price
                price_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)', message)
                if price_match:
                    price = float(price_match.group(1).replace(',', ''))
                    
                    # Determine alert type
                    alert_type = 'above' if any(word in message.lower() for word in ['above', 'over', 'exceeds']) else 'below'
                    
                    if coin_id and price:
                        # Create alert
                        alert = PriceAlert(
                            coin_id,
                            price,
                            alert_type,
                            st.session_state.user_profile.username
                        )
                        st.session_state.alert_manager.add_alert(alert)
                        st.session_state.alert_manager.save_alerts()
                        
                        response = f"✅ Alert created! I'll notify you when {coin_id.capitalize()} {alert_type} ${price:.2f}"
                    else:
                        response = "I couldn't identify the cryptocurrency or price in your message. Please try again with something like 'Alert me when Bitcoin goes above $50,000'"
                else:
                    response = "Please include a price in your alert request, for example: 'Alert me when Ethereum falls below $2,000'"
            else:
                response = "Please log in or create a profile to set price alerts."
        else:
            response = "I can help you set price alerts for cryptocurrencies. Try saying something like 'Alert me when Bitcoin goes above $50,000' or 'Notify me if Ethereum drops below $2,000'"
        
        if response:
            st.session_state.alerts_chat_history.append({"role": "assistant", "content": response})
            
    except Exception as e:
        response = f"I encountered an error while processing your request: {str(e)}"
        st.session_state.alerts_chat_history.append({"role": "assistant", "content": response})

def display_price_alerts():
    st.subheader("Price Alerts")
    
    # Chat interface
    st.write("💬 Chat with me to set up price alerts!")
    
    # Display chat history
    for message in st.session_state.alerts_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if message := st.chat_input("Ask about setting price alerts..."):
        handle_alerts_chat(message)
    
    st.markdown("---")
    st.write("🔔 Or use the traditional interface:")
    
    if st.session_state.user_profile:
        # Show existing alerts
        user_alerts = st.session_state.alert_manager.get_alerts_by_username(st.session_state.user_profile.username)
        if user_alerts:
            for i, alert in enumerate(user_alerts):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    status = "✅ TRIGGERED" if alert.triggered else "⏳ ACTIVE"
                    direction = "rises above" if alert.alert_type == 'above' else "falls below"
                    st.write(f"{status} - {alert.coin_id.capitalize()} {direction} ${alert.target_price:.2f}")
                with col2:
                    if not alert.triggered and st.button(f"Delete Alert {i+1}"):
                        st.session_state.alert_manager.remove_alert(i)
                        st.session_state.alert_manager.save_alerts()
                        st.experimental_rerun()
        else:
            st.info("No active alerts. Create one below!")
        
        # Create new alert
        with st.form("create_alert"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                coin = st.selectbox("Select Coin", [
                    "Bitcoin", "Ethereum", "Cardano", "Solana", "Polkadot",
                    "Ripple", "Dogecoin", "Avalanche", "Chainlink", "Polygon", "Near"
                ])
            with col2:
                alert_type = st.selectbox("Alert Type", ["Above", "Below"])
            with col3:
                price = st.number_input("Target Price", min_value=0.0, step=100.0)
            
            if st.form_submit_button("Create Alert"):
                coin_id = coin.lower().replace(" ", "-")
                alert = PriceAlert(
                    coin_id,
                    price,
                    alert_type.lower(),
                    st.session_state.user_profile.username
                )
                st.session_state.alert_manager.add_alert(alert)
                st.session_state.alert_manager.save_alerts()
                st.success("Alert created successfully!")
                st.experimental_rerun()
    else:
        st.warning("Please log in to manage price alerts.")

def handle_technical_chat(message):
    # Process the chat message and get technical analysis response
    response = None
    try:
        if any(word in message.lower() for word in ['analysis', 'trend', 'price', 'indicator', 'moving average', 'rsi']):
            # Add message to chat history
            st.session_state.technical_chat_history.append({"role": "user", "content": message})
            
            # Extract coin name from message
            coins = {
                "bitcoin": "bitcoin", "btc": "bitcoin",
                "ethereum": "ethereum", "eth": "ethereum",
                "cardano": "cardano", "ada": "cardano",
                "solana": "solana", "sol": "solana",
                "polkadot": "polkadot", "dot": "polkadot",
                "ripple": "ripple", "xrp": "ripple",
                "dogecoin": "dogecoin", "doge": "dogecoin",
                "avalanche": "avalanche-2", "avax": "avalanche-2",
                "chainlink": "chainlink", "link": "chainlink",
                "polygon": "polygon", "matic": "polygon",
                "near": "near"
            }
            
            coin_id = None
            for key, value in coins.items():
                if key in message.lower():
                    coin_id = value
                    break
            
            # Determine timeframe
            days = 30  # default
            if "week" in message.lower() or "7" in message:
                days = 7
            elif "month" in message.lower() or "30" in message:
                days = 30
            elif "90" in message or "quarter" in message.lower():
                days = 90
            
            if coin_id:
                # Fetch data and calculate indicators
                historical_data = fetch_historical_data(coin_id, days)
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
                        
                        response = f"📊 {coin_id.capitalize()} Technical Analysis ({days} days):\n\n"
                        response += f"• Current Price: ${indicators['current_price']:.2f}\n"
                        response += f"• Signal: {trend_result['signal'].upper()}\n"
                        response += f"• Confidence: {trend_result['confidence']:.1f}%\n\n"
                        
                        if indicators['rsi'] is not None:
                            rsi_status = "OVERSOLD" if indicators['rsi'] < 30 else "OVERBOUGHT" if indicators['rsi'] > 70 else "NEUTRAL"
                            response += f"• RSI: {indicators['rsi']:.1f} ({rsi_status})\n"
                        
                        if 'momentum' in indicators:
                            response += f"• 7-Day Momentum: {indicators['momentum']['7d']:.2f}%\n"
                        
                        response += f"\n{trend_result['explanation']}"
                    else:
                        response = "Sorry, I couldn't calculate the technical indicators. Please try again later."
                else:
                    response = "Sorry, I couldn't fetch the historical data. Please try again later."
            else:
                response = "Please specify a cryptocurrency in your question. For example: 'What's the technical analysis for Bitcoin?' or 'Show me Ethereum's trend'"
        else:
            response = "I can help you analyze cryptocurrency trends and technical indicators. Try asking something like 'What's the technical analysis for Bitcoin?' or 'Show me Ethereum's trend'"
        
        if response:
            st.session_state.technical_chat_history.append({"role": "assistant", "content": response})
            
    except Exception as e:
        response = f"I encountered an error while processing your request: {str(e)}"
        st.session_state.technical_chat_history.append({"role": "assistant", "content": response})

def display_technical_analysis():
    st.subheader("Technical Analysis")
    
    # Chat interface
    st.write("💬 Chat with me about technical analysis!")
    
    # Display chat history
    for message in st.session_state.technical_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if message := st.chat_input("Ask about technical analysis..."):
        handle_technical_chat(message)
    
    st.markdown("---")
    st.write("📈 Or use the traditional interface:")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        coin = st.selectbox("Select Cryptocurrency", [
            "Bitcoin", "Ethereum", "Cardano", "Solana", "Polkadot",
            "Ripple", "Dogecoin", "Avalanche", "Chainlink", "Polygon", "Near"
        ])
    with col2:
        timeframe = st.selectbox("Timeframe", ["7 days", "30 days", "90 days"])
    
    if st.button("Analyze"):
        with st.spinner(f"Analyzing {coin}..."):
            days = 30
            if timeframe == "7 days":
                days = 7
            elif timeframe == "90 days":
                days = 90
            
            coin_id = coin.lower().replace(" ", "-")
            if coin_id == "avalanche":
                coin_id = "avalanche-2"
            
            historical_data = fetch_historical_data(coin_id, days)
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
                    
                    # Display current metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${indicators['current_price']:.2f}")
                    with col2:
                        st.metric("Signal", trend_result['signal'].upper())
                    with col3:
                        st.metric("Confidence", f"{trend_result['confidence']:.1f}%")
                    
                    # Display key indicators
                    st.subheader("Key Indicators")
                    col1, col2 = st.columns(2)
                    with col1:
                        if indicators['rsi'] is not None:
                            rsi_status = "OVERSOLD" if indicators['rsi'] < 30 else "OVERBOUGHT" if indicators['rsi'] > 70 else "NEUTRAL"
                            st.metric("RSI", f"{indicators['rsi']:.1f}", rsi_status)
                    with col2:
                        if 'momentum' in indicators and '7d' in indicators['momentum']:
                            st.metric("7-Day Momentum", f"{indicators['momentum']['7d']:.2f}%")
                    
                    # Plot price and moving averages
                    if historical_data['prices']:
                        df = pd.DataFrame(historical_data['prices'], columns=['date', 'price'])
                        df['date'] = pd.to_datetime(df['date'], unit='ms')
                        df.set_index('date', inplace=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['price'],
                            name="Price",
                            line=dict(color='blue')
                        ))
                        
                        if indicators['sma_7'] and indicators['sma_30']:
                            fig.add_trace(go.Scatter(
                                x=[df.index[-1]],
                                y=[indicators['sma_7']],
                                name="7-Day MA",
                                line=dict(color='orange')
                            ))
                            fig.add_trace(go.Scatter(
                                x=[df.index[-1]],
                                y=[indicators['sma_30']],
                                name="30-Day MA",
                                line=dict(color='red')
                            ))
                        
                        fig.update_layout(
                            title=f"{coin} Price Chart ({timeframe})",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)"
                        )
                        st.plotly_chart(fig)
                else:
                    st.error("Failed to calculate technical indicators.")
            else:
                st.error("Failed to fetch historical data.")

def handle_buddy_chat(message):
    # Process the chat message and get a response using all crypto buddy functionalities
    response = None
    try:
        # Add message to chat history
        st.session_state.buddy_chat_history.append({"role": "user", "content": message})
        
        # Interpret the user's query
        intent = interpret_query(message.lower())
        disclaimer = get_disclaimer()
        
        if intent:
            if intent.get('is_greeting', False):
                response = f"{get_greeting()}\n\nI can help you with:\n"
                response += "• Portfolio management and suggestions\n"
                response += "• Price alerts and monitoring\n"
                response += "• Technical analysis and trends\n"
                response += "• Cryptocurrency market insights"
                
            elif intent.get('wants_trending', False):
                trending_response = get_trending_response() if 'trending' in intent else get_no_trending_response()
                response = f"🔥 Trending Analysis:\n\n{trending_response}"
                
            elif intent.get('wants_sustainable', False):
                sustain_response = get_sustainable_response() if 'sustainable' in intent else get_less_sustainable_response()
                response = f"🌱 Sustainability Analysis:\n\n{sustain_response}"
                
            elif intent.get('wants_longterm', False):
                longterm_response = get_longterm_response() if 'longterm' in intent else get_no_longterm_response()
                response = f"📈 Long-term Investment Perspective:\n\n{longterm_response}"
                
            elif "market" in message.lower() or "outlook" in message.lower():
                # Add market overview using available data
                coins = ["bitcoin", "ethereum", "cardano", "solana"]
                market_response = "📊 Current Market Overview:\n\n"
                
                for coin in coins:
                    data = fetch_crypto_data([coin])
                    if data and coin in data:
                        price = data[coin]['usd']
                        change_24h = data[coin].get('usd_24h_change', 0)
                        change_emoji = "📈" if change_24h > 0 else "📉"
                        market_response += f"{change_emoji} {coin.capitalize()}: ${price:,.2f} ({change_24h:+.2f}%)\n"
                
                response = market_response
                
            elif "portfolio" in intent or "investment" in intent:
                if st.session_state.user_profile:
                    # Extract investment amount
                    amount_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)', message)
                    investment_amount = 1000  # default
                    if amount_match:
                        investment_amount = float(amount_match.group(1).replace(',', ''))
                    
                    # Determine risk profile from message and user preferences
                    risk_profile = 'balanced'  # default
                    user_risk = st.session_state.user_profile.risk_tolerance
                    
                    if any(word in message.lower() for word in ['conservative', 'safe', 'low risk']):
                        risk_profile = 'conservative'
                    elif any(word in message.lower() for word in ['aggressive', 'high risk', 'risky']):
                        risk_profile = 'aggressive'
                    elif any(word in message.lower() for word in ['eco', 'sustainable', 'green']):
                        risk_profile = 'eco_friendly'
                    elif user_risk == 'low':
                        risk_profile = 'conservative'
                    elif user_risk == 'high':
                        risk_profile = 'aggressive'
                    
                    portfolio = st.session_state.portfolio_manager.get_portfolio_suggestion(
                        risk_profile,
                        investment_amount
                    )
                    
                    if portfolio:
                        response = f"💼 {risk_profile.capitalize()} Portfolio (${investment_amount:,.2f})\n\n"
                        response += "Allocation Strategy:\n"
                        total_allocation = 0
                        for alloc in portfolio['allocations']:
                            response += f"• {alloc['coin'].capitalize()}: {alloc['allocation_percentage']:.1f}% (${alloc['fiat_amount']:,.2f})\n"
                            total_allocation += alloc['allocation_percentage']
                        
                        performance = st.session_state.portfolio_manager.get_portfolio_performance(portfolio)
                        if performance:
                            response += f"\n📈 30-day Performance Analysis:\n"
                            response += f"• Initial: ${performance['initial_investment']:,.2f}\n"
                            response += f"• Current: ${performance['current_value']:,.2f}\n"
                            response += f"• Change: {performance['percent_change']:+.2f}%\n\n"
                            
                            # Add coin-specific performance
                            best_performer = max(performance['coin_performance'], key=lambda x: x['percent_change'])
                            worst_performer = min(performance['coin_performance'], key=lambda x: x['percent_change'])
                            
                            response += "Notable Performers:\n"
                            response += f"⭐ Best: {best_performer['coin'].capitalize()} ({best_performer['percent_change']:+.2f}%)\n"
                            response += f"⚠️ Challenging: {worst_performer['coin'].capitalize()} ({worst_performer['percent_change']:+.2f}%)\n\n"
                            
                            # Add personalized advice based on performance and user profile
                            response += "💡 Personalized Insights:\n"
                            if performance['percent_change'] > 10:
                                response += "• Consider taking some profits while maintaining core positions\n"
                                if user_risk == 'low':
                                    response += "• Given your conservative profile, consider rebalancing to lock in gains\n"
                            elif performance['percent_change'] < -10:
                                response += "• This could be a buying opportunity if you believe in long-term potential\n"
                                if user_risk == 'high':
                                    response += "• Given your risk tolerance, consider averaging down on strong assets\n"
                            
                            # Add sustainability note if relevant
                            if st.session_state.user_profile.sustainability_preference == 'high':
                                response += "\n🌱 Sustainability Note: Consider increasing allocation to eco-friendly cryptocurrencies"
                else:
                    response = "Please log in or create a profile to get personalized portfolio suggestions."
                
            elif "alert" in intent or "notify" in intent:
                if st.session_state.user_profile:
                    coins = {
                        "bitcoin": "bitcoin", "btc": "bitcoin",
                        "ethereum": "ethereum", "eth": "ethereum",
                        "cardano": "cardano", "ada": "cardano",
                        "solana": "solana", "sol": "solana",
                        "polkadot": "polkadot", "dot": "polkadot",
                        "ripple": "ripple", "xrp": "ripple",
                        "dogecoin": "dogecoin", "doge": "dogecoin",
                        "avalanche": "avalanche-2", "avax": "avalanche-2",
                        "chainlink": "chainlink", "link": "chainlink",
                        "polygon": "polygon", "matic": "polygon",
                        "near": "near"
                    }
                    
                    coin_id = None
                    for key, value in coins.items():
                        if key in message.lower():
                            coin_id = value
                            break
                    
                    price_match = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d{1,2})?)', message)
                    if coin_id and price_match:
                        price = float(price_match.group(1).replace(',', ''))
                        alert_type = 'above' if any(word in message.lower() for word in ['above', 'over', 'exceeds', 'reaches', 'hits']) else 'below'
                        
                        # Get current price for context
                        current_data = fetch_crypto_data([coin_id])
                        if current_data and coin_id in current_data:
                            current_price = current_data[coin_id]['usd']
                            price_diff = abs(price - current_price)
                            price_diff_percent = (price_diff / current_price) * 100
                            
                            # Get existing alerts for this coin
                            existing_alerts = [a for a in st.session_state.alert_manager.get_alerts_by_username(
                                st.session_state.user_profile.username) if a.coin_id == coin_id]
                            
                            alert = PriceAlert(coin_id, price, alert_type, st.session_state.user_profile.username)
                            st.session_state.alert_manager.add_alert(alert)
                            st.session_state.alert_manager.save_alerts()
                            
                            response = f"✅ Alert Set: {coin_id.capitalize()}\n\n"
                            response += f"I'll notify you when the price goes {alert_type} ${price:,.2f}\n"
                            response += f"Current price: ${current_price:,.2f}\n"
                            if alert_type == 'above':
                                response += f"Needed change: +${price_diff:,.2f} ({price_diff_percent:+.1f}%)\n"
                            else:
                                response += f"Needed change: -${price_diff:,.2f} ({price_diff_percent:+.1f}%)\n"
                            
                            if existing_alerts:
                                response += f"\nℹ️ You have {len(existing_alerts)} other alert(s) for {coin_id.capitalize()}:"
                                for a in existing_alerts:
                                    response += f"\n• {a.alert_type.capitalize()}: ${a.target_price:,.2f}"
                    else:
                        response = "I need both a cryptocurrency and a target price to set an alert. Try something like:\n"
                        response += "• 'Alert me when Bitcoin goes above $50,000'\n"
                        response += "• 'Notify me if ETH drops below $2,000'"
                else:
                    response = "Please log in or create a profile to set price alerts."
                
            elif "analysis" in intent or "trend" in intent or "price" in intent:
                coins = {
                    "bitcoin": "bitcoin", "btc": "bitcoin",
                    "ethereum": "ethereum", "eth": "ethereum",
                    "cardano": "cardano", "ada": "cardano",
                    "solana": "solana", "sol": "solana",
                    "polkadot": "polkadot", "dot": "polkadot",
                    "ripple": "ripple", "xrp": "ripple",
                    "dogecoin": "dogecoin", "doge": "dogecoin",
                    "avalanche": "avalanche-2", "avax": "avalanche-2",
                    "chainlink": "chainlink", "link": "chainlink",
                    "polygon": "polygon", "matic": "polygon",
                    "near": "near"
                }
                
                coin_id = None
                for key, value in coins.items():
                    if key in message.lower():
                        coin_id = value
                        break
                
                if coin_id:
                    days = 30  # default
                    if "week" in message.lower() or "7" in message:
                        days = 7
                    elif "month" in message.lower() or "30" in message:
                        days = 30
                    elif "90" in message or "quarter" in message.lower():
                        days = 90
                        
                    historical_data = fetch_historical_data(coin_id, days)
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
                            
                            # Create comprehensive analysis response
                            response = f"📊 {coin_id.capitalize()} Analysis ({days} days)\n\n"
                            
                            # Price information
                            response += f"💰 Price Metrics:\n"
                            response += f"• Current: ${indicators['current_price']:,.2f}\n"
                            if 'momentum' in indicators:
                                for period, value in indicators['momentum'].items():
                                    response += f"• {period.upper()} Change: {value:+.2f}%\n"
                            
                            # Technical indicators
                            response += f"\n📈 Technical Signals:\n"
                            
                            # Moving Averages
                            ma_trend = "BULLISH 📈" if indicators['sma_7'] > indicators['sma_30'] else "BEARISH 📉"
                            response += f"• Moving Averages: {ma_trend}\n"
                            response += f"• Signal: {trend_result['signal'].upper()}\n"
                            response += f"• Confidence: {trend_result['confidence']:.1f}%\n"
                            
                            if indicators['rsi'] is not None:
                                rsi = indicators['rsi']
                                rsi_emoji = "⚠️" if rsi < 30 or rsi > 70 else "✅"
                                rsi_status = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
                                response += f"• RSI: {rsi:.1f} {rsi_emoji} ({rsi_status})\n"
                            
                            if indicators['macd'] is not None:
                                macd_signal = "BULLISH 📈" if indicators['macd'] > 0 else "BEARISH 📉"
                                response += f"• MACD: {macd_signal}\n"
                            
                            # Volume analysis
                            if 'volume_change' in indicators:
                                vol_change = indicators['volume_change']
                                vol_emoji = "📈" if vol_change > 0 else "📉"
                                response += f"\n📊 Volume Analysis:\n"
                                response += f"• 24h Change: {vol_change:+.2f}% {vol_emoji}\n"
                            
                            # Support and Resistance
                            if 'support' in indicators and 'resistance' in indicators:
                                response += f"\n📍 Key Levels:\n"
                                response += f"• Support: ${indicators['support']:,.2f}\n"
                                response += f"• Resistance: ${indicators['resistance']:,.2f}\n"
                            
                            # Trading insights
                            response += f"\n💡 Trading Insights:\n"
                            response += trend_result['explanation'] + "\n"
                            
                            # Risk warning for volatile conditions
                            if abs(indicators.get('volatility', 0)) > 5:
                                response += "\n⚠️ High volatility detected - consider:"
                                response += "\n• Using stop-loss orders"
                                response += "\n• Trading smaller position sizes"
                                response += "\n• Implementing a scaled entry strategy"
                            
                            # Add disclaimer
                            response += f"\n\n{disclaimer}"
                        else:
                            response = "Sorry, I couldn't calculate the technical indicators. Please try again later."
                    else:
                        response = "Sorry, I couldn't fetch the historical data. Please try again later."
                else:
                    response = "Which cryptocurrency would you like me to analyze? Try asking about specific coins like BTC, ETH, or SOL."
            
            else:
                # General response with helpful suggestions
                response = "I can help you with:\n\n"
                response += "📊 Portfolio Management:\n"
                response += "• 'Create a conservative portfolio with $5000'\n"
                response += "• 'Show me an eco-friendly investment strategy'\n\n"
                response += "🔔 Price Alerts:\n"
                response += "• 'Alert me when Bitcoin goes above $50,000'\n"
                response += "• 'Notify me if ETH drops below $2,000'\n\n"
                response += "📈 Technical Analysis:\n"
                response += "• 'Analyze Bitcoin's trend'\n"
                response += "• 'Show me ETH technical indicators'\n\n"
                response += "💡 General Info:\n"
                response += "• 'What's trending in crypto?'\n"
                response += "• 'Tell me about sustainable cryptocurrencies'\n\n"
                response += "🌐 Market Overview:\n"
                response += "• 'Show me the current market outlook'\n"
                response += "• 'How are the top cryptocurrencies performing?'"

            # Add response to chat history
            if response:
                st.session_state.buddy_chat_history.append({"role": "assistant", "content": response})
            
    except Exception as e:
        response = f"I encountered an error while processing your request: {str(e)}"
        st.session_state.buddy_chat_history.append({"role": "assistant", "content": response})

def display_buddy_chat():
    st.subheader("Chat with CryptoBuddy")
    
    # Welcome message
    st.write("👋 Hi! I'm your AI Crypto Advisor. I can help you with:")
    st.write("• 📊 Portfolio suggestions and analysis")
    st.write("• 🔔 Setting up price alerts")
    st.write("• 📈 Technical analysis and trends")
    st.write("• 💡 General cryptocurrency advice")
    
    # Display chat history
    for message in st.session_state.buddy_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if message := st.chat_input("Ask me anything about crypto..."):
        handle_buddy_chat(message)

def main():
    st.title("CryptoBuddy - Your AI Crypto Advisor 🤖")
    
    # Sidebar for user profile
    with st.sidebar:
        st.subheader("User Profile")
        if st.session_state.user_profile:
            st.write(f"Welcome, {st.session_state.user_profile.username}! 👋")
            st.write("Profile Settings:")
            st.write(f"Risk Tolerance: {st.session_state.user_profile.risk_tolerance}")
            st.write(f"Sustainability: {st.session_state.user_profile.sustainability_preference}")
            st.write(f"Investment Horizon: {st.session_state.user_profile.investment_horizon}")
            if st.button("Logout"):
                st.session_state.user_profile = None
                st.experimental_rerun()
        else:
            tab1, tab2 = st.tabs(["Login", "Create Profile"])
            with tab1:
                load_profile()
            with tab2:
                create_profile()
    
    # Main content tabs
    if st.session_state.user_profile:
        tab1, tab2, tab3, tab4 = st.tabs(["CryptoBuddy Chat", "Portfolio", "Price Alerts", "Technical Analysis"])
        
        with tab1:
            display_buddy_chat()
            
        with tab2:
            display_portfolio_analysis()
        
        with tab3:
            display_price_alerts()
        
        with tab4:
            display_technical_analysis()
    else:
        st.info("👋 Welcome to CryptoBuddy! Please log in or create a profile to get started.")
        st.write("""
        CryptoBuddy helps you:
        - 📊 Get personalized cryptocurrency recommendations
        - 💰 Create diversified portfolio suggestions
        - 🔔 Set price alerts for your favorite coins
        - 📈 Analyze market trends and technical indicators
        """)

if __name__ == "__main__":
    initialize_nlp()
    main()
