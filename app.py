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
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

def display_portfolio_analysis():
    st.subheader("Portfolio Analysis")
    
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

def display_price_alerts():
    st.subheader("Price Alerts")
    
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

def display_technical_analysis():
    st.subheader("Technical Analysis")
    
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
        tab1, tab2, tab3 = st.tabs(["Portfolio", "Price Alerts", "Technical Analysis"])
        
        with tab1:
            display_portfolio_analysis()
        
        with tab2:
            display_price_alerts()
        
        with tab3:
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
