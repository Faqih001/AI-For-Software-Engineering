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
    crypto_buddy_response
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
if 'crypto_data' not in st.session_state or st.session_state.crypto_data is None:
    with st.spinner('Fetching cryptocurrency data...'):
        try:
            st.session_state.crypto_data = fetch_crypto_data()
            if not st.session_state.crypto_data:
                st.error("Failed to fetch cryptocurrency data. API may be rate-limited. Please try again later.")
        except Exception as e:
            st.error(f"Error fetching cryptocurrency data: {str(e)}")
            st.session_state.crypto_data = None

# Error handling function for API calls
@st.cache_data(ttl=60)  # Cache for 60 seconds
def safe_api_call(func, *args, **kwargs):
    """Execute API calls with proper error handling and caching."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"API Error: {str(e)}. CoinGecko might be rate-limiting requests.")
        return None

# Setup NLTK Data
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

setup_nltk()

# App title and styling
st.title("💰 CryptoBuddy")
st.markdown("### Your AI Cryptocurrency Advisor")

# Create tabs for different features
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Profile", "Technical Analysis", "Portfolio", "Price Alerts"])

# Sidebar for app info and branding
with st.sidebar:
    st.header("About CryptoBuddy")
    st.info("""
    CryptoBuddy is your intelligent cryptocurrency assistant.
    
    Ask me about:
    - Crypto recommendations
    - Technical analysis
    - Portfolio suggestions
    - Price alerts
    
    Data powered by CoinGecko API
    """)
    
    # Refresh data button
    if st.button("🔄 Refresh Market Data"):
        with st.spinner('Updating cryptocurrency data...'):
            st.session_state.crypto_data = fetch_crypto_data()
            st.success("Market data has been refreshed!")

# Tab 1: Chat Interface
with tab1:
    st.header("💬 Chat with CryptoBuddy")
    st.markdown("Ask me anything about cryptocurrencies!")
    
    # Display chat disclaimer
    st.info("⚠️ Disclaimer: This is not financial advice. Always do your own research before investing!")
    
    # Chat input
    user_input = st.text_input("Your question:", placeholder="Example: What's a good cryptocurrency to invest in?")
    
    # When the user submits a question
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"user": user_input})
        
        # Get response from CryptoBuddy
        with st.spinner('CryptoBuddy is thinking...'):
            response = crypto_buddy_response(user_input, st.session_state.user_profile)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"bot": response})
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in reversed(st.session_state.chat_history):
            if "user" in message:
                st.markdown(f"**You**: {message['user']}")
            if "bot" in message:
                st.markdown(f"**CryptoBuddy**: {message['bot']}")
            st.markdown("---")

# Tab 2: User Profile
with tab2:
    st.header("👤 User Profile")
    
    # User login/profile creation
    username = st.text_input("Username:", placeholder="Enter your username")
    
    if st.button("Load Profile"):
        if username:
            with st.spinner('Loading profile...'):
                st.session_state.user_profile = UserProfile.load_profile(username)
                st.success(f"Welcome, {username}! Your profile has been loaded.")
        else:
            st.error("Please enter a username.")
    
    # Display and edit profile if loaded
    if st.session_state.user_profile:
        st.subheader("Your Profile Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk tolerance selection
            risk_options = {"low": "Low Risk (Conservative)", 
                           "medium": "Medium Risk (Balanced)", 
                           "high": "High Risk (Aggressive)"}
            
            selected_risk = st.selectbox(
                "Risk Tolerance:",
                options=list(risk_options.keys()),
                format_func=lambda x: risk_options[x],
                index=list(risk_options.keys()).index(st.session_state.user_profile.risk_tolerance)
            )
            
            # Sustainability preference selection
            sustain_options = {"low": "Low Priority", 
                              "medium": "Medium Priority", 
                              "high": "High Priority"}
            
            selected_sustain = st.selectbox(
                "Sustainability Preference:",
                options=list(sustain_options.keys()),
                format_func=lambda x: sustain_options[x],
                index=list(sustain_options.keys()).index(st.session_state.user_profile.sustainability_preference)
            )
        
        with col2:
            # Investment horizon selection
            horizon_options = {"short": "Short Term (< 1 year)", 
                              "medium": "Medium Term (1-3 years)", 
                              "long": "Long Term (> 3 years)"}
            
            selected_horizon = st.selectbox(
                "Investment Horizon:",
                options=list(horizon_options.keys()),
                format_func=lambda x: horizon_options[x],
                index=list(horizon_options.keys()).index(st.session_state.user_profile.investment_horizon)
            )
            
            # Favorite coins
            favorite_coins = st.text_input(
                "Favorite Coins (comma-separated):",
                value=",".join(st.session_state.user_profile.favorite_coins) if st.session_state.user_profile.favorite_coins else ""
            )
        
        # Save profile button
        if st.button("Save Profile"):
            # Update profile with new values
            st.session_state.user_profile.set_risk_tolerance(selected_risk)
            st.session_state.user_profile.set_sustainability_preference(selected_sustain)
            st.session_state.user_profile.set_investment_horizon(selected_horizon)
            
            # Parse and set favorite coins
            if favorite_coins:
                st.session_state.user_profile.favorite_coins = [coin.strip() for coin in favorite_coins.split(',')]
            else:
                st.session_state.user_profile.favorite_coins = []
            
            # Save the profile
            if st.session_state.user_profile.save_profile():
                st.success("Profile saved successfully!")
            else:
                st.error("Failed to save profile.")
    else:
        st.info("Please enter a username and load your profile.")

# Tab 3: Technical Analysis
with tab3:
    st.header("📊 Technical Analysis")
    
    # Coin selection for analysis
    coin_options = {
        'bitcoin': 'Bitcoin (BTC)',
        'ethereum': 'Ethereum (ETH)',
        'cardano': 'Cardano (ADA)',
        'solana': 'Solana (SOL)',
        'polkadot': 'Polkadot (DOT)',
        'ripple': 'Ripple (XRP)',
        'dogecoin': 'Dogecoin (DOGE)',
        'avalanche-2': 'Avalanche (AVAX)',
        'chainlink': 'Chainlink (LINK)',
        'polygon': 'Polygon (MATIC)',
        'near': 'Near Protocol (NEAR)'
    }
    
    selected_coin = st.selectbox(
        "Select a cryptocurrency:",
        options=list(coin_options.keys()),
        format_func=lambda x: coin_options[x]
    )
    
    period_options = {
        7: "7 days",
        14: "14 days",
        30: "30 days",
        90: "90 days",
        180: "180 days",
        365: "1 year"
    }
    
    selected_period = st.selectbox(
        "Select time period:",
        options=list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=2  # Default to 30 days
    )
    
    if st.button("Analyze"):
        with st.spinner(f'Analyzing {coin_options[selected_coin]}...'):
            # Fetch historical data with safe API call
            historical_data = safe_api_call(fetch_historical_data, selected_coin, selected_period)
            
            if historical_data and 'prices' in historical_data:
                # Calculate technical indicators
                indicators = calculate_technical_indicators(historical_data)
                
                if indicators:
                    # Get trend signal
                    trend_result = get_trend_signal(
                        indicators['sma_7'], 
                        indicators['sma_30'], 
                        indicators['rsi'], 
                        indicators['macd'], 
                        indicators
                    )
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create price chart with moving averages
                        price_data = historical_data['prices']
                        df = pd.DataFrame(price_data, columns=['timestamp', 'price'])
                        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        fig = go.Figure()
                        
                        # Add price line
                        fig.add_trace(go.Scatter(
                            x=df['date'], 
                            y=df['price'],
                            mode='lines',
                            name='Price',
                            line=dict(color='#5B21B6', width=2)
                        ))
                        
                        # Add SMA lines if available
                        if indicators['sma_7'] is not None:
                            # Calculate SMA for each point
                            df['sma7'] = df['price'].rolling(window=7).mean()
                            fig.add_trace(go.Scatter(
                                x=df['date'], 
                                y=df['sma7'],
                                mode='lines',
                                name='SMA 7',
                                line=dict(color='#059669', width=1.5)
                            ))
                            
                        if indicators['sma_30'] is not None:
                            # Calculate SMA for each point
                            df['sma30'] = df['price'].rolling(window=30).mean()
                            fig.add_trace(go.Scatter(
                                x=df['date'], 
                                y=df['sma30'],
                                mode='lines',
                                name='SMA 30',
                                line=dict(color='#DC2626', width=1.5)
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f'{coin_options[selected_coin]} Price Chart ({period_options[selected_period]})',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            template='plotly_white',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display key indicators
                        signal_color = "green" if trend_result['signal'] == 'bullish' else "red" if trend_result['signal'] == 'bearish' else "orange"
                        
                        st.metric(
                            "Current Price", 
                            f"${indicators['current_price']:.2f}"
                        )
                        
                        st.markdown(f"<h3 style='color:{signal_color};'>Signal: {trend_result['signal'].upper()}</h3>", unsafe_allow_html=True)
                        st.markdown(f"**Confidence: {trend_result['confidence']:.1f}%**")
                        
                        # Display RSI gauge chart
                        if indicators['rsi'] is not None:
                            rsi = indicators['rsi']
                            
                            # Create the RSI gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=rsi,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "RSI"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#636EFA"},
                                    'steps': [
                                        {'range': [0, 30], 'color': '#10B981'},
                                        {'range': [30, 70], 'color': '#F59E0B'},
                                        {'range': [70, 100], 'color': '#EF4444'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 2},
                                        'thickness': 0.75,
                                        'value': rsi
                                    }
                                }
                            ))
                            
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed indicators section
                    st.subheader("Detailed Technical Indicators")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("SMA (7-day)", f"${indicators['sma_7']:.2f}" if indicators['sma_7'] else "N/A")
                        st.metric("SMA (30-day)", f"${indicators['sma_30']:.2f}" if indicators['sma_30'] else "N/A")
                        
                        if 'momentum' in indicators and '7d' in indicators['momentum']:
                            st.metric("7-day Momentum", f"{indicators['momentum']['7d']:.2f}%")
                        
                    with col2:
                        st.metric("RSI", f"{indicators['rsi']:.2f}" if indicators['rsi'] else "N/A")
                        st.metric("MACD", f"{indicators['macd']:.4f}" if indicators['macd'] else "N/A")
                        st.metric("MACD Signal", f"{indicators['macd_signal']:.4f}" if indicators['macd_signal'] else "N/A")
                        
                    with col3:
                        st.metric("Volatility", f"{indicators['volatility']:.2f}%" if indicators['volatility'] else "N/A")
                        
                        if 'bollinger' in indicators and indicators['bollinger']['upper']:
                            st.metric("Bollinger Upper", f"${indicators['bollinger']['upper']:.2f}")
                            st.metric("Bollinger Lower", f"${indicators['bollinger']['lower']:.2f}")
                    
                    # Signal details and explanation
                    st.subheader("Signal Details")
                    
                    details_col1, details_col2 = st.columns(2)
                    
                    with details_col1:
                        st.write("**Signal Component Breakdown:**")
                        st.write(f"- Bullish signals: {trend_result['bullish_count']}")
                        st.write(f"- Bearish signals: {trend_result['bearish_count']}")
                        st.write(f"- Neutral signals: {trend_result['neutral_count']}")
                        
                    with details_col2:
                        st.write("**Individual Indicator Signals:**")
                        for indicator_name, signal in trend_result['details'].items():
                            st.write(f"- {indicator_name.capitalize()}: {signal}")
                    
                    # Analysis interpretation
                    st.subheader("Interpretation")
                    if trend_result['signal'] == 'bullish':
                        st.markdown("""
                        **Bullish Outlook**: Technical indicators suggest positive momentum. Watch for:
                        - Continuation of upward price movement
                        - Increasing volume to confirm the trend
                        - Key resistance levels that may present challenges
                        """)
                    elif trend_result['signal'] == 'bearish':
                        st.markdown("""
                        **Bearish Outlook**: Technical indicators suggest downward pressure. Watch for:
                        - Continued price decline or consolidation
                        - Potential support levels where price might bounce
                        - Oversold conditions that could lead to a reversal
                        """)
                    else:
                        st.markdown("""
                        **Neutral Outlook**: Technical indicators are mixed. The market appears to be consolidating. Watch for:
                        - A breakout from the current range
                        - Increasing volume as a signal for the next move
                        - Developing patterns that might indicate future direction
                        """)
                    
                    # Disclaimer
                    st.info("⚠️ Disclaimer: Technical analysis is just one tool for decision-making. Past performance doesn't guarantee future results.")
                else:
                    st.error("Failed to calculate technical indicators.")
            else:
                st.error(f"Failed to fetch historical data for {coin_options[selected_coin]}.")

# Tab 4: Portfolio Management
with tab4:
    st.header("📈 Portfolio Management")
    
    if st.session_state.user_profile:
        # Portfolio suggestion section
        st.subheader("Portfolio Suggestions")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Investment amount input
            investment_amount = st.number_input(
                "Investment Amount (USD):",
                min_value=100.0,
                value=1000.0,
                step=100.0
            )
            
            # Risk profile selection
            risk_profiles = {
                'conservative': 'Conservative (Low Risk)',
                'balanced': 'Balanced (Medium Risk)',
                'aggressive': 'Aggressive (High Risk)',
                'eco_friendly': 'Eco-Friendly (Sustainable)',
                'high_cap': 'Blue Chip (Large Cap)'
            }
            
            # Default risk profile based on user preferences
            default_risk = 'balanced'
            if st.session_state.user_profile.sustainability_preference == 'high':
                default_risk = 'eco_friendly'
            elif st.session_state.user_profile.risk_tolerance == 'low':
                default_risk = 'conservative'
            elif st.session_state.user_profile.risk_tolerance == 'high':
                default_risk = 'aggressive'
            
            selected_risk_profile = st.selectbox(
                "Portfolio Type:",
                options=list(risk_profiles.keys()),
                format_func=lambda x: risk_profiles[x],
                index=list(risk_profiles.keys()).index(default_risk)
            )
            
            # Generate portfolio button
            if st.button("Generate Portfolio"):
                with st.spinner('Generating portfolio suggestion...'):
                    portfolio = safe_api_call(
                        st.session_state.portfolio_manager.get_portfolio_suggestion,
                        selected_risk_profile, 
                        investment_amount
                    )
                    
                    if portfolio:
                        st.session_state.current_portfolio = portfolio
                    else:
                        st.error("Failed to generate portfolio. Try again later.")
        
        # Display portfolio if available
        if 'current_portfolio' in st.session_state:
            with col2:
                portfolio = st.session_state.current_portfolio
                
                # Create portfolio allocation DataFrame
                portfolio_data = []
                for coin in portfolio['allocations']:
                    portfolio_data.append({
                        'Coin': coin['coin'].capitalize(),
                        'Allocation (%)': coin['allocation_percentage'],
                        'Amount ($)': coin['fiat_amount'],
                        'Quantity': coin['coin_amount'],
                        'Price ($)': coin['current_price']
                    })
                
                if portfolio_data:
                    df = pd.DataFrame(portfolio_data)
                    st.dataframe(df, hide_index=True, use_container_width=True)
                    
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    coins = [item['Coin'] for item in portfolio_data]
                    allocations = [item['Allocation (%)'] for item in portfolio_data]
                    
                    # Define a color map based on the portfolio type
                    if selected_risk_profile == 'eco_friendly':
                        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(allocations)))
                    elif selected_risk_profile == 'conservative':
                        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(allocations)))
                    elif selected_risk_profile == 'aggressive':
                        colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(allocations)))
                    else:
                        colors = plt.cm.tab10(np.arange(len(allocations)))
                    
                    ax.pie(allocations, labels=coins, autopct='%1.1f%%', startangle=90, colors=colors)
                    ax.axis('equal')
                    st.pyplot(fig)
            
            # Portfolio performance analysis
            st.subheader("Portfolio Performance Analysis")
            
            time_periods = {
                7: "1 Week",
                14: "2 Weeks",
                30: "1 Month",
                90: "3 Months"
            }
            
            selected_period = st.selectbox(
                "Analysis Period:",
                options=list(time_periods.keys()),
                format_func=lambda x: time_periods[x],
                index=2  # Default to 30 days
            )
            
            if st.button("Analyze Performance"):
                with st.spinner('Analyzing portfolio performance...'):
                    performance = st.session_state.portfolio_manager.get_portfolio_performance(
                        st.session_state.current_portfolio,
                        selected_period
                    )
                    
                    if performance:
                        # Display overall performance metrics
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        
                        with perf_col1:
                            st.metric("Initial Investment", f"${performance['initial_investment']:.2f}")
                            
                        with perf_col2:
                            st.metric("Current Value", f"${performance['current_value']:.2f}")
                            
                        with perf_col3:
                            delta_color = "normal" if performance['percent_change'] >= 0 else "inverse"
                            st.metric(
                                "Performance", 
                                f"{performance['percent_change']:.2f}%",
                                delta=f"{performance['percent_change']:.2f}%",
                                delta_color=delta_color
                            )
                        
                        # Display individual coin performance
                        if performance['coin_performance']:
                            st.subheader("Individual Coin Performance")
                            
                            coin_perf_data = []
                            for coin_perf in performance['coin_performance']:
                                coin_perf_data.append({
                                    'Coin': coin_perf['coin'].capitalize(),
                                    'Initial Value ($)': coin_perf['initial_value'],
                                    'Current Value ($)': coin_perf['current_value'],
                                    'Change (%)': coin_perf['percent_change']
                                })
                            
                            # Sort by performance (descending)
                            sorted_perf = sorted(coin_perf_data, key=lambda x: x['Change (%)'], reverse=True)
                            perf_df = pd.DataFrame(sorted_perf)
                            
                            # Display as styled dataframe with formatting
                            def style_negative(v):
                                return 'color: red' if v < 0 else 'color: green'
                            
                            styled_df = perf_df.style.format({
                                'Initial Value ($)': '${:.2f}',
                                'Current Value ($)': '${:.2f}',
                                'Change (%)': '{:.2f}%'
                            }).applymap(style_negative, subset=['Change (%)'])
                            
                            st.dataframe(styled_df, hide_index=True, use_container_width=True)
                            
                            # Create a horizontal bar chart of performance
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Reverse sort for the chart so best performer is on top
                            sorted_perf.reverse()
                            
                            coins = [item['Coin'] for item in sorted_perf]
                            changes = [item['Change (%)'] for item in sorted_perf]
                            
                            # Create color map based on positive/negative values
                            colors = ['#EF4444' if x < 0 else '#10B981' for x in changes]
                            
                            bars = ax.barh(coins, changes, color=colors)
                            ax.set_xlabel('Change (%)')
                            ax.set_title('Coin Performance Comparison')
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add values at end of bars
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                label_x_pos = width if width >= 0 else width - 1
                                ax.text(
                                    label_x_pos, 
                                    bar.get_y() + bar.get_height()/2, 
                                    f'{width:.2f}%', 
                                    va='center',
                                    fontweight='bold',
                                    color='black'
                                )
                            
                            st.pyplot(fig)
                    else:
                        st.error("Failed to calculate portfolio performance.")
    else:
        st.info("Please load your user profile in the Profile tab to get personalized portfolio suggestions.")

# Tab 5: Price Alerts
with tab5:
    st.header("🔔 Price Alerts")
    
    if st.session_state.user_profile:
        # Alert creation section
        st.subheader("Create Price Alert")
        
        alert_col1, alert_col2 = st.columns([1, 1])
        
        with alert_col1:
            # Coin selection for alert
            alert_coin = st.selectbox(
                "Select Cryptocurrency:",
                options=list(coin_options.keys()),
                format_func=lambda x: coin_options[x],
                key="alert_coin"
            )
            
            # Get current price for reference
            try:
                cg = CoinGeckoAPI()
                current_prices = cg.get_price(ids=[alert_coin], vs_currencies='usd')
                if alert_coin in current_prices:
                    current_price = current_prices[alert_coin]['usd']
                    st.info(f"Current price: ${current_price:.2f} USD")
            except:
                st.warning("Couldn't fetch current price. API may be rate limited.")
        
        with alert_col2:
            # Alert type selection
            alert_type = st.radio(
                "Alert Type:",
                options=["above", "below"],
                format_func=lambda x: f"When price goes {x} target",
                horizontal=True
            )
            
            # Target price input
            target_price = st.number_input(
                "Target Price (USD):",
                min_value=0.01,
                value=float(current_price * 1.1) if alert_type == "above" else float(current_price * 0.9),
                step=10.0,
                format="%.2f"
            )
        
        # Create alert button
        if st.button("Create Alert"):
            if st.session_state.user_profile:
                alert = PriceAlert(
                    alert_coin,
                    target_price,
                    alert_type,
                    st.session_state.user_profile.username
                )
                
                # Add to alert manager
                st.session_state.alert_manager.add_alert(alert)
                
                st.success(f"Alert created! You'll be notified when {coin_options[alert_coin]} price goes {alert_type} ${target_price:.2f} USD.")
                
                # Refresh alerts display
                st.session_state.alerts = st.session_state.alert_manager.get_alerts_by_username(st.session_state.user_profile.username)
            else:
                st.error("Please load your user profile first.")
        
        # Display user's alerts
        st.subheader("Your Price Alerts")
        
        # Get user's alerts
        user_alerts = st.session_state.alert_manager.get_alerts_by_username(
            st.session_state.user_profile.username
        )
        
        if user_alerts:
            alert_data = []
            for i, alert in enumerate(user_alerts):
                alert_data.append({
                    "ID": i,
                    "Coin": alert.coin_id.capitalize(),
                    "Target Price": f"${alert.target_price:.2f}",
                    "Condition": f"When price goes {alert.alert_type} target",
                    "Status": "Triggered" if alert.triggered else "Active",
                    "Created": datetime.datetime.fromisoformat(alert.created_at).strftime("%Y-%m-%d %H:%M")
                })
            
            alert_df = pd.DataFrame(alert_data)
            st.dataframe(alert_df, hide_index=True, use_container_width=True)
            
            # Alert deletion section
            st.subheader("Delete Alert")
            
            alert_ids = [alert["ID"] for alert in alert_data]
            if alert_ids:
                selected_alert = st.selectbox(
                    "Select Alert to Delete:",
                    options=alert_ids,
                    format_func=lambda x: f"Alert #{x}: {alert_data[x]['Coin']} {alert_data[x]['Condition']} {alert_data[x]['Target Price']}"
                )
                
                if st.button("Delete Selected Alert"):
                    if st.session_state.alert_manager.remove_alert(
                        st.session_state.alert_manager.alerts.index(user_alerts[selected_alert])
                    ):
                        st.success("Alert deleted successfully!")
                        
                        # Refresh alerts display
                        st.session_state.alerts = st.session_state.alert_manager.get_alerts_by_username(
                            st.session_state.user_profile.username
                        )
                    else:
                        st.error("Failed to delete alert.")
                        
                if st.button("Delete All Alerts"):
                    confirm = st.checkbox("I confirm I want to delete all my alerts")
                    
                    if confirm:
                        for alert in list(user_alerts):
                            st.session_state.alert_manager.alerts.remove(alert)
                        
                        st.session_state.alert_manager.save_alerts()
                        st.success("All alerts deleted.")
                        
                        # Refresh alerts display
                        st.session_state.alerts = st.session_state.alert_manager.get_alerts_by_username(
                            st.session_state.user_profile.username
                        )
        else:
            st.info("You don't have any price alerts set. Create one above!")
    else:
        st.info("Please load your user profile in the Profile tab to create and manage price alerts.")

# Footer with app info
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p><small>CryptoBuddy - Your AI Cryptocurrency Advisor | Data provided by CoinGecko API</small></p>
    <p><small>⚠️ Disclaimer: This application is for educational purposes only and does not constitute financial advice.</small></p>
</div>
""", unsafe_allow_html=True)

# Run alert monitoring in the background
def check_alerts_background():
    """Function to periodically check alerts in the background."""
    if 'alert_manager' in st.session_state:
        alert_manager = st.session_state.alert_manager
        triggered = alert_manager.check_alerts()
        
        # If any alerts were triggered, update the session state
        if triggered:
            alert_manager.save_alerts()
            # This will be displayed on the next rerun

# Run alert check when the app loads
check_alerts_background()
