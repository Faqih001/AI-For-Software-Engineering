# 🚀 CryptoBuddy - AI-Powered Crypto Recommendation Web App

## 🌟 Overview

CryptoBuddy is an intelligent cryptocurrency advisory application that leverages Natural Language Processing (NLP) to interpret user queries and provide personalized cryptocurrency recommendations. Originally developed as a command-line chatbot, it's now a web application built with Streamlit, offering an intuitive graphical interface while maintaining all the powerful features of the original version. By analyzing real-time market data from the CoinGecko API and applying rule-based recommendation algorithms, CryptoBuddy offers insights on trending cryptocurrencies, sustainable options, and long-term investment opportunities.

## ✨ Key Features

- **🧠 Natural Language Processing**: Understands natural language queries through NLTK integration
- **📊 Real-time Data**: Fetches current cryptocurrency prices, market caps, and trends from CoinGecko API
- **🌱 Sustainability Focus**: Includes environmental impact considerations for crypto recommendations
- **💬 Interactive Chat Interface**: Engaging, varied responses with randomized greetings and disclaimers
- **📝 Multiple Query Types**: Handles different intents like trending, sustainable, and long-term investment queries
- **🔍 Advanced Technical Analysis**: Interactive visualizations for sophisticated market indicators including RSI, MACD and Bollinger Bands
- **🔔 Price Alerts**: Set and manage custom price alerts for your favorite cryptocurrencies through an intuitive interface
- **👤 Personalized Profiles**: Save your preferences and get tailored cryptocurrency recommendations
- **💼 Portfolio Management**: Interactive portfolio suggestions with visual breakdowns based on various risk profiles
- **📊 Data Visualizations**: Beautiful charts and graphs for analyzing cryptocurrency performance
- **📱 Responsive Web Interface**: User-friendly Streamlit interface accessible from any browser

## 🛠️ Technical Components

### API Integration

- **CoinGecko API**: Retrieves real-time cryptocurrency market data
- **Data points**: Prices, 24-hour changes, market capitalization

### NLP Capabilities

- **NLTK Integration**: Natural language understanding for user queries
- **Word Tokenization**: Breaks down queries into processable tokens
- **Part-of-Speech Tagging**: Analyzes grammatical structure
- **WordNet Integration**: Identifies synonyms for enhanced intent recognition

### Recommendation Logic

- **Trend Analysis**: Identifies cryptocurrencies with positive price momentum
- **Sustainability Rating**: Evaluates environmental impact (currently mocked data)
- **Market Cap Evaluation**: Assesses market stability and prominence
- **Combined Metrics**: Balances multiple factors for well-rounded recommendations

### Response Generation

- **Dynamic Responses**: 10+ different response templates for each query type
- **Randomized Greetings**: Various friendly introductions
- **Ethical Disclaimers**: Important investment warnings with variety
- **Rich Language**: Engaging vocabulary with crypto terminology

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- pip (Python package manager)

### Installation

1. **Clone the repository**

   ```bash

git clone <https://github.com/Faqih001/AI-For-Software-Engineering.git>
    cd AI-For-Software-Engineering

   ```bash

2. **Set up a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the launcher script (recommended)**

   ```bash
   python run_crypto_buddy.py
   ```

5. **Or run either version directly**

   For the Streamlit web application:

   ```bash
   streamlit run app.py
   ```

   For the command-line version:

   ```bash
   python crypto_buddy.py
   ```

## 💡 Usage Examples

### Natural Language Queries

Ask CryptoBuddy questions like:

- "What cryptocurrency is trending right now?"
- "Which crypto is the most eco-friendly?"
- "What's a good long-term investment in crypto?"
- "What do you recommend for a balanced investment?"

### Command Interface

Use slash commands for specialized functions:

- `/help` - Display all available commands
- `/profile` - Check and update your user preferences
- `/alerts` - Manage your price alert notifications
- `/portfolio` - Get personalized investment suggestions
- `/analyze bitcoin` - View technical analysis for Bitcoin

### Advanced Features

Try these enhanced capabilities:

- "Set an alert for Ethereum when it drops below $2000"
- "Show me a technical analysis of Cardano"
- "Suggest an aggressive portfolio for $5000"
- "What are the key indicators for Solana?"

### Streamlit Web Interface

The new web application provides all the functionality of the command-line version with an intuitive GUI:

1. **Chat Tab**: Ask questions and get cryptocurrency recommendations in a chat interface
2. **Profile Tab**: Create and manage your user profile with easy-to-use form inputs
3. **Technical Analysis Tab**: Generate interactive visualizations with price charts and indicators
4. **Portfolio Tab**: Create and analyze portfolio suggestions with visual breakdowns
5. **Price Alerts Tab**: Set up and manage price alerts through a user-friendly interface

## 🧩 How It Works

1. **Query Interpretation**:
   - The chatbot tokenizes and analyzes your question using NLTK
   - It identifies keywords related to trends, sustainability, or long-term investment
   - It expands keyword recognition through WordNet synonyms
   - It recognizes command patterns with regex for specialized functions

2. **Data Retrieval**:
   - Fetches real-time data from CoinGecko API for 11 cryptocurrencies
   - Retrieves historical price data for technical analysis
   - Loads user profiles and preferences for personalized responses
   - Monitors price movements for user-defined alerts

3. **Technical Analysis**:
   - Calculates RSI (Relative Strength Index) to identify overbought/oversold conditions
   - Generates MACD (Moving Average Convergence Divergence) signals
   - Creates Bollinger Bands to identify price volatility and potential reversals
   - Computes momentum indicators across multiple timeframes
   - Produces confidence-scored trend signals integrating multiple indicators

4. **Portfolio Management**:
   - Matches user risk tolerance to appropriate portfolio templates
   - Allocates investment amounts across multiple cryptocurrencies
   - Calculates expected coin quantities based on current prices
   - Analyzes historical performance of portfolios
   - Identifies top and bottom performers in a portfolio

5. **User Profile System**:
   - Stores and retrieves personalized user preferences
   - Tracks favorite cryptocurrencies, risk tolerance, and investment horizon
   - Manages price alerts linked to user profiles
   - Creates backups of profile data for data safety
   - Provides version tracking for backward compatibility

6. **Response Generation**:
   - Selects a random greeting and disclaimer
   - Chooses an appropriate response template based on intent
   - Incorporates technical analysis data when relevant
   - Personalizes responses based on user profile
   - Returns a friendly, informative response or formatted data visualization

## 📋 Current Cryptocurrency Coverage

CryptoBuddy provides information on a wide range of cryptocurrencies:

- **Bitcoin (BTC)**: The original cryptocurrency
- **Ethereum (ETH)**: The leading smart contract platform
- **Cardano (ADA)**: A proof-of-stake blockchain platform
- **Solana (SOL)**: High-performance blockchain
- **Polkadot (DOT)**: Multi-chain interoperability protocol
- **Ripple (XRP)**: Digital payment protocol
- **Dogecoin (DOGE)**: Popular meme cryptocurrency
- **Avalanche (AVAX)**: Platform for decentralized applications
- **Chainlink (LINK)**: Decentralized oracle network
- **Polygon (MATIC)**: Ethereum scaling platform
- **NEAR Protocol (NEAR)**: Layer 1 blockchain

## ✅ Web App Conversion

CryptoBuddy has been converted from a command-line application to a full-featured web application using Streamlit, making it more accessible and user-friendly while preserving all the advanced functionality of the original version. The web interface provides:

1. **Intuitive Navigation**: Tabbed interface for easy access to different features
2. **Interactive Data Visualization**: Dynamic charts and graphs for technical analysis
3. **Responsive Design**: Accessible from desktop and mobile browsers
4. **Live Updates**: Real-time data fetching and alert monitoring
5. **Enhanced User Experience**: Form-based inputs and dropdown menus for easier interaction

### 📱 Application Screenshots

#### Chat Interface

The chat interface features a conversation-style layout with user messages and CryptoBuddy responses, making it easy to ask questions about cryptocurrencies in natural language.

#### Technical Analysis Dashboard

The Technical Analysis tab provides interactive price charts with technical indicators like RSI, SMA, Bollinger Bands, and more, along with automated signal interpretations.

#### Portfolio Visualization

The Portfolio tab displays allocation recommendations with pie charts and performance metrics, allowing users to compare different investment strategies.

#### User Profile Management

The Profile tab offers an intuitive form to set risk tolerance, sustainability preferences, and other user settings that affect recommendations.

#### Price Alert Configuration

The Price Alerts tab provides a simple interface to set, monitor, and manage cryptocurrency price alerts with real-time notifications.

## ✅ Advanced Features

### 📊 Technical Analysis

- **Comprehensive Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **Signal Generation**: Trend signals with confidence scores
- **Momentum Analysis**: 1-day, 7-day, 14-day, and 30-day price momentum tracking
- **Market Position**: Identification of overbought/oversold conditions

### 🔔 Price Alerts

- **Custom Alert Thresholds**: Set alerts for when price rises above or falls below specific values
- **Alert Management**: Add, remove, and monitor multiple price alerts
- **Persistent Alerts**: Alerts are saved between sessions
- **Real-time Notifications**: Get notified when your alert conditions are met

### 💼 Portfolio Management

- **Risk-based Portfolio Templates**: Conservative, balanced, aggressive, eco-friendly, and high-cap templates
- **Personalized Suggestions**: Recommendations based on user profile preferences
- **Investment Allocation**: Detailed breakdown of investment allocations with exact coin amounts
- **Historical Performance**: Track portfolio performance over time
- **Performance Metrics**: See top and bottom performers in your portfolio

### 👤 User Profiles

- **Preference Storage**: Save and load user preferences between sessions
- **Customizable Risk Parameters**: Set risk tolerance, sustainability preference, and investment horizon
- **Favorite Coins**: Track your favorite cryptocurrencies
- **Profile Versioning**: Automatic profile upgrades with backward compatibility
- **Profile Backups**: Automatic backup creation for data safety

### 💬 Enhanced Command Interface

- **/help**: Display available commands
- **/profile**: View and update user profile settings
- **/alerts**: Manage your price alerts
- **/portfolio**: Get personalized portfolio suggestions
- **/analyze <coin>**: Generate technical analysis for specific cryptocurrency
- **/exit**: Save your profile and exit the chatbot

## 🧪 Technical Details

### File Structure

- **crypto_buddy.py**: Main application file containing all functionality
- **profiles/**: Directory storing user profile data in JSON format
- **alerts/**: Directory storing price alert configurations in JSON format

### Class Structure

- **PriceAlert**: Manages individual cryptocurrency price alerts
- **AlertManager**: Orchestrates multiple price alerts and monitoring
- **UserProfile**: Handles user preferences and profile management
- **PortfolioManager**: Creates and evaluates cryptocurrency portfolio suggestions

### Dependencies

- **pycoingecko**: Python wrapper for CoinGecko API
- **nltk**: Natural Language Toolkit for query interpretation
- **json**: Used for profile and alert data serialization
- **datetime**: Timestamp handling for alerts and historical data
- **threading**: Background monitoring of price alerts
- **re**: Regular expression pattern matching for command processing
- **random**: Used for response randomization
- **sys**: System-specific parameters and functions
- **time**: Time-related functions for API rate limiting
- **os**: File and directory management

### Error Handling

- Robust error handling for API connectivity issues
- Graceful fallback for NLTK data availability problems
- SSL certificate verification workarounds
- Profile data backup and recovery mechanism
- Exception handling for price monitoring and data retrieval

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [CoinGecko API](https://www.coingecko.com/api/documentations/v3) for cryptocurrency market data
- [NLTK](https://www.nltk.org/) for natural language processing capabilities

---

⚠️ **Disclaimer**: CryptoBuddy is an educational tool and does not provide financial advice. Cryptocurrency investments involve significant risks. Always conduct your own research before making investment decisions.
