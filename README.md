# 🚀 CryptoBuddy - The AI-Powered Crypto Recommendation Chatbot

## 🌟 Overview

CryptoBuddy is an intelligent cryptocurrency advisory chatbot that leverages Natural Language Processing (NLP) to interpret user queries and provide personalized cryptocurrency recommendations. By analyzing real-time market data from the CoinGecko API and applying rule-based recommendation algorithms, CryptoBuddy offers insights on trending cryptocurrencies, sustainable options, and long-term investment opportunities.

## ✨ Key Features

- **🧠 Natural Language Processing**: Understands natural language queries through NLTK integration
- **📊 Real-time Data**: Fetches current cryptocurrency prices, market caps, and trends from CoinGecko API
- **🌱 Sustainability Focus**: Includes environmental impact considerations for crypto recommendations
- **💬 Interactive Interface**: Engaging, varied responses with randomized greetings and disclaimers
- **📝 Multiple Query Types**: Handles different intents like trending, sustainable, and long-term investment queries

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
   pip install nltk pycoingecko
   ```

4. **Run the application**

   ```bash
   python crypto_buddy.py
   ```

## 💡 Usage Examples

Ask CryptoBuddy questions like:

- "What cryptocurrency is trending right now?"
- "Which crypto is the most eco-friendly?"
- "What's a good long-term investment in crypto?"
- "What do you recommend for a balanced investment?"

## 🧩 How It Works

1. **Query Interpretation**:
   - The chatbot tokenizes and analyzes your question
   - It identifies keywords related to trends, sustainability, or long-term investment
   - It expands keyword recognition through WordNet synonyms

2. **Data Retrieval**:
   - Fetches real-time data from CoinGecko API
   - Augments with sustainability ratings (currently simulated data)

3. **Recommendation Algorithm**:
   - Trending intent: Prioritizes coins with rising trends and high market caps
   - Sustainability intent: Ranks coins by environmental impact scores
   - Long-term intent: Balances growth trends and sustainability
   - General intent: Creates balanced recommendations across all factors

4. **Response Generation**:
   - Selects a random greeting and disclaimer
   - Chooses an appropriate response template based on intent
   - Populates the template with relevant cryptocurrency data
   - Returns a friendly, informative response

## 📋 Current Cryptocurrency Coverage

CryptoBuddy currently provides information on:

- **Bitcoin (BTC)**: The original cryptocurrency
- **Ethereum (ETH)**: The leading smart contract platform
- **Cardano (ADA)**: A proof-of-stake blockchain platform

## ⚠️ Limitations

- **Limited Cryptocurrency Coverage**: Currently only analyzes Bitcoin, Ethereum, and Cardano
- **Mocked Sustainability Data**: Environmental impact scores are simulated, not real-world data
- **API Rate Limits**: CoinGecko's free API has usage restrictions (30 calls/minute)
- **Simplified Market Analysis**: Does not incorporate complex technical indicators

## 🔮 Future Enhancements

- **Expanded Cryptocurrency Coverage**: Add support for more cryptocurrencies
- **Real Sustainability Data**: Integrate actual environmental impact metrics
- **Technical Analysis**: Incorporate more sophisticated market indicators
- **User Profiles**: Save user preferences for personalized recommendations
- **Portfolio Suggestions**: Recommend balanced cryptocurrency portfolios
- **Price Alerts**: Notify users of significant price movements
- **Historical Performance**: Add historical performance data and trends

## 🧪 Technical Details

### File Structure

- **crypto_buddy.py**: Main application file containing all functionality

### Dependencies

- **pycoingecko**: Python wrapper for CoinGecko API
- **nltk**: Natural Language Toolkit for query interpretation
- **random**: Used for response randomization
- **sys**: System-specific parameters and functions
- **time**: Time-related functions for API rate limiting

### Error Handling

- Robust error handling for API connectivity issues
- Graceful fallback for NLTK data availability problems
- SSL certificate verification workarounds

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [CoinGecko API](https://www.coingecko.com/api/documentations/v3) for cryptocurrency market data
- [NLTK](https://www.nltk.org/) for natural language processing capabilities

---

⚠️ **Disclaimer**: CryptoBuddy is an educational tool and does not provide financial advice. Cryptocurrency investments involve significant risks. Always conduct your own research before making investment decisions.
