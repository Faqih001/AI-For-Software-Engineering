#!/usr/bin/env python3
"""
CryptoBuddy Launcher Script
This script provides a simple interface to launch either the command-line or
Streamlit web version of CryptoBuddy.
"""

import os
import sys
import time

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the CryptoBuddy banner."""
    banner = r"""
  _____                  _        ____            _     _
 / ____|                | |      |  _ \          | |   | |
| |     _ __ _   _ _ __ | |_ ___ | |_) |_   _  __| | __| |_   _
| |    | '__| | | | '_ \| __/ _ \|  _ <| | | |/ _` |/ _` | | | |
| |____| |  | |_| | |_) | || (_) | |_) | |_| | (_| | (_| | |_| |
 \_____|_|   \__, | .__/ \__\___/|____/ \__,_|\__,_|\__,_|\__, |
              __/ | |                                      __/ |
             |___/|_|                                     |___/
    """
    print("\033[36m" + banner + "\033[0m")  # Cyan color
    print("\033[1m" + "Your AI Cryptocurrency Advisor" + "\033[0m")
    print("\033[3m" + "Powered by CoinGecko API & Natural Language Processing" + "\033[0m")
    print("\n")

def main():
    """Main launcher function."""
    clear_screen()
    print_banner()
    
    print("Welcome to CryptoBuddy! How would you like to use the application?")
    print("\n1. Launch Streamlit Web Application (Recommended)")
    print("2. Launch Command-Line Interface")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        clear_screen()
        print_banner()
        print("Launching Streamlit Web Application...")
        print("You can access the application in your web browser momentarily.")
        print("\nPress Ctrl+C in the terminal to stop the application when you're done.")
        time.sleep(2)
        os.system("streamlit run app.py")
    
    elif choice == "2":
        clear_screen()
        print_banner()
        print("Launching Command-Line Interface...")
        time.sleep(1)
        from crypto_buddy import run_crypto_buddy
        run_crypto_buddy()
    
    elif choice == "3":
        print("\nThank you for using CryptoBuddy! Goodbye!")
        sys.exit(0)
    
    else:
        print("\nInvalid choice. Please try again.")
        time.sleep(1)
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting CryptoBuddy. Thank you for using our application!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check that you have all required dependencies installed.")
        print("You can install them with: pip install -r requirements.txt")
        sys.exit(1)
