# Outlines a Trading Template, to be combined with different APIs, Brokers, and Strategies

import requests
import time

class TradingBot:
    def __init__(self, api_url, access_token):
        self.api_url = api_url
        self.access_token = access_token

    def get_market_data(self, symbol):
        """ Fetch market data for a given symbol. """
        url = f"{self.api_url}/marketdata/{symbol}"
        response = requests.get(url, headers={"Authorization": f"Bearer {self.access_token}"})
        return response.json()

    def make_trade_decision(self, market_data):
        """
        Implement your trading strategy here.
        This is a simple example strategy: buy if the asset has dropped 5% from its average, sell if it's up 5%.
        """
        average_price = sum(market_data["prices"]) / len(market_data["prices"])
        current_price = market_data["prices"][-1]
        if current_price < 0.95 * average_price:
            return "BUY"
        elif current_price > 1.05 * average_price:
            return "SELL"
        else:
            return "HOLD"

    def execute_trade(self, symbol, action):
        """ Execute a trade on the platform. """
        url = f"{self.api_url}/trade"
        data = {"symbol": symbol, "action": action}
        response = requests.post(url, json=data, headers={"Authorization": f"Bearer {self.access_token}"})
        return response.json()

    def run(self, symbol):
        """ Main method to run the bot. """
        while True:
            market_data = self.get_market_data(symbol)
            decision = self.make_trade_decision(market_data)
            if decision != "HOLD":
                trade_response = self.execute_trade(symbol, decision)
                print(f"Executed {decision} for {symbol}: {trade_response}")
            time.sleep(60)  # Wait for 1 minute before next check

# Example usage
api_url = "https://api.tradingplatform.com"
access_token = "your_access_token_here"
symbol = "AAPL"

bot = TradingBot(api_url, access_token)
bot.run(symbol)
