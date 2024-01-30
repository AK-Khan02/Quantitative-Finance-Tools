import pandas as pd
import yfinance as yf

# Retrieve S&P 500 stock tickers from Wikipedia
sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_tables = pd.read_html(sp500_url)
sp500_tickers = sp500_tables[0]['Symbol'].values.tolist()

# Initialize dictionary to store financial ratios
financial_ratios = {}

# Process each company in the S&P 500 list
for ticker in sp500_tickers:
    stock_info = yf.Ticker(ticker)
    ratios = {}

    # Helper function to safely extract info, defaults to 0 if not found
    def get_info(key):
        return stock_info.info.get(key, 0)

    # Collect financial ratios and other metrics
    ratios['P/E Ratio'] = get_info('forwardPE')
    ratios['P/BV Ratio'] = get_info('priceToBook')
    ratios['D/E Ratio'] = get_info('debtToEquity')
    ratios['OPM'] = get_info('operatingMargins')  # Operating Profit Margin
    ratios['GPM'] = get_info('grossMargins')  # Gross Profit Margin
    ratios['EV/EBITDA'] = get_info('enterpriseToEbitda')
    ratios['PE/G'] = get_info('pegRatio')  # P/E to Growth Ratio
    ratios['ROE'] = get_info('returnOnEquity')  # Return on Equity
    ratios['Current Ratio'] = get_info('currentRatio')
    ratios['Dividend Yield'] = get_info('dividendYield')
    ratios['Dividend Payout'] = get_info('payoutRatio')
    ratios['PS'] = get_info('priceToSalesTrailing12Months')  # Price to Sales Ratio
    ratios['EPS'] = get_info('trailingEps')  # Earnings Per Share
    ratios['Rev Growth'] = get_info('revenueGrowth')  # Revenue Growth
    ratios['Earning Growth'] = get_info('earningsGrowth')
    ratios['Analyst Rec.'] = 1 if get_info('recommendationKey') == "buy" else -1
    ratios['Rec. Mean'] = get_info('recommendationMean')
    ratios['QR'] = get_info('quickRatio')  # Quick Ratio
    ratios['NPM'] = get_info('profitMargins')  # Net Profit Margin

    # Update the main dictionary
    financial_ratios[ticker] = ratios

# Create a DataFrame from the collected financial ratios
stock_df = pd.DataFrame.from_dict(financial_ratios, orient="index")

# Normalize ratios by their mean to standardize for comparison
ratios_mean = stock_df.mean()
normalized_stock_df = stock_df / ratios_mean

# Assign weights to each financial ratio based on its importance
weights = {
    'P/E Ratio': 1.20, 'P/BV Ratio': 1.20, 'D/E Ratio': -0.9, 'OPM': 1, 'GPM': 1,
    'EV/EBITDA': 0, 'PE/G': 1.2, 'ROE': 1.10, 'Current Ratio': 1, 'Dividend Yield': 1.1,
    'Dividend Payout': 1.05, 'PS': -1.2, 'EPS': 1.075, 'Rev Growth': 0.75,
    'Earning Growth': 0.25, 'Analyst Rec.': 2, 'Rec. Mean': 2, 'QR': 1, 'NPM': 1.15
}

# Calculate the 'Investment Score' for each stock based on weighted ratios
normalized_stock_df['Investment Score'] = normalized_stock_df.dot(pd.Series(weights))

# Sort stocks by their 'Investment Score' in descending order
sorted_stocks = normalized_stock_df.sort_values(by='Investment Score', ascending=False)

print(sorted_stocks[['Investment Score']])
