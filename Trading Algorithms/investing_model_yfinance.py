import pandas as pd
import requests
import yfinance as yf


list_of_companies = []

# Now get all stock tickers in S & P 500
stockTable = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

first_table = stockTable[0]
second_table = stockTable[1]
data_frame = first_table

stockTickers = data_frame['Symbol'].values.tolist() # All Stock Tickets
stockPos = 0
financial_ratios = {}

for companyX in stockTickers:
    if stockPos < 500:
        stock_info = yf.Ticker(stockTickers[stockPos])
        financial_ratios[companyX] = {}

    try:
        financial_ratios[companyX]['P/E Ratio'] = stock_info.info['forwardPE']  # Price to Earnings Ratio
        if financial_ratios[companyX]['P/E Ratio'] is None:
            financial_ratios[companyX]['P/E Ratio'] = 0
        financial_ratios[companyX]['P/BV Ratio'] = stock_info.info['priceToBook']  # Price to Book Value Ratio
        if financial_ratios[companyX]['P/BV Ratio'] is None:
            financial_ratios[companyX]['P/BV Ratio'] = 0
        financial_ratios[companyX]['D/E Ratio'] = stock_info.info['debtToEquity']  # Debt to Equity Ratio
        if financial_ratios[companyX]['D/E Ratio'] is None:
            financial_ratios[companyX]['D/E Ratio'] = 0
        financial_ratios[companyX]['OPM'] = stock_info.info['operatingMargins']  # Operating Profit Margin
        if financial_ratios[companyX]['OPM'] is None:
            financial_ratios[companyX]['OPM'] = 0
        financial_ratios[companyX]['GPM'] = stock_info.info['grossMargins']  # Gross Profit Margin
        if financial_ratios[companyX]['GPM'] is None:
            financial_ratios[companyX]['GPM'] = 0
        financial_ratios[companyX]['EV/EBITDA'] = stock_info.info['enterpriseToEbitda'] # Enterprise Value by EBITDA
        if financial_ratios[companyX]['EV/EBITDA'] is None:
            financial_ratios[companyX]['EV/EBITDA'] = 0
        financial_ratios[companyX]['PE/G'] = stock_info.info['pegRatio']  # Price/Earnings to Growth Ratio
        if financial_ratios[companyX]['PE/G'] is None:
            financial_ratios[companyX]['PE/G'] = 0
        financial_ratios[companyX]['ROE'] = stock_info.info['returnOnEquity']  # Return on Equity Ratio
        if financial_ratios[companyX]['ROE'] is None:
            financial_ratios[companyX]['ROE'] = 0
        #financial_ratios[companyX]['Interest Coverage'] = stock_info.info['ebitda']  # Interest Coverage Ratio
        financial_ratios[companyX]['Current Ratio'] = stock_info.info['currentRatio']  # Current Ratio
        if financial_ratios[companyX]['Current Ratio'] is None:
            financial_ratios[companyX]['Current Ratio'] = 0
        #financial_ratios[companyX]['AT Ratio'] = ratios_of_company[0]['fixedAssetTurnover']  # Asset Turnover Ratio
        financial_ratios[companyX]['Dividend Yield'] = stock_info.info['dividendYield']  # Dividend Yield
        if financial_ratios[companyX]['Dividend Yield'] is None:
            financial_ratios[companyX]['Dividend Yield'] = 0
        financial_ratios[companyX]['Dividend Payout'] = stock_info.info['payoutRatio']  # Dividend Payout Ratio
        if financial_ratios[companyX]['Dividend Payout'] is None:
            financial_ratios[companyX]['Dividend Payout'] = 0
        financial_ratios[companyX]['PS'] = stock_info.info['priceToSalesTrailing12Months']  # Price to Sales Ratio
        if financial_ratios[companyX]['PS'] is None:
            financial_ratios[companyX]['PS'] = 0
        financial_ratios[companyX]['EPS'] = stock_info.info['trailingEps'] # Earnings Per Share
        if financial_ratios[companyX]['EPS'] is None:
            financial_ratios[companyX]['EPS'] = 0
        financial_ratios[companyX]['Rev Growth'] = stock_info.info['revenueGrowth']  # Revenue Growth
        if financial_ratios[companyX]['Rev Growth'] is None:
            financial_ratios[companyX]['Rev Growth'] = 0
        financial_ratios[companyX]['Earning Growth'] = stock_info.info['earningsGrowth']  # Revenue Growth
        if financial_ratios[companyX]['Earning Growth'] is None:
            financial_ratios[companyX]['Earning Growth'] = 0
        if stock_info.info['recommendationKey'] == "buy":
            financial_ratios[companyX]['Analyst Rec.'] = 1  # Revenue Growth
        if financial_ratios[companyX]['OPM'] is None:
            financial_ratios[companyX]['OPM'] = 0
        else:
            financial_ratios[companyX]['Analyst Rec.'] = -1
        financial_ratios[companyX]['Rec. Mean'] = stock_info.info['recommendationMean']  # Revenue Growth
        if financial_ratios[companyX]['Rec. Mean'] is None:
            financial_ratios[companyX]['Rec. Mean'] = 0
        financial_ratios[companyX]['QR'] = stock_info.info['quickRatio']  # Revenue Growth
        if financial_ratios[companyX]['QR'] is None:
            financial_ratios[companyX]['QR'] = 0
        #financial_ratios[companyX]['P/OCF'] = stock_info.info['currentPrice'] / ( stock_info.info['operatingCashflow'] / stock_info.info['fiftyDayAverage'] )  # Price to Operating Cash Flow Ratio
        financial_ratios[companyX]['NPM'] = stock_info.info['profitMargins']  # Net Profit Margin
        if financial_ratios[companyX]['NPM'] is None:
            financial_ratios[companyX]['NPM'] = 0
    except:
        pass

    stockPos += 1

# Ratio Importance Settings
pe_importance = 1.20
pbv_importance = 1.20
de_importance = -0.9
opm_importance = 1
gpm_importance = 1
ev_ebitda_importance = 0
peg_importance = 1.2
roe_importance = 1.10
#ic_importance = 1
cr_importance = 1
#at_importance = 1
dy_importance = 1.1
dp_importance = 1.05
ps_importance = -1.2
eps_importance = 1.075
rev_growth_importance = .75
earnings_growth_importance = .25
analyst_rec_importance = 2
analyst_mean_importance = 2
qr_importance = 1
p_ocf_importance = 0.12899
npm_importance = 1.15

# Create Panda Dataframe
stock_DF = pd.DataFrame.from_dict(financial_ratios, orient="index")

ratios_mean = []
for item in stock_DF.columns:
    ratios_mean.append(stock_DF[item].mean())

stock_DF = stock_DF / ratios_mean
stock_DF['Investment Score'] = stock_DF['P/E Ratio'] * pe_importance + stock_DF['P/BV Ratio'] * pbv_importance + stock_DF['D/E Ratio'] * de_importance + stock_DF['OPM'] * opm_importance + stock_DF['GPM'] * gpm_importance + stock_DF['EV/EBITDA'] * ev_ebitda_importance + stock_DF['PE/G'] * peg_importance + stock_DF['ROE'] + roe_importance + stock_DF['Current Ratio'] * cr_importance + stock_DF['Dividend Yield'] * dy_importance + stock_DF['Dividend Payout'] * dp_importance + stock_DF['PS'] * ps_importance + stock_DF['EPS'] * eps_importance + stock_DF['Rev Growth'] * rev_growth_importance + stock_DF['Analyst Rec.'] * analyst_rec_importance + stock_DF['Rec. Mean'] * analyst_mean_importance + stock_DF['QR'] * qr_importance + 0 * p_ocf_importance + stock_DF['NPM'] * npm_importance

print(stock_DF.sort_values(by=['Investment Score'], ascending=False))