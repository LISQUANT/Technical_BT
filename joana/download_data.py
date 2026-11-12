import yfinance
import os

prices = yfinance.download(tickers="^GSPC", start = "2000-01-01", end = "2026-10-31") #get the S&P 500 data from 2000 to 2026
prices.columns = prices.columns.get_level_values(0) #remove the ticker column


print(prices.tail()) #.tail returns a table with the last 5 rows of the data, prices.tail(10) shows the last 10
print(len(prices)) #len returns the number of rows in the data (6748)

os.makedirs("data", exist_ok=True) #create a folder called data
prices.to_csv("data/sp500_data.csv") #save the data to a CSV file