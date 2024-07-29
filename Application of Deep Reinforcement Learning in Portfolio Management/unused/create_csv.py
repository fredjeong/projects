
import pandas as pd
from binance.client import Client
import datetime as dt

# Initialize the Binance client
api_key = 'YOUR_BINANCE_API_KEY'
api_secret = 'YOUR_BINANCE_API_SECRET'
client = Client(api_key, api_secret)

# Define the cryptocurrencies and the date range
cryptos = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT']
start_date = '2020-04-01'
end_date = '2024-06-22'

# Convert date range to milliseconds
start_ts = int(dt.datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
end_ts = int(dt.datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

# Function to fetch historical data
def fetch_klines(symbol, interval, start, end):
    return client.get_historical_klines(symbol, interval, start, end)

# Create an empty DataFrame
df = pd.DataFrame()

# Fetch data for each cryptocurrency
for crypto in cryptos:
    print(f"Fetching data for {crypto}...")
    klines = fetch_klines(crypto, Client.KLINE_INTERVAL_1HOUR, start_ts, end_ts)
    
    # Extract relevant data
    data = [[
        kline[0],  # Timestamp
        crypto,   # Cryptocurrency
        float(kline[1]),  # Open price
        float(kline[2]),  # Highest price
        float(kline[3]),   # Lowest price
        float(kline[4])  # Close price
    ] for kline in klines]
    
    # Create a temporary DataFrame
    temp_df = pd.DataFrame(data, columns=['Timestamp', 'Cryptocurrency', 'Open', 'High', 'Low', 'Close'])
    
    # Append to the main DataFrame
    df = pd.concat([df, temp_df])

# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

# Save to CSV
df.to_csv('cryptocurrency_data.csv', index=False)

print("Data fetching and saving to CSV complete.")

