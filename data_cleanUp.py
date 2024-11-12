import pandas as pd
import datetime

# Read the CSV file into a DataFrame
df = pd.read_csv('SortedData.csv', low_memory=False)

# Set 'Date' column as index
df.set_index('Date', drop=True, inplace=True)

# Drop unnecessary columns
# df.drop(['52 Weeks High', '52 Weeks Low', 'Conf.', 'Diff', 'Diff %',  'Prev. Close', 'Range', 'Range %', 'S.No', 'Trans.', 'Turnover', 'VWAP', 'VWAP %','120 Days','180 Days'], axis=1, inplace=True)
df.drop(['52 Weeks High', '52 Weeks Low', 'Conf.', 'Diff', 'Diff %',  'Prev. Close', 'Range', 'Range %', 'S.No', 'Trans.', 'Turnover', 'VWAP', 'VWAP %'], axis=1, inplace=True)


# Sort DataFrame by 'Symbol' and 'Date'
df.sort_values(by=['Symbol','Date'], ascending=True, inplace=True)


# Filter DataFrame by 'Symbol' == 'PRVU'
commercial_bankName = 'SCB'
df = df[df['Symbol'] == commercial_bankName]

# Reset index to bring 'Date' column back as a regular column
df.reset_index(inplace=True)

# Convert 'Date' column from string to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as index
df.set_index('Date', inplace=True)

# Replace commas with empty string and convert to float for selected columns
columns = ['Close', 'High', 'Low', 'Open', 'Vol']
df[columns] = df[columns].replace(',', '', regex=True).astype(float)
# Save cleaned DataFrame to CSV
df.to_csv(f'{commercial_bankName}.csv')
print('.............completed.................')
