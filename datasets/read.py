#use this file just to read the column and row names for datasets. helps to be organized and know what to put in code
import pandas as pd
df = pd.read_csv('datasets/ikea_new_w_prices.csv') #change to wtv dataset
print(df.columns.tolist())
print(df.head(5))