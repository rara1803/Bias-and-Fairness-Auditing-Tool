import pandas as pd

df = pd.read_csv("data/german.data", header=None, delim_whitespace=True)
print(df.shape)
print(df.head())
