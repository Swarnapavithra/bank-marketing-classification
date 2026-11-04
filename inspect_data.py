import pandas as pd

# Load dataset
df = pd.read_csv("data/bank-full.csv", sep=";")

# Display basic information
print("\nFirst 5 records:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nTarget distribution:")
print(df["y"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nUnique values per column:")
print(df.nunique())