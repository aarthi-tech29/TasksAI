# CSV Data Cleaner Tool

import pandas as pd

# Step 1: Load CSV file
df = pd.read_csv("raw_data.csv")

print("Original Data:")
print(df)


# Step 2: Remove duplicate rows
df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df)


# Step 3: Handle missing values
# Fill missing numeric values with 0
df["Marks"] = df["Marks"].fillna(0)

# Fill missing class with 'Unknown'
df["Class"] = df["Class"].fillna("Unknown")

print("\nAfter Handling Missing Values:")
print(df)


# Step 4: Remove extra spaces from text columns
df["Name"] = df["Name"].str.strip()

print("\nAfter Removing Extra Spaces:")
print(df)


# Step 5: Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned data saved as 'cleaned_data.csv'")
