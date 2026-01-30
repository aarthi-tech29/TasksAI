import pandas as pd

# Create product price data
data = {
    "Product": ["Mobile", "Laptop", "Headphones", "Tablet", "Smartwatch"],
    "Amazon": [15000, 55000, 2000, 22000, 8000],
    "Flipkart": [14500, 56000, 2100, 21500, 8200],
    "Reliance": [14800, 54000, 2050, 22500, 7900]
}

# Convert to DataFrame
df = pd.DataFrame(data)

print("\n=== Product Price Table ===")
print(df)

# Dataset info
print("\n=== Dataset Info ===")
df.info()

# Cheapest price
df["Cheapest_Price"] = df[["Amazon", "Flipkart", "Reliance"]].min(axis=1)

# Cheapest store
df["Cheapest_Store"] = df[["Amazon", "Flipkart", "Reliance"]].idxmin(axis=1)

# Average price
df["Average_Price"] = df[["Amazon", "Flipkart", "Reliance"]].mean(axis=1)

print("\n=== Price Comparison Result ===")
print(df)

# Store-wise average
print("\n=== Store-wise Average Price ===")
print(df[["Amazon", "Flipkart", "Reliance"]].mean())

# Expensive products
print("\n=== Products Above ₹20,000 ===")
print(df[df["Cheapest_Price"] > 20000])

# Save to CSV
df.to_csv("product_price_comparison.csv", index=False)

print("\nComparison saved as product_price_comparison.csv")
