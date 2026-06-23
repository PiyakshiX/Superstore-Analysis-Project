import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("superstore.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Month'] = df['Order Date'].dt.month

monthly_sales = df.groupby('Month')['Sales'].sum()

print("Monthly Sales:")
print(monthly_sales)

monthly_sales.plot(kind='line', marker='o')

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)

plt.show()

top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False)

print("\nTop Products:")
print(top_products)

top_products.plot(kind='bar')

plt.title("Top Selling Products")
plt.xlabel("Product")
plt.ylabel("Sales")

plt.show()

profit = df.groupby('Category')['Profit'].sum()

print("\nProfit by Category:")
print(profit)

profit.plot(kind='bar')

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.show()

print("\n✅ Sales Analysis Completed!")