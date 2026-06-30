import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
print("Current Folder:", os.getcwd())
print("Files:",os.listdir(os.getcwd()))

df=pd.read_csv("Sample - Superstore.csv",encoding='latin1')
print("\n Dataset loaded successfully")

print("Shape:", df.shape)
df.head()
print(df.columns.tolist())
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
print(df.duplicated().sum())

print("Categories:", df['Category'].unique())
print("Regions:", df['Region'].unique())
print("Segements:", df['Segment'].unique())

print(df.columns)

df['Order Date']=pd.to_datetime(df['Order Date'])
df['Ship Date']=pd.to_datetime(df['Ship Date'])
print("Start Date:", df['Order Date'].min())
print("End Date:", df['Order Date'].max())

df['Order Year']=df['Order Date'].dt.year
print(df['Order Year'].unique())

df['Order Month'] = df['Order Date'].dt.month_name()
print(df['Order Month'].unique())

df['Order quarter']=df['Order Date'].dt.quarter
print(df['Order quarter'].unique())

df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
print(df['Shipping Days'].head())
print(df['Shipping Days'].describe())

print(df.columns.tolist())
print(df[['Order Date',
          'Order Year',
          'Order Month',
          'Order quarter',
          'Shipping Days']].head())

df.to_csv("Superstore_Cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")

total_sales=df['Sales'].sum()
print("Total Sales: $", round(total_sales, 2))

total_profit=df['Profit'].sum()
print("Total Profit: $", round(total_profit, 2))

sales_region=df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
print(sales_region)
sales_region.plot(kind='bar')
plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.show()

sales_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print(sales_category)
sales_category.plot(kind='bar')
plt.title('Sales by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()

top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
print(top_products)
top_products.plot(kind='bar')
plt.title('Top 10 Products by Sales')
plt.xlabel('Product Name')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()

sales_year = df.groupby('Order Year')['Sales'].sum()
print(sales_year)
sales_year.plot(marker='o')
plt.title('Year-wise Sales Trend')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()


profit_subcategory = df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False)
print(profit_subcategory)
profit_subcategory.plot(kind='bar', figsize=(10,5))

plt.title("Profit by Sub-Category")
plt.xlabel("Sub-Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.show()

loss_subcategory = profit_subcategory[profit_subcategory < 0]
print(loss_subcategory)

loss_products = df.groupby('Product Name')['Profit'].sum().sort_values().head(10)
print(loss_products)
loss_products.plot(kind='bar')

plt.title("Top 10 Loss-Making Products")
plt.xlabel("Product")
plt.ylabel("Profit")
plt.xticks(rotation=90)
plt.show()


top_customers_sales = (
    df.groupby('Customer Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customers_sales)
top_customers_sales.plot(kind='bar')
plt.title("Top 10 Customers by Sales")
plt.xlabel("Customer")
plt.ylabel("Sales")
plt.xticks(rotation=90)
plt.show()

top_customers_profit = (
    df.groupby('Customer Name')['Profit']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customers_profit)
top_customers_profit.plot(kind='bar')
plt.title("Top 10 Customers by Profit")
plt.xlabel("Customer")
plt.ylabel("Profit")
plt.xticks(rotation=90)
plt.show()

segment_sales = (
    df.groupby('Segment')['Sales']
    .sum()
    .sort_values(ascending=False)
)
print(segment_sales)
segment_sales.plot(kind='bar')
plt.title("Sales by Segment")
plt.xlabel("Segment")
plt.ylabel("Sales")
plt.show()

segment_profit = (
    df.groupby('Segment')['Profit']
    .sum()
    .sort_values(ascending=False)
)
print(segment_profit)
segment_profit.plot(kind='bar')
plt.title("Profit by Segment")
plt.xlabel("Segment")
plt.ylabel("Profit")
plt.show()

print(df['Discount'].value_counts().sort_index())

discount_profit = (
    df.groupby('Discount')['Profit']
    .mean()
    .sort_index()
)
print(discount_profit)
discount_profit.plot(marker='o')
plt.title("Average Profit by Discount")
plt.xlabel("Discount")
plt.ylabel("Average Profit")
plt.show()

discount_profit.plot(kind='bar')
plt.title("Average Profit by Discount Level")
plt.xlabel("Discount")
plt.ylabel("Average Profit")
plt.show()

high_discount = df[df['Discount'] >= 0.5]
print("Number of High Discount Orders:", len(high_discount))
print("Average Profit of High Discount Orders:")
print(high_discount['Profit'].mean())

correlation = df['Discount'].corr(df['Profit'])
print("Correlation between Discount and Profit:", correlation)

plt.hist(df['Sales'], bins=30)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

plt.hist(df['Profit'], bins=30)
plt.title("Distribution of Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.show()

plt.hist(df['Discount'], bins=10)
plt.title("Distribution of Discount")
plt.xlabel("Discount")
plt.ylabel("Frequency")
plt.show()

plt.hist(df['Quantity'], bins=15)
plt.title("Distribution of Quantity")
plt.xlabel("Quantity")
plt.ylabel("Frequency")
plt.show()

plt.boxplot(df['Sales'])
plt.title("Boxplot of Sales")
plt.show()

plt.boxplot(df['Profit'])
plt.title("Boxplot of Profit")
plt.show()