# ==========================================
# Sales Distribution Analysis (L1 & L2)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# ------------------------------------------
# Basic Statistics
# ------------------------------------------

print("\n========== SALES STATISTICS ==========\n")
print(df["Sales"].describe())

# ------------------------------------------
# Separation Point
# ------------------------------------------

threshold = 610

print("\n========== SALES LEVEL ==========\n")
print(f"L1 : Sales <= {threshold}")
print(f"L2 : Sales > {threshold}")

# ------------------------------------------
# Create Sales Level Column
# ------------------------------------------

df["Sales Level"] = df["Sales"].apply(
    lambda x: "L1" if x <= threshold else "L2"
)

# ------------------------------------------
# Show Sample Records
# ------------------------------------------

print("\n========== FIRST 10 RECORDS ==========\n")
print(df[["Sales", "Sales Level"]].head(10))

# ------------------------------------------
# Count Orders
# ------------------------------------------

print("\n========== SALES LEVEL COUNT ==========\n")

counts = df["Sales Level"].value_counts()

print(counts)

# ------------------------------------------
# Percentage
# ------------------------------------------

percentage = round(counts / len(df) * 100, 2)

print("\n========== PERCENTAGE ==========\n")

print(percentage)

# ==========================================
# Graph 1 : Histogram
# ==========================================

plt.figure(figsize=(12,6))

sns.histplot(
    df["Sales"],
    bins=100,
    kde=True,
    color="skyblue",
    edgecolor="black"
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    linewidth=3,
    label=f"L1 / L2 Boundary = {threshold}"
)

plt.xlim(0,1200)

plt.title(
    "Distribution of Sales (L1 and L2)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Sales")

plt.ylabel("Frequency")

plt.legend()

plt.grid(alpha=0.3)

plt.show()

# ==========================================
# Graph 2 : Count Plot
# ==========================================

plt.figure(figsize=(7,5))

ax = sns.countplot(
    data=df,
    x="Sales Level",
    hue="Sales Level",
    palette=["royalblue","red"],
    legend=False
)

for container in ax.containers:
    ax.bar_label(container)

plt.title(
    "Count of Orders in Each Sales Level",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Sales Level")

plt.ylabel("Number of Orders")

plt.grid(axis="y", alpha=0.3)

plt.show()

# ==========================================
# Graph 3 : Pie Chart
# ==========================================

plt.figure(figsize=(6,6))

colors = ["royalblue","red"]

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    explode=(0.05,0.05)
)

plt.title(
    "Sales Level Distribution",
    fontsize=15,
    fontweight="bold"
)

plt.show()

# ==========================================
# Graph 4 : Box Plot
# ==========================================

plt.figure(figsize=(10,2))

sns.boxplot(
    x=df["Sales"],
    color="lightblue"
)

plt.xlim(0,1000)

plt.title(
    "Box Plot of Sales",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Sales")

plt.show()