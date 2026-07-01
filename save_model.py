import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Create new features
df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month

# Select features
features = [
    "Quantity",
    "Discount",
    "Order Year",
    "Order Month",
    "Category",
    "Sub-Category",
    "Region",
    "Segment",
    "Ship Mode"
]

X = df[features]
y = df["Sales"]

# One Hot Encoding
X = pd.get_dummies(
    X,
    columns=[
        "Category",
        "Sub-Category",
        "Region",
        "Segment",
        "Ship Mode"
    ],
    drop_first=True
)

# Save column names
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "sales_model.pkl")

print("Model saved successfully!")