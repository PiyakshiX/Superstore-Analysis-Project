import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_model.pkl")

# Load training columns
columns = joblib.load("model_columns.pkl")

print("===== SALES PREDICTION SYSTEM =====")

quantity = int(input("Enter Quantity: "))
discount = float(input("Enter Discount (0 - 0.8): "))
year = int(input("Enter Order Year: "))
month = int(input("Enter Order Month (1-12): "))

category = input("Category (Furniture/Office Supplies/Technology): ")

subcategory = input(
    "Sub-Category (Accessories, Appliances, Art, Binders, Bookcases, Chairs, Copiers, Envelopes, Fasteners, Furnishings, Labels, Machines, Paper, Phones, Storage, Supplies, Tables): "
)

region = input("Region (Central/East/South/West): ")

segment = input("Segment (Consumer/Corporate/Home Office): ")

shipmode = input(
    "Ship Mode (First Class/Second Class/Standard Class/Same Day): "
)

# Create dataframe
sample = pd.DataFrame({
    "Quantity": [quantity],
    "Discount": [discount],
    "Order Year": [year],
    "Order Month": [month],
    "Category": [category],
    "Sub-Category": [subcategory],
    "Region": [region],
    "Segment": [segment],
    "Ship Mode": [shipmode]
})

# One Hot Encoding
sample = pd.get_dummies(sample)

# Match columns
sample = sample.reindex(columns=columns, fill_value=0)

# Prediction
prediction = model.predict(sample)

print("\nPredicted Sales = {:.2f}".format(prediction[0]))