import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

# ==========================================
# Date Features
# ==========================================

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Order Year"] = df["Order Date"].dt.year
df["Order Month"] = df["Order Date"].dt.month

# ==========================================
# Create L1 / L2 Target
# ==========================================

threshold = 610

df["Sales Level"] = df["Sales"].apply(
    lambda x: "L1" if x <= threshold else "L2"
)

print("\nSales Level Count\n")
print(df["Sales Level"].value_counts())

# ==========================================
# Features
# ==========================================

X = df[
[
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
]

# One-Hot Encoding

X = pd.get_dummies(X)

# ==========================================
# Encode Target
# ==========================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["Sales Level"])

print("\nTarget Classes")
print(encoder.classes_)

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==========================================
# KNN Model
# ==========================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = knn.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== KNN RESULT ==========")

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))