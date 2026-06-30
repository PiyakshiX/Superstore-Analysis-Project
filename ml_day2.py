import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
print(df.head())

features = [
    'Quantity',
    'Discount',
    'Category',
    'Region',
    'Segment'
]
X = df[features]
y = df['Sales']

print(X.head())

X = pd.get_dummies(
    X,
    columns=['Category', 'Region', 'Segment'],
    drop_first=True
)
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))