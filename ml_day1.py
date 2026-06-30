import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")
print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)

X = df[['Discount', 'Quantity']]
y = df['Sales']
print("\nFeatures")
print(X.head())
print("\nTarget")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nModel Performance")
print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

comparison = pd.DataFrame({

    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred

})
print("\nFirst 10 Predictions")
print(comparison.head(10))