import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Sample - Superstore.csv", encoding="latin1")

df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
print(df[['Order Date','Order Year','Order Month']].head())

features = [
    'Quantity',
    'Discount',
    'Order Year',
    'Order Month',
    'Category',
    'Sub-Category',
    'Region',
    'Segment',
    'Ship Mode'
]
X = df[features]
y = df['Sales']

X = pd.get_dummies(
    X,
    columns=[
        'Category',
        'Sub-Category',
        'Region',
        'Segment',
        'Ship Mode'
    ],
    drop_first=True
)
print(X.head())
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients.sort_values(by="Coefficient", ascending=False).head(10))