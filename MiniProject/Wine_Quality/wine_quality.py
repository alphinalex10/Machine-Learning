import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_csv("winequality-red.csv")

print("Dataset Shape:", data.shape)
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())
data = data.drop_duplicates()
print("Shape after dropping duplicates:", data.shape)

X = data.drop('quality', axis=1)
y = data['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

joblib.dump(model, 'wine_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved.")