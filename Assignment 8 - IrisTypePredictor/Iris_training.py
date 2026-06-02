import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("Iris.csv")
x = data.drop(columns=["Id", "Species"])
y = data["Species"]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)
print("Model Accuracy:", accuracy*100, "%")

joblib.dump(model, "iris_model.pkl")
print("Model saved successfully")