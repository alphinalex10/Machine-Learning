import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import math
data = pd.read_csv("KNN_Dataset.csv")
x = data[["Temperature"]]
y = data[["Fuel_Consumption"]]


model = KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)

temp_1 = 58
y_pred_1 = model.predict([[temp_1]])
print("Predicted Fuel Consumption for Temperature 58: ", y_pred_1)


temp = int(input("Enter the temperature: "))
y_pred = model.predict([[temp]])
print()
print("Predicted Fuel Consumption: ", y_pred)
