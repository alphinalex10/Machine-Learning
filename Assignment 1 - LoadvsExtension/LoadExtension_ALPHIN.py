
import pandas as pd
import matplotlib.pyplot as mpt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("data.csv")
x = data[["Load (N)"]]
y = data[["Extension (mm)"]]

mpt.scatter(x,y)
mpt.show()
model = LinearRegression()
model.fit(x,y)
print("Coeff: ",model.coef_)
print("Intercept: ",model.intercept_)

load_1 = 55
extension_1 = model.predict([[load_1]])
print("The extension for load 55 is: ", extension_1)

load = int(input("Enter the load: "))
extension = model.predict([[load]])
print("The extension is: ", extension)