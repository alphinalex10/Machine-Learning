import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv("power_data.csv")
x = data[["Wind_Speed", "Blade_Angle", "Rotor_Speed"]]
y = data[["Power_Output"]]

model = LinearRegression()
model.fit(x,y)
print("Coeff: ",model.coef_)
print("Intercept: ",model.intercept_)

data_1 = 9,11,100
power_output_1 = model.predict([data_1])
print("The power output for (Wind_Speed=9, Blade_Angle=11, Rotor_Speed=100) is: ", power_output_1)

Wind_Speed= int(input("Enter the Wind_Speed : "))
Blade_Angle= int(input("Enter the Blade_Angle : "))
Rotor_Speed= int(input("Enter the Rotor_Speed : "))
power_output = model.predict([[Wind_Speed,Blade_Angle,Rotor_Speed]])
print("The power output is: ", power_output)