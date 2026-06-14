import pandas as pd
import joblib

model = joblib.load('wine_quality_model.pkl')

fixed_acidity = float(input("Enter fixed acidity: "))
volatile_acidity = float(input("Enter volatile acidity: "))
citric_acid = float(input("Enter citric acid: "))
residual_sugar = float(input("Enter residual sugar: "))
chlorides = float(input("Enter chlorides: "))
free_sulfur_dioxide = float(input("Enter free sulfur dioxide: "))
total_sulfur_dioxide = float(input("Enter total sulfur dioxide: "))
density = float(input("Enter density: "))
pH = float(input("Enter pH: "))
sulphates = float(input("Enter sulphates: "))
alcohol = float(input("Enter alcohol: "))

input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,  free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
prediction = model.predict(input_data)

print("Predicted Wine Quality:", prediction[0])
if prediction[0] >= 6:
    print("Good quality wine")
else:
    print("Average quality wine")
