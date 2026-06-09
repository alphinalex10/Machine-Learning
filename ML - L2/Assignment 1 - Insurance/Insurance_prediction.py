import pandas as pd
import joblib

model = joblib.load("model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

sex_encoder = joblib.load("sex.pkl")
smoker_encoder = joblib.load("smoker.pkl")
region_encoder = joblib.load("region.pkl")

age = int(input("Enter Age: "))
sex = input("Enter Sex (male/female): ")
bmi = float(input("Enter BMI: "))
children = int(input("Enter Number of Children: "))
smoker = input("Enter Smoker Status (yes/no): ")
region = input("Enter Region (southwest/southeast/northwest/northeast): ")

input_data = pd.DataFrame([[ age,bmi,children,sex_encoder.transform([sex])[0],smoker_encoder.transform([smoker])[0],region_encoder.transform([region])[0]]],
    columns=['age','bmi','children','sex_encoded','smoker_encoded','region_encoded'])
input_data_scaled = x_scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
real_prediction = y_scaler.inverse_transform(prediction)
print("Predicted Medical Insurance Cost:", real_prediction)