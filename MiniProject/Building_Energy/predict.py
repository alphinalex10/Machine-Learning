import pandas as pd
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

surf_area = float(input("Enter Surface Area (m2): "))
wall_area = float(input("Enter Wall Area (m2): "))
roof_area = float(input("Enter Roof Area (m2): "))
height = float(input("Enter Height (m): "))
glazing = float(input("Enter Glazing (% of floor area): "))

input_data = pd.DataFrame([[surf_area, wall_area, roof_area, height, glazing]],
                          columns=['SurfaceArea', 'WallArea', 'RoofArea', 'Height', 'Glazing'])

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)

print("Predicted Heating Load:", prediction[0])
