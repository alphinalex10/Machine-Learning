import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("machine_failure.csv")
data = data.drop(["UDI","Product ID"], axis=1)

label_encoder = LabelEncoder()
data["Machine_Type_encoded"] = label_encoder.fit_transform(data["Type"])
X = data[["Machine_Type_encoded", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "TWF", "HDF", "PWF", "OSF", "RNF"]]
Y = data[["Machine failure"]]

model = LogisticRegression()
model.fit(X,Y)

Machine_Type = "M"
mtype_enc = label_encoder.transform([Machine_Type])[0]
Machine_failure = model.predict([[mtype_enc, 301, 311, 1450, 52, 28, 1, 0, 0, 0, 0]])
print("Predicted Machine Failure (0 = no failure, 1 = failure): ", Machine_failure)
if Machine_failure == 1:
    print("The machine will fail.")
else:    print("The machine will not fail.")
