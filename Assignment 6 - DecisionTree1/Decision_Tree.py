import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv('Decision_Tree.csv')

label_encoder = LabelEncoder()

data['Temperature_encoded'] = label_encoder.fit_transform(data['Temperature'])
data['Vibration_encoded'] = label_encoder.fit_transform(data['Vibration'])
data['Failure_encoded'] = label_encoder.fit_transform(data['Failure'])

print("Encoded Dataset:")
print(data)

X = data[['Temperature_encoded', 'Vibration_encoded']]
y = data['Failure_encoded']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

input = [[0, 2]]

prediction = model.predict(input)
print("Prediction Result:", prediction[0])

if prediction[0] == 1:
    print("The machine will fail.")
else:
    print("The machine will not fail.")