import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('Data_DecisionTree.csv')
X = data[['tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']]
y = data['fail']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)
input_data = [[4, 5, 3, 6, 1, 45, 5, 1]]


prediction = model.predict(input_data)
print("Prediction Result:", prediction[0])
if prediction[0] == 1:
    print("The machine will fail.")
else:
    print("The machine will not fail.")

plt.figure(figsize=(18, 10))
plot_tree(model,feature_names=['tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature'],class_names=['No Failure', 'Failure'],filled=True)
plt.title("Decision Tree Classifier")
plt.show()