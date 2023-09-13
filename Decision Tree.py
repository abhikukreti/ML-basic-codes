
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

data = pd.DataFrame({
    'Size': [1, 2, 3, 2, 3, 4, 1, 2, 3, 4],
    'Color': [1, 2, 2, 3, 3, 1, 2, 3, 3, 1],
    'Label': ['rotten', 'rotten', 'rotten', 'rotten', 'rotten', 'fresh', 'fresh', 'fresh', 'fresh', 'fresh']
})

X = data[['Size', 'Color']]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier() //classifier model


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)//predict 

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)
