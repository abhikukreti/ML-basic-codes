from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np

# Generate some example data (replace this with your dataset)
X = np.random.rand(100, 2)  # Example features
y = np.random.randint(0, 2, 100)  # Example labels (binary classification)

# Initialize a variable to store the accuracy scores
accuracy_scores = []

# Create a LeaveOneOut cross-validator
loo = LeaveOneOut()

# Perform leave-one-out cross-validation
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
   
    accuracy = 0.0 
    accuracy_scores.append(accuracy)


average_accuracy = np.mean(accuracy_scores)
print(f"Average Accuracy: {average_accuracy}")
