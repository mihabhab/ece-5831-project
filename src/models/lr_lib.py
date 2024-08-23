import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
#print(df.head())
# Standardization
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_standardized['species'] = df['species']
#print(df_standardized)
# Normalization
scaler_normalize = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_normalize.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_normalized['species'] = df['species']
print(df_standardized)
# # Visualize the standardized data
# sns.pairplot(df_standardized, hue="species", markers=["o", "s", "D"])
# plt.suptitle('Pairwise scatter plots of the standardized Iris features', verticalalignment='bottom')
# plt.show()

# # Visualize the normalized data
# sns.pairplot(df_normalized, hue="species", markers=["o", "s", "D"])
# plt.suptitle('Pairwise scatter plots of the normalized Iris features', verticalalignment='bottom')
# plt.show()

# Convert labels to one-hot encoding
X = df_standardized.drop('species', axis=1)  # Features
y = df_standardized['species']              # Labels
print(X)
print(y)
#---------------------

# Split the dataset into training and testing sets
# 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
# Standardization improves the convergence of the logistic regression algorithm
# NOTE isn't this redundant? X & y are already standardized
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
# Using 'multinomial' for multi-class classification and 'lbfgs' solver which handles multinomial loss
model = LogisticRegression(multi_class='multinomial',
                           solver='lbfgs',
                           max_iter=1000)

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the class labels for the test set
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Evaluate the model
# Confusion matrix to see the performance of the classification
cm = confusion_matrix(y_test, y_pred_test)

# Classification report for detailed metrics
report = classification_report(y_test, y_pred_test, target_names=iris.target_names)

# Calculate accuracy
train_accuracy = np.mean(y_pred_train == y_train)
test_accuracy = np.mean(y_pred_test == y_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

precision = precision_score(y_test, y_pred_test, average='macro')
recall = recall_score(y_test, y_pred_test, average='macro')
f1 = f1_score(y_test, y_pred_test, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

# Print out the mean accuracy and the accuracy of each fold
print("Accuracy scores for each fold:", scores)
print("Mean accuracy:", scores.mean())

# Print the evaluation results
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()