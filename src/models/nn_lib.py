
import os # operating system
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from utils import confusion_matrix_custom, new_learning_curve

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier # multi layer perceptron
from sklearn.model_selection import LearningCurveDisplay, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the Iris dataset (NOTE: the data is in iris.data (data attribute) dimension 150x4)
iris = load_iris()

print(f'measurements/features: {iris.feature_names}') # name of the 4 features
print(f'Species: {iris.target_names}\n') # 0 = Setosa, 1 = Vesicolor, 2 = Virginica 
# print(f'Species representation: {iris.target}\n') # intigers representig the 3 species

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Standardization
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_standardized['species'] = df['species']

# Normalization
scaler_normalize = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_normalize.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])
df_normalized['species'] = df['species']

# Separate features and target
X = df_normalized.iloc[:, :-1]
y = df_normalized['species'].cat.codes
class_names = iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the MLP models
mlp_relu = MLPClassifier(hidden_layer_sizes=(128,),
                    activation='relu',
                    max_iter=1000,
                    random_state=42)
mlp_tanh = MLPClassifier(hidden_layer_sizes=(128,),
                    activation='tanh',
                    max_iter=1000,
                    random_state=42)
mlp_logistic = MLPClassifier(hidden_layer_sizes=(128,),
                    activation='logistic',
                    max_iter=1000,
                    random_state=42)

# Train the models
mlp_relu.fit(X_train, y_train)
mlp_tanh.fit(X_train, y_train)
mlp_logistic.fit(X_train, y_train)

# Model Predictions
y_pred_relu = mlp_relu.predict(X_test)
y_pred_tanh = mlp_tanh.predict(X_test)
y_pred_logistic = mlp_logistic.predict(X_test)

# Calculate test accuracy
test_accuracy_values = {
    'Model': ['MLP_relu', 'MLP_tanh', 'MLP_logistic'],
    'Test Accuracy': [
        accuracy_score(y_test, y_pred_relu), 
        accuracy_score(y_test, y_pred_tanh), 
        accuracy_score(y_test, y_pred_logistic)
    ]
}
# Calculate training accuracy
train_accuracy_values = {
    'Model': ['MLP_relu', 'MLP_tanh', 'MLP_logistic'],
    'Training Accuracy': [
        mlp_relu.score(X_train, y_train), 
        mlp_tanh.score(X_train, y_train), 
        mlp_logistic.score(X_train, y_train)
    ]
}
# Calculate classification reports
classification_reports = {
    'MLP_relu': classification_report(y_test, y_pred_relu, output_dict=True),
    'MLP_tanh': classification_report(y_test, y_pred_tanh, output_dict=True),
    'MLP_logistic': classification_report(y_test, y_pred_logistic, output_dict=True)
}
# A DataFrame for Accuracy values
test_accuracy_df = pd.DataFrame(test_accuracy_values)
train_accuracy_df = pd.DataFrame(train_accuracy_values)
# Extract precision, recall, f1-score for each model and combine into one DataFrame
data = []

for model_name, report in classification_reports.items():
    row = {
        'Model': model_name,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-score': report['weighted avg']['f1-score']
    }
    data.append(row)

clss_rep_df = pd.DataFrame(data)
# Merge DataFrames
final_df = pd.merge( train_accuracy_df, test_accuracy_df, on='Model')
final_df = pd.merge(final_df, clss_rep_df, on='Model')
# final_df = pd.merge(final_df, on='Model')

# Using Seaborn to plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(final_df.set_index('Model').astype(float), annot=True, cmap='Blues_r')
plt.title("Accuracy and Classification Report Heatmap - sklearn")
plt.show()
#-----------------------------------------------
#-----------------------------------------------

# Plotting the confusion matrix
confusion_matrix_custom(y_true=y_test,
                        y_pred=y_pred_relu,
                        class_names=class_names,
                        title='Confusion Matrix | Relu - sklearn',
                        cmap='Blues')

confusion_matrix_custom(y_true=y_test,
                        y_pred=y_pred_tanh,
                        class_names=class_names,
                        title='Confusion Matrix | Tanh - sklearn',
                        cmap='Greys')

confusion_matrix_custom(y_true=y_test,
                        y_pred=y_pred_logistic,
                        class_names=class_names,
                        title='Confusion Matrix | Logistic - sklearn',
                        cmap='OrRd')

#-----------------------------------------------
#-----------------------------------------------

# Plotting the learning curve
new_learning_curve(mlp_relu, X, y, "Learning Curve for MLP Classifier | ReLu - sklearn")
new_learning_curve(mlp_tanh, X, y, "Learning Curve for MLP Classifier | Tanh - sklearn")
new_learning_curve(mlp_logistic, X, y, "Learning Curve for MLP Classifier | Logistic - sklearn")

#-----------------------------------------------
#-----------------------------------------------

# Plotting the decision boundary
# decision_boundary(mlp_relu, X_train, y_train,
#                   title="Decision Boundary on Training Data | Relu",
#                   target_names=class_names)
# decision_boundary(mlp_tanh, X_train, y_train,
#                   title="Decision Boundary on Training Data | Tanh",
#                   target_names=class_names)
# decision_boundary(mlp_logistic, X_train, y_train,
#                   title="Decision Boundary on Training Data | Logistic",
#                   target_names=class_names)

#-----------------------------------------------
#-----------------------------------------------