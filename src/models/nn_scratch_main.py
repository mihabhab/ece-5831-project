import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import confusion_matrix_custom

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.models.nn_scratch import MLP

# Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Feature Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **************
# MLP Creation
# **************
mlp_relu = MLP(layers=[4, 10, 3],
               hidden_activation='relu',
               output_activation='softmax', 
               learning_rate=0.01)

mlp_tanh = MLP(layers=[4, 10, 3],
               hidden_activation='tanh',
               output_activation='softmax',
               learning_rate=0.01)

mlp_sigmoid = MLP(layers=[4, 10, 3],
                  hidden_activation='sigmoid',
                  output_activation='softmax',
                  learning_rate=0.01)
# **************
# Model Training
# **************
mlp_relu.fit(X_train, y_train, epochs=10000)
mlp_tanh.fit(X_train, y_train, epochs=10000)
mlp_sigmoid.fit(X_train, y_train, epochs=10000)

# **************
# MLP Prediction
# **************
y_pred_relu = mlp_relu.predict(X_test)
y_pred_tanh = mlp_tanh.predict(X_test)
y_pred_sigmoid = mlp_sigmoid.predict(X_test)

# Calculate test accuracy
test_accuracy_values = {
    'Model': ['MLP_relu', 'MLP_tanh', 'MLP_sigmoid'],
    'Test Accuracy': [
        accuracy_score(y_test, y_pred_relu), 
        accuracy_score(y_test, y_pred_tanh), 
        accuracy_score(y_test, y_pred_sigmoid)
    ]
}
train_accuracy_values = {
    'Model': ['MLP_relu', 'MLP_tanh', 'MLP_sigmoid'],
    'train Accuracy': [
        mlp_relu.accuracy(X_train, y_train), 
        mlp_tanh.accuracy(X_train, y_train), 
        mlp_sigmoid.accuracy(X_train, y_train)
    ]
}

# A DataFrame for Accuracy values
test_accuracy_df = pd.DataFrame(test_accuracy_values)
train_accuracy_df = pd.DataFrame(train_accuracy_values)

# Calculate classification reports
classification_reports = {
    'MLP_relu': classification_report(y_test, y_pred_relu, output_dict=True),
    'MLP_tanh': classification_report(y_test, y_pred_tanh, output_dict=True),
    'MLP_sigmoid': classification_report(y_test, y_pred_sigmoid, output_dict=True)
}

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
final_df = pd.merge(train_accuracy_df, test_accuracy_df, on='Model')
final_df = pd.merge(final_df, clss_rep_df, on='Model')

# Using Seaborn to plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(final_df.set_index('Model').astype(float), annot=True, cmap='Blues_r')
plt.title("Accuracy and Classification Report Heatmap - scratch")
plt.show()

# Plotting the confusion matrix
confusion_matrix_custom(y_true=y_test,
                        y_pred=y_pred_relu,
                        class_names=class_names,
                        title='Confusion Matrix | Relu - scratch',
                        cmap='Blues')

confusion_matrix_custom(y_true=y_test,
                        y_pred=y_pred_tanh,
                        class_names=class_names,
                        title='Confusion Matrix | Tanh - scratch',
                        cmap='Greys')

confusion_matrix_custom(y_true=y_test,
                        y_pred=y_pred_sigmoid,
                        class_names=class_names,
                        title='Confusion Matrix | Sigmoid - scratch',
                        cmap='OrRd')

# Plot decision boundaries
mlp_relu.plot_decision_boundaries(X=X_train, y=y_train,
                                  title='Decision boundary | MLP ReLu - scratch')
mlp_tanh.plot_decision_boundaries(X=X_train, y=y_train,
                                  title='Decision boundary | MLP Tanh - scratch')
mlp_sigmoid.plot_decision_boundaries(X=X_train, y=y_train,
                                     title='Decision boundary | MLP Sigmoid - scratch')

# Plot learning curve
mlp_relu.plot_learning_curve(title='Learning Curve | MLP ReLu - scratch')
mlp_tanh.plot_learning_curve(title='Learning Curve | MLP Tanh - scratch')
mlp_sigmoid.plot_learning_curve(title='Learning Curve | MLP Sigmoid - scratch')