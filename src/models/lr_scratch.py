import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# Visualize the standardized data
sns.pairplot(df_standardized, hue="species", markers=["o", "s", "D"])
plt.suptitle('Pairwise scatter plots of the standardized Iris features', verticalalignment='bottom')
plt.show()

# Visualize the normalized data
sns.pairplot(df_normalized, hue="species", markers=["o", "s", "D"])
plt.suptitle('Pairwise scatter plots of the normalized Iris features', verticalalignment='bottom')
plt.show()

# Convert labels to one-hot encoding
X = df_standardized.drop('species', axis=1)  # Features
y = df_standardized['species']              # Labels
print(X)
print(y)
def manual_label_encode(y):
    
    unique_classes = np.unique(y)
    class_to_int = {key: idx for idx, key in enumerate(unique_classes)}

    integer_encoded = np.array([class_to_int[item] for item in y])
    return integer_encoded
def manual_one_hot_encode(y):
    # Conversion to integer encoding
    integer_encoded = manual_label_encode(y)

    # Creating One Hot Codes
    one_hot = np.zeros((integer_encoded.size, integer_encoded.max() + 1))
    one_hot[np.arange(integer_encoded.size), integer_encoded] = 1
    return one_hot

y_one_hot = manual_one_hot_encode(y)
print(y_one_hot)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
print(y_train)

# Softmax function
def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-12)) / m
    return loss

# Gradient computation
def compute_gradients(X, y, y_pred):
    m = X.shape[0]
    dz = y_pred - y
    dw = np.dot(X.T, dz) / m
    return dw

# Update weights
def update_weights(w, dw, learning_rate=0.01):
    w -= learning_rate * dw
    return w

# Training logistic regression from scratch
def train_logistic_regression(X, y, learning_rate=0.01, epochs=100):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    weights = np.random.rand(X.shape[1], y.shape[1])
    for epoch in range(epochs):
        scores = np.dot(X, weights)
        predictions = softmax(scores)
        loss = cross_entropy_loss(predictions, y)
        gradients = compute_gradients(X, y, predictions)
        weights = update_weights(weights, gradients, learning_rate)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return weights

# Train the model
weights = train_logistic_regression(X_train, y_train,epochs = 1000)

# Prediction
def predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    scores = np.dot(X, weights)
    predictions = softmax(scores)
    return np.argmax(predictions, axis=1)

# Evaluate the model
y_pred_train = predict(X_train, weights)
y_pred_test = predict(X_test, weights)

# Convert one-hot to labels for accuracy calculation
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
train_accuracy = np.mean(y_pred_train == y_train_labels)
test_accuracy = np.mean(y_pred_test == y_test_labels)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision, recall, and F1-score
precision = precision_score(y_test_labels, y_pred_test, average='macro')
recall = recall_score(y_test_labels, y_pred_test, average='macro')
f1 = f1_score(y_test_labels, y_pred_test, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

from sklearn.metrics import classification_report, confusion_matrix
y_test_labels = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_labels, y_pred_test)

report = classification_report(y_test_labels, y_pred_test, target_names=iris.target_names)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (From Scratch)')
plt.show()

from sklearn.model_selection import KFold

def cross_validate(X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    # Make sure X and y are numpy array
    X_np = X if isinstance(X, np.ndarray) else X.values
    y_np = y if isinstance(y, np.ndarray) else y.values

    for train_index, test_index in kf.split(X_np):
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
                
        
        
        weights = train_logistic_regression(X_train, y_train, epochs=10000)
        
        y_pred_test = predict(X_test, weights)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(y_pred_test == y_test_labels)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

# Perform cross-validation
cv_accuracy = cross_validate(X, y_one_hot)
print("Cross-Validation Accuracy:", cv_accuracy)

def plot_decision_boundaries(ax, X_train, y_train, X_test, y_test, model_weights, feature_names):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid = np.insert(X_grid, 0, 1, axis=1)  
    
    Z = softmax(np.dot(X_grid, model_weights))
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=100, cmap=plt.cm.Spectral, label="Train")

    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='x', s=100, cmap=plt.cm.Spectral, label="Test")
    
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f'{feature_names[0]} vs {feature_names[1]}')
    ax.legend()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

selected_features = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, (f1, f2) in enumerate(selected_features):
    X_train_sub = X_train.iloc[:, [f1, f2]].values
    X_test_sub = X_test.iloc[:, [f1, f2]].values
    feature_names = [iris.feature_names[f1], iris.feature_names[f2]]
    
    weights_sub = train_logistic_regression(X_train_sub, y_train, epochs=10000)
    
    
    ax = axes[i // 3, i % 3]
    plot_decision_boundaries(ax, X_train_sub, np.argmax(y_train, axis=1), X_test_sub, np.argmax(y_test, axis=1), weights_sub, feature_names)

plt.suptitle("Decision Boundary (From Scratch)", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Decision_Boundary_From_Scratch_epoch10000.jpeg',dpi=1000)
plt.show()

def logistic_regression(X, y, learning_rate=0.01, epochs=100):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    weights = np.random.rand(X.shape[1], y.shape[1])
    train_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for epoch in range(epochs):
        scores = np.dot(X, weights)
        predictions = softmax(scores)
        loss = cross_entropy_loss(predictions, y)
        gradients = compute_gradients(X, y, predictions)
        weights = update_weights(weights, gradients, learning_rate)
        
        # Evaluate the model
        y_pred_train = predict(X_train, weights)
        y_pred_test = predict(X_test, weights)
 
        # Convert one-hot to labels for accuracy calculation
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_test == y_test_labels)
        precision = precision_score(y_test_labels, y_pred_test, average='macro')
        recall = recall_score(y_test_labels, y_pred_test, average='macro')
        f1 = f1_score(y_test_labels, y_pred_test, average='macro')
        
        # Record metrics
        train_accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    return weights, train_accuracies, precisions, recalls, f1_scores

# Train the model
weights, accuracies, precisions, recalls, f1s = logistic_regression(X_train, y_train, epochs=10000)

# Plot the metrics
plt.figure(figsize=(10, 8))
plt.plot(precisions, label='Precision')
plt.plot(recalls, label='Recall')
plt.plot(f1s, label='F1 Score')
plt.title('Training Metrics Over Epochs (From Scratch)')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True)
plt.savefig('Training_Metrics_Over_Epochs_From_Scratch.jpeg',dpi=1000)
plt.show()