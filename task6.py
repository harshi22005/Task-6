# Task 6: KNN Classification using your Iris.csv dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load YOUR DATASET
# ---------------------------------------
df = pd.read_csv("C:\\Users\\G HARSHITHA\\Downloads\\archive (7)\\Iris.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Remove ID column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# ---------------------------------------
# 2. Split Features and Target
# ---------------------------------------
X = df.drop("Species", axis=1)
y = df["Species"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# KNN Model
# ---------------------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ---------------------------------------
# Evaluation
# ---------------------------------------
y_pred = knn.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------
# K vs Accuracy graph
# ---------------------------------------
accuracies = []
k_values = range(1, 11)

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.grid()
plt.show()

