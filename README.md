Task 6 – K-Nearest Neighbors (KNN) Classification
📌 Objective

The goal of this task is to understand and implement the K-Nearest Neighbors (KNN) algorithm for classification, using a real dataset.
The model is trained, evaluated, and visualized to demonstrate how distance-based classification works.

🛠️ Tools & Libraries Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

📂 Dataset

Dataset used: Iris Dataset

File: Iris.csv

Columns include:
Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species

📌 Steps Performed
✔ 1. Load the dataset

The dataset is read from a local CSV file.

✔ 2. Preprocess data

Removed the Id column

Encoded the target column Species

Normalized features using StandardScaler

✔ 3. Train/Test Split

The dataset is split into 80% train and 20% test.

✔ 4. Train KNN Model

Used KNeighborsClassifier from scikit-learn.

✔ 5. Evaluate Performance

Accuracy Score

Confusion Matrix

✔ 6. Mathematical Understanding

The KNN algorithm predicts the class based on majority vote of k nearest data points.

🧪 Model Output

Example output when running the code:

Accuracy: 1.0

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

▶️ How to Run

Install required libraries:

pip install pandas numpy scikit-learn matplotlib


Place your dataset (Iris.csv) in the correct path.

Run the script:

python task6.py

👩‍💻 Author

G Harshitha
AIML student
