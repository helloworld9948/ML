
# Machine Learning Lab Manual

---

## Experiment 1: Central Tendency Measures & Dispersion

### **Compute Central Tendency Measures: Mean, Median, Mode; Measure of Dispersion: Variance, Standard Deviation.**

#### Mean
- **Definition**: The mean value is the average value.
- **Formula**: 
  \[
  \text{Mean} = \frac{\sum \text{values}}{\text{number of values}}
  \]

**Example Calculation**:
```python
import numpy as np

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = np.mean(speed)
print(x)  # Output: 89.76923076923077
```

#### Median
- **Definition**: The median value is the value in the middle after sorting all the values.

**Example Calculation**:
```python
import numpy as np

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = np.median(speed)
print(x)  # Output: 87.0
```

#### Mode
**Example Calculation**:
```python
from scipy import stats

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = stats.mode(speed)
print(x)  # Output: ModeResult(mode=array([86]), count=array([3]))
```

#### Variance
- **Definition**: Variance indicates how spread out the values are.

**Example Calculation**:
```python
import numpy as np

speed = [32, 111, 138, 28, 59, 77, 97]
x = np.var(speed)
print(x)  # Output: 1432.2448979591834
```

#### Standard Deviation
- **Definition**: The square root of the variance.

**Example Calculation**:
```python
import numpy as np

speed = [32, 111, 138, 28, 59, 77, 97]
x = np.std(speed)
print(x)  # Output: 37.84501153334721
```

---

## Experiment 2: Study of Python Basic Libraries

### **Study Python Basic Libraries: Statistics, Math, Numpy, and Scipy.**

#### Statistics Module
- **Description**: Python's built-in module for calculating mathematical statistics.

**Example**:
```python
import statistics

print(statistics.mean([1, 3, 5, 7, 9, 11, 13]))  # Output: 7
print(statistics.mean([1, 3, 5, 7, 9, 11]))     # Output: 6
print(statistics.mean([-11, 5.5, -3.4, 7.1, -9, 22]))  # Output: 1.8666666666666667
```

#### Math Module
- **Description**: Contains additional functions for mathematical calculations.

**Example**:
```python
import math

r = 4
pie = math.pi
print(pie * r * r)  # Output: 50.26548245743669
```

#### NumPy Module
- **Description**: A fundamental package for scientific computing with Python.

**Example**:
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)  # Output: [1 2 3 4 5]
```

#### SciPy Module
- **Description**: A scientific computation library that uses NumPy underneath.

**Example**:
```python
from scipy import constants

print(constants.pi)  # Output: 3.141592653589793
```

---

## Experiment 3: Study of Python Libraries for ML Applications

### **Study Python Libraries for ML applications: Pandas and Matplotlib.**

#### Pandas
- **Description**: Provides high-level data structures and tools for data analysis.

**Example**:
```python
import pandas as pd

data = {
    "country": ["Brazil", "Russia", "India", "China", "South Africa"],
    "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
    "area": [8.516, 17.10, 3.286, 9.597, 1.221],
    "population": [200.4, 143.5, 1252, 1357, 52.98]
}

data_table = pd.DataFrame(data)
print(data_table)
```

#### Matplotlib
- **Description**: A library for data visualization.

**Example**:
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.plot(x, x, label ='linear')
plt.legend()
plt.show()
```

---

## Experiment 4: Simple Linear Regression

### **Implement Simple Linear Regression in Python.**

**Program**:
```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_coeff(p, q):
    n1 = np.size(p)
    m_p = np.mean(p)
    m_q = np.mean(q)
    SS_pq = np.sum(q * p) - n1 * m_q * m_p
    SS_pp = np.sum(p * p) - n1 * m_p * m_p
    b_1 = SS_pq / SS_pp
    b_0 = m_q - b_1 * m_p
    return (b_0, b_1)

def plot_regression_line(p, q, b):
    plt.scatter(p, q, color="m", marker="o", s=30)
    q_pred = b[0] + b[1] * p
    plt.plot(p, q_pred, color="g")
    plt.xlabel('p')
    plt.ylabel('q')
    plt.show()

def main():
    p = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    q = np.array([11, 13, 12, 15, 17, 18, 18, 19, 20, 22])
    b = estimate_coeff(p, q)
    print("Estimated coefficients are:\nb_0 = {}\nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(p, q, b)

if __name__ == "__main__":
    main()
```

---

## Experiment 5: Multiple Linear Regression for House Price Prediction

### **Implement Multiple Linear Regression for House Price Prediction using sklearn.**

**Installation Commands**:
```bash
sudo apt update
sudo apt install python3 python3-pip
pip3 install pandas numpy matplotlib scikit-learn
```

**Program**:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a synthetic dataset
np.random.seed(42)
num_samples = 100
data = {
    'feature1': np.random.randint(1000, 4000, num_samples),  # Size of the house in sq ft
    'feature2': np.random.randint(1, num_samples),           # Number of bedrooms
    'feature3': np.random.randint(1, 4, num_samples),        # Number of bathrooms
    'price': np.random.randint(100000, 500000, num_samples)  # House price
}

# Create a DataFrame
df = pd.DataFrame(data)
df.to_csv('house_prices.csv', index=False)

# Step 2: Load the dataset
data = pd.read_csv('house_prices.csv')

# Step 3: Display the first few rows of the dataset
print("Dataset Head:")
print(data.head())

# Step 4: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 5: Fill or drop missing values (if any)
data.fillna(data.mean(), inplace=True)

# Step 6: Select features and target variable
X = data[['feature1', 'feature2', 'feature3']]  # Features
y = data['price']  # Target variable (house price)

# Step 7: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 11: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)  # 45-degree line
plt.show()
```

---

## Experiment 6: Decision Tree Implementation and Parameter Tuning

### **Implement Decision Tree using sklearn and perform parameter tuning.**

#### Importing the Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
%matplotlib inline
```

#### Data Preprocessing
```python
# Reading the CSV file
df = pd.read_csv(r"/kaggle/input/heart-disease-prediction/heart_v2.csv")
df.columns
# Displaying first few rows
df.head()
# Data Information
df.info()
# Data Shape
df.shape
```

#### Data Visualization
```python
plt.figure(figsize=(10,5))
sns.violinplot(df['age'])
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(df['sex'])
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(df['BP'])
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(df['cholestrol'])
plt.show()
plt.figure(figsize=(15,5))
sns.countplot(df['heart disease'])
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(y='age', x='heart disease', data=df)
plt.show()
plt.figure(figsize=(10,5))
sns.countplot(x="sex", hue="heart disease", data=df)
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(y='BP', x='heart disease', data=df)
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(y='cholestrol', x='heart disease', data=df)
plt.show()
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, cmap="rainbow")
plt.show()
```

#### Splitting the Dataset
```python
# Features and Target Variable
X = df.drop('heart disease', axis=1)
y = df['heart disease']
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=50)
```

#### Training the Decision Tree
```python
# Fitting the decision tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
# Plotting the tree
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dt, feature_names=X.columns, class_names=['No Disease', "Disease"], filled=True)
```

#### Evaluating Model Performance
```python
# Predictions
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
# Accuracy and Confusion Matrix
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Train Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
```

#### Parameter Tuning
```python
# Control Tree Depth
dt_depth = DecisionTreeClassifier(max_depth=3)
dt_depth.fit(X_train, y_train)
# Minimum Samples Before Split
dt_min_split = DecisionTreeClassifier(min_samples_split=20)
dt_min_split.fit(X_train, y_train)
```

---

## Experiment 7: K-Nearest Neighbors (KNN) Implementation

### **Implement KNN using sklearn.**

#### Importing the Required Libraries
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
df.to_csv('breast_cancer.csv', index=False)
print("Dataset saved as 'breast_cancer.csv'")

df = pd.read_csv('breast_cancer.csv')
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

---

## Experiment 8: Logistic Regression Implementation

### **Implement Logistic Regression using sklearn.**

#### Importing the Required Libraries
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
```

#### Implementing Logistic Regression
```python
# Load dataset
X, Y = load_iris(return_X_y=True)
# Create Logistic Regression Model
logreg = LogisticRegression(random_state=0)
logreg.fit(X, Y)
# Predictions
Y_pred = logreg.predict(X[:2, :])
print(Y_pred)
Y_predict = logreg.predict_proba(X[:2, :])
print(Y_predict)
# Model Accuracy
score = logreg.score(X, Y)
print(score)
```

---

## Experiment 9: K-Means Clustering Implementation

### **Implement K-Means Clustering.**

#### Importing the Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

#### Implementing K-Means Clustering
```python
# Generate random data
np.random.seed(42)
X = np.random.randn(100, 2)
# Plot the data
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
# Perform K-Means Clustering
kmeans = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
kmeans.fit(X)
# Plot the clusters and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

---

## Contributors

- [TEJA](https://github.com/helloworld9948)

