import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({i: species for i, species in enumerate(iris.target_names)})

# 1. Histogram of sepal length
plt.figure(figsize=(6, 4))
plt.hist(df['sepal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 2. Boxplot of all features
plt.figure(figsize=(10, 6))
df.drop('species', axis=1).plot(kind='box')
plt.title('Boxplot of Iris Features')
plt.grid(True)
plt.show()

# 3. Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], label=species)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Pie chart of species distribution
species_counts = df['species'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Species Distribution')
plt.show()
