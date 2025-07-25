from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#load the dataset
iris = load_iris()

#The 'iris' object is a Bunch object (like a dictionary)
#it contains:
#- data (the features)
#- target (the species labels)
#- feature_names (names of the features)
#- target_names (names of the species)
#- DESCR (a description of the dataset)

# Let's use Pandas Dataframe for easier viewing and manipulation

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target (species) to the DataFrame
df['species'] = iris.target

# Add species names for better readability
df['species_names'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

# Display the first 5 rows of the DataFrame to see what we have
print(df.head())

# Also, let;s see some basic info about the dataset
print("\nDataset Info")
print(df.info())

print("\nSpecies distribution:")
print(df['species_names'].value_counts())

# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(df.describe())

# Set the style for plots
sns.set_style("whitegrid")

# Create histograms for each feature
# We'll plot each feature on a separate subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot Sepal Length
sns.histplot(df['sepal length (cm)'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0,0].set_title('Distribution of Sepal Length')

# Plot Sepal Width
sns.histplot(df['sepal width (cm)'], kde=True, ax=axes[0, 1], color='lightcoral')
axes[0,1].set_title('Distribution of Sepal Width')

# Plot Petal Length
sns.histplot(df['petal length (cm)'], kde = True, ax = axes[1,0], color = 'lightgreen')
axes[1,0].set_title('Distribution of Petal Length')

# Plot Petal Width
sns.histplot(df['petal width (cm)'], kde = True, ax = axes[1,1], color = 'gold')
axes[1,1].set_title('Distribution of Petal Width')

plt.tight_layout() # Adjust subplot params for a tight layout
plt.show()

# Box plots to see feature distribution per species
plt.figure(figsize=(14, 10))

plt.subplot(2,2,1) # (rows, columns, plot_number)
sns.boxplot(x= 'species_names', y= 'sepal length (cm)', data = df)
plt.title('Sepal Length by Species')

plt.subplot(2,2,2)
sns.boxplot(x= 'species_names', y= 'sepal width (cm)', data = df)
plt.title('Sepal Width by Species')

plt.subplot(2,2,3)
sns.boxplot(x= 'species_names', y= 'petal length (cm)', data = df)
plt.title('Petal Length by Species')

plt.subplot(2,2,4)
sns.boxplot(x= 'species_names', y= 'petal width (cm)', data = df)
plt.title('Petal Width by Species')

plt.tight_layout()
plt.show()

# Pair plot - shows scatter plots for all pairs of features,
# and histograms/KDEs for single features on the diagonal.
# The 'hue' parameter colors points by species.

sns.pairplot(df, hue= 'species_names', diag_kind = 'kde')
plt.suptitle('Pair Plot of Iris Dataset Features by Species', y = 1.02) #Add title to entire plot
plt.show()

# Calculate the correlation matrix
correlation_matrix = df[iris.feature_names].corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title('Correlation Matrix of Iris Features')
plt.show()

# X will contain our features
X = df[iris.feature_names]

# y will contain our target variable (the species as numerical labels 0, 1, 2)
Y = df['species']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", Y.shape)
print("\nFirst 5 rows of X:")
print(X.head())
print("\nFirst 5 values of Y:")
print(Y.head())

# Split the data into training and testing sets
# test_size = 0.30 means 30% of the data will be used for testing, 70 for training
# random_state ensures reproducibility (you'll get the same split every time you run it)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 42)

print("\nTraining Features (X_train) shape:", X_train.shape)
print("Testing Features (X_test) shape:", X_test.shape)
print("Training Target (Y_train) shape:", Y_train.shape)
print("Testing Target (Y_test) shape:", Y_test.shape)

# Initialize the KNN classifier
# n_neighbors is 'k', the number of neighbors to consider.
# A common starting point is 3 or 5. Let's start with 3.
knn_model = KNeighborsClassifier(n_neighbors = 5)

print("\nKNN Model Initialized:")
print(knn_model)

# Train the model using the training data
knn_model.fit(X_train, Y_train)

print("\nKNN Model trained successfully!")
