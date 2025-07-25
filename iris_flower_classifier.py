from sklearn.datasets import load_iris
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