from sklearn.datasets import load_iris
import pandas as pd

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