import pandas as pd
import numpy as np
from sklearn.manifold import MCA

# Load the Titanic data set
df = pd.read_csv('titanic.csv')

# Create an MCA object
mca = MCA(n_components=2)

# Fit the MCA object to the data set
mca.fit(df)

# Project the data onto a lower-dimensional space
X = mca.transform(df)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1])
plt.show()