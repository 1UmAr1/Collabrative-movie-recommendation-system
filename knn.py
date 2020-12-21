import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

data = pd.read_csv("Book2.csv", encoding="cp850")
data.dropna(axis=0)

train, test = train_test_split(data, test_size=0.3)

model = NearestNeighbors(metric='cosine', algorithm="brute", n_neighbors=20, n_jobs=1)

model.fit(train)

print(model)
print(model.kneighbors(n_neighbors=4), )