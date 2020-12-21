import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


Movies = pd.read_csv("Data/Precossed_Data.csv", encoding="cp850")
print(Movies.head())
Movies.rename(columns={"id": "movieId"}, inplace=True)
rating = pd.read_csv("Data/ratings_small.csv")
print(rating.head())

Data = pd.merge(Movies, rating, on="movieId")
print(Data.tail())

m = Data['vote_count'].quantile(0.9)
C = Data['vote_average'].mean()

d = Data['revenue'].mean()
p = Data["popularity"].quantile(0.9)


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']

    s = x["revenue"]
    t = x["popularity"]

    z = (v/(v+m) * R) + (m/(m+v) * C)
    y = (s/(s+d) * t) + (d/(d+s) * p)
    return z, y


mov = Data.copy().loc[Data["vote_count"] >= m]
mov["score"] = mov.apply(weighted_rating, axis=1)
print(mov[['title', 'vote_count', 'vote_average', 'score']])






plots = Movies['overview']
tfidf = TfidfVectorizer(stop_words='english', max_df=4, min_df=1)
plots = plots.fillna('')
tfidf_matrix = tfidf.fit_transform(plots)

cos_similar = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cos_similar.shape)

indices = pd.Series(Movies.index, index=Movies['title']).drop_duplicates()


def get_movies(title):
    idx = indices[title]
    similar = list(enumerate(cos_similar[idx]))
    similar = sorted(similar , key = lambda x: x[1] , reverse = True)
    similar = similar[:11]
    indic = []
    for i in similar:
        indic.append(i[0])
    return Movies['title'].iloc[indic]


print(get_movies('Spider-Man 3'))





