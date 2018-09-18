import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.model_selection.validation import cross_validate
from surprise import accuracy
from surprise.model_selection import KFold

ratings = pd.read_csv('ratings.csv')
movie = pd.read_csv('movies.csv')
df = pd.DataFrame({'itemID': list(ratings.movieId), 'userID': list(ratings.userId), 'rating': list(ratings.rating)})
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

print("====================================Sparsity of movie rating dataset=========================================================")
num_movies = len(set(ratings['movieId']))
num_users = len(set(ratings['userId']))
sparsity = len(ratings) / (num_movies * num_users)

print("====================================Histogram=============================================================")
plt.hist(ratings['rating'],bins = np.arange(0.25,5.5,0.5),color = '#58CCF2')

print("====================================Distribution of Ratings Among Movies=====================================================")
c1 = Counter(ratings['movieId'])
plt.bar(np.arange(num_movies),sorted(c1.values(),reverse = True))
plt.xticks([], [])
plt.title('Distribution of ratings among movies')
plt.ylabel('Number of ratings')
plt.xlabel('Movies')

print("===================================Distribution of Ratings Among Actors=====================================================")
c2 = Counter(ratings['userId'])
plt.bar(np.arange(num_users),sorted(c2.values(),reverse = True))
plt.xticks([], [])
plt.title('Distribution of ratings among users')
plt.ylabel('Number of ratings')
plt.xlabel('Users')

print("=========================Distribution of variance of ratings===================================================================")
var = ratings.groupby('movieId')['rating'].var().fillna(0).tolist()
plt.hist(var, bins = np.arange(0,11,0.5))
plt.xlabel('Variance of ratings')
plt.ylabel('Number of movies')
plt.title('Distribution of variance of ratings')
