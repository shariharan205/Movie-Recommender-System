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
