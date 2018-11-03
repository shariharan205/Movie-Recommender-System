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
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import NMF
from sklearn.metrics import mean_squared_error

ratings = pd.read_csv('ratings.csv')
movie = pd.read_csv('movies.csv')
df = pd.DataFrame({'itemID': list(ratings.movieId), 'userID': list(ratings.userId), 'rating': list(ratings.rating)})
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


def popular_trim(data):
    print("Popular trimming")
    movie_id_counter = Counter([val[1] for val in data])
    popular_trimmed_data = [val for val in data if movie_id_counter[val[1]] > 2]
    return popular_trimmed_data


def unpopular_trim(data):
    print("Unpopular trimming")
    movie_id_counter = Counter([val[1] for val in data])
    popular_trimmed_data = [val for val in data if movie_id_counter[val[1]] <= 2]
    return popular_trimmed_data


def high_var_trim(data):
    print("High variance trimming")
    movie_rating_map = defaultdict(list)
    for val in data:
        movie_rating_map[val[1]].append(val[2])
    high_var_data = [val for val in data if
                     len(movie_rating_map[val[1]]) >= 5 and np.var(movie_rating_map[val[1]]) >= 2.0]
    return high_var_data


print("====================================Design and test via cross-validation=============================================================")
avg_rating = df.groupby(['userID'])['rating'].mean().tolist()


def naive_prediction(dataset):
    predictions = [avg_rating[dataset[i][0] - 1] for i in range(len(dataset))]
    return predictions
