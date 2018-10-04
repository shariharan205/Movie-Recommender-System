import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_curve, auc
from surprise import accuracy, Reader, Dataset
from surprise.model_selection import cross_validate, train_test_split, KFold
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
df = pd.DataFrame({'itemID': list(ratings.movieId), 'userID': list(ratings.userId), 'rating': list(ratings.rating)})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

print("====================================Trimming================================================================")
def popular_trim(data):
    movie_id_counter = Counter([val[1] for val in data])
    popular_trimmed_data = [val for val in data if movie_id_counter[val[1]] > 2]
    return popular_trimmed_data


def unpopular_trim(data):
    movie_id_counter = Counter([val[1] for val in data])
    popular_trimmed_data = [val for val in data if movie_id_counter[val[1]] <= 2]
    return popular_trimmed_data


def high_var_trim(data):
    movie_rating_map = defaultdict(list)
    for val in data:
        movie_rating_map[val[1]].append(val[2])

    high_var_data = [val for val in data if
                     len(movie_rating_map[val[1]]) >= 5 and np.var(movie_rating_map[val[1]]) >= 2.0]
    return high_var_data

print("=====================Non-negative Matrix Factorization based filtering=============================================================")
print("Evaluating NNMF colloborative filtering based on #of latent factors vs RMSE and MAE errors on 10folds cross-validation")

k_range = range(2, 51, 2)
avg_rmse, avg_mae = [], []

for k in k_range:
    algo = NMF(n_factors=k)
    cv_result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=False)
    avg_rmse.append(np.mean(cv_result['test_rmse']))
    avg_mae.append(np.mean(cv_result['test_mae']))

plt.plot(k_range, avg_rmse, label="Average RMSE")
plt.plot(k_range, avg_mae, label="Average MAE")
plt.xlabel('Number of latent factors', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.legend()
plt.show()

print("=================================Optimal Number of Latent Factors=============================================================")
all_genres = set('|'.join(movies.genres).split('|'))
print('#of Genres - ', len(all_genres))

print("===================================NNMF collaborative filtering on popular movie trimmed set=================================================")
avg_rmse = []
k_range = range(2, 51, 2)
kf = KFold(n_splits=10)
