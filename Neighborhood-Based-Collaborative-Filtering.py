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

print("====================================KNN Colloborative Filtering=============================================================")
k_range = range(2,100,2)
avg_rmse, avg_mae = [], []
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=False)
    avg_rmse.append(np.mean(cv_results['test_rmse']))
    avg_mae.append(np.mean(cv_results['test_mae']))
    
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.plot(k_range, avg_mae, label = "Average MAE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()

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
    high_var_data = [val for val in data if len(movie_rating_map[val[1]]) >= 5 and np.var(movie_rating_map[val[1]]) >= 2.0]
    return high_var_data