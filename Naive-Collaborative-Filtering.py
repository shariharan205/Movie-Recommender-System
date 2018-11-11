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


k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true, y_pred))
avg_rmse = np.mean(k_rmse)
print('The average RMSE using naive collaborative filter is %0.4f' % avg_rmse)

print("==================================Naive collaborative filter performance on trimmed set====================================================")
print("====================================Popular trimming=============================================================")
k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    testset = popular_trim(testset)
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true, y_pred))
avg_rmse = np.mean(k_rmse)
print('The average RMSE for popular movie trimmed set is %0.4f' % avg_rmse)

print("====================================Unpopular trimming=============================================================")
k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    testset = unpopular_trim(testset)
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true, y_pred))
avg_rmse = np.mean(k_rmse)
print('The average RMSE for unpopular movie trimmed set is %0.4f' % avg_rmse)

print("====================================High Variance Trimming========================================================")
k_rmse = []
kf = KFold(n_splits=10)
for trainset, testset in kf.split(data):
    testset = high_var_trim(testset)
    y_pred = naive_prediction(testset)
    y_true = [testset[i][2] for i in range(len(testset))]
    k_rmse.append(mean_squared_error(y_true, y_pred))
avg_rmse = np.mean(k_rmse)
print('The average RMSE for high variance movie trimmed set is %0.4f' % avg_rmse)

print("==============================Performance Comparison=================================================")
k_range = range(2, 50, 2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = SVD(n_factors=k)
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))

plt.plot(k_range, avg_rmse, label="Average RMSE")
plt.xlabel('Number of latent factors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' % (np.min(avg_rmse), np.argmin(avg_rmse)))

print("====================================Ranking=====================================================")
k_range = range(2, 51, 2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = SVD(n_factors=k)
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(popular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))

plt.plot(k_range, avg_rmse, label="Average RMSE")
plt.xlabel('Number of latent factors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' % (np.min(avg_rmse), np.argmin(avg_rmse)))


def get_top_t(predictions, t=10):
    # First map the predictions to each user.
    top_t = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_t[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_t.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_t[uid] = user_ratings[:t]

    return top_t


train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)
algo = KNNWithMeans(k=20, sim_options={'name': 'pearson'})
algo.fit(train_set)
predictions = algo.test(test_set)
top_recos = get_top_t(predictions)
