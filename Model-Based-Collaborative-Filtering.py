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
print("Evaluating NNMF collaborative filtering based on #of latent factors vs RMSE and MAE errors on 10folds cross-validation")

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

for k in k_range:
    algo = NMF(n_factors=k)

    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(popular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))

    avg_rmse.append(np.mean(k_rmse))

print('Minimum average RMSE is ', min(avg_rmse), ' for k = ', k_range[np.argmin(avg_rmse)])
plt.plot(k_range, avg_rmse)
plt.xlabel('Number of latent factors', fontsize=15)
plt.ylabel('Average RMSE', fontsize=15)
plt.title('#latent factors vs Average RMSE for NMF Popular trimming')
plt.show()

print("========================NNMF collaborative filtering on unpopular movie trimmed set==================================")
avg_rmse = []
k_range = range(2, 51, 2)
kf = KFold(n_splits=10)

for k in k_range:
    algo = NMF(n_factors=k)

    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(unpopular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))

    avg_rmse.append(np.mean(k_rmse))

print('Minimum average RMSE is ', min(avg_rmse), ' for k = ', k_range[np.argmin(avg_rmse)])
plt.plot(k_range, avg_rmse)
plt.xlabel('Number of latent factors', fontsize=15)
plt.ylabel('Average RMSE', fontsize=15)
plt.title('#latent factors vs Average RMSE for NMF Unpopular trimming')
plt.show()

print("=========================NNMF collaborative filtering on high variance movie trimmed set============================")
avg_rmse = []
k_range = range(2, 51, 2)
kf = KFold(n_splits=10)

for k in k_range:

    print(k)
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo = NMF(n_factors=k, reg_bu=0, reg_bi=0, reg_qi=0, reg_pu=0)
        algo.fit(trainset)
        predictions = algo.test(high_var_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))

    avg_rmse.append(np.mean(k_rmse))

print('Minimum average RMSE is ', min(avg_rmse), ' for k = ', k_range[np.argmin(avg_rmse)])
plt.plot(k_range, avg_rmse)
plt.xlabel('Number of latent factors', fontsize=15)
plt.ylabel('Average RMSE', fontsize=15)
plt.title('#latent factors vs Average RMSE for NMF High Variance trimming')
plt.show()

print("=============================ROC Curve for NNMF for different thresholds==============================")
optimal_latent_factors = 20
trainset, testset = train_test_split(data, test_size=.10)

algo = NMF(n_factors=optimal_latent_factors)
algo.fit(trainset)
predictions = algo.test(testset)

actual = [i.r_ui for i in predictions]
predicted = [i.est for i in predictions]

thresholds = [2.5, 3, 3.5, 4]

for val in thresholds:
    print("ROC Curve with threshold = ", val)
    y_test = np.array(actual) >= val
    # y_score = np.array(predicted) >= val
    y_score = predicted
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score)

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, lw=2, label='area under curve = %0.4f' % auc(false_positive_rate,
                                                                                                   true_positive_rate))
    plt.grid(color='0.7', linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc="lower right")
    plt.title('ROC Curve with threshold ' + str(val))
    plt.show()

print("==================================Latent factors and Movie Genre relation========================================")
optimal_latent_factors = 20
algo = NMF(n_factors = optimal_latent_factors)
trainset, _ = train_test_split(data, test_size=0.01)
algo.fit(trainset)

u = algo.pu
v = algo.qi

for col in range(v.shape[1]):
    print('\nColumn number - ', col)
    v_col = v[: ,col]
    top10_movies = v_col.argsort()[-10:][::-1]
    top10_movies_genres = [(movies.genres[i]) for i in top10_movies]
    print(top10_movies_genres)

print("===========================MF with bias collaborative filtering on 10folds cross-validation=====================")
k_range = range(2, 51, 2)
kf = KFold(n_splits=10)

for initmean in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:

    print(initmean)

    avg_rmse, avg_mae = [], []

    for k in k_range:

        k_rmse, k_mae = [], []
        for trainset, testset in kf.split(data):
            algo = SVD(n_factors=k, init_mean=initmean)
            algo.fit(trainset)
            predictions = algo.test(testset)
            k_rmse.append(accuracy.rmse(predictions, verbose=False))
            k_mae.append(accuracy.mae(predictions, verbose=False))

        avg_rmse.append(np.mean(k_rmse))
        avg_mae.append(np.mean(k_mae))