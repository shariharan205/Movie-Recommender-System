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

print("====================================Average RMSE for popular trim=============================================================")
k_range = range(2,100,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(popular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

print("===================================Average RMSE for unpopular trim============================================================")
k_range = range(2,100,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(unpopular_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

print("================================Average RMSE for high variance trim===========================================================")
k_range = range(2,100,2)
avg_rmse = []
kf = KFold(n_splits=10)
for k in k_range:
    algo = KNNWithMeans(k=k, sim_options = {'name':'pearson'}) 
    k_rmse = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(high_var_trim(testset))
        k_rmse.append(accuracy.rmse(predictions, verbose=False))
    avg_rmse.append(np.mean(k_rmse))
        
plt.plot(k_range, avg_rmse, label = "Average RMSE")
plt.xlabel('Number of neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()
print('The minimum average RMSE is %f for k = %d' %(np.min(avg_rmse),np.argmin(avg_rmse)))

print("====================================ROC Curves=============================================================")
train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)
thresholds = [2.5,3,3.5,4]
algo = KNNWithMeans(k=20, sim_options = {'name':'pearson'}) 
algo.fit(train_set)
predictions = algo.test(test_set)
pred_est = np.array([i.est for i in predictions])
actual_ratings = np.array([i.r_ui for i in predictions])
for threshold in thresholds:
    y_score = pred_est
    y_true = actual_ratings>=threshold
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='Threshold = %0.1f, AUC = %0.4f' % (threshold,roc_auc))
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate')
    plt.legend(loc ='lower right')
    plt.title('Receiver Operating Characteristics (ROC) for threshold = %0.1f' %threshold)
    plt.show()
