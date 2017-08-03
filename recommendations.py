import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# data loading of movie lensdata
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('D:/general_data/ml/u.data', sep='\t', names=header)

# Shape of dataset
print('Shape of Dataset :', df.shape )

# Number of unique users and unique movies
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of Users = '+str(n_users) + '| Number of movies = ' + str(n_items))

#Split train and test set
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

# we will print train_data and train_data_matrix to check how things are working
print(train_data)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

# Outlook of train data matrix
print(train_data_matrix.shape)

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

# prediction for recommendation

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

# printing prediction
print('User based prediction : ' , user_prediction)
from sklearn.metrics import mean_squared_error

from math import sqrt

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

# SVD recommendation

import scipy.sparse as sp
from scipy.sparse.linalg import svds, splu

#get SVD components from train matrix. Choose k(its value should be from 1<=k<min(A.shape)
#where A is train data matrix

x = []
y = []
for i in range(5,100,10):
    #print('##', i ,'##')
    u, s, vt = svds(train_data_matrix, k = i)
    s_diag_matrix=np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    #print('SVD Predictions : ' , X_pred)
    #print ('User-based_CF_MSE: ' + str(rmse(X_pred, test_data_matrix)))
    x.append(rmse(X_pred, test_data_matrix))
    y.append(i)
    #print(x)
#print(y)
# plotting k and error
plt.plot(y,x, 'ro')
plt.xlabel('k')
plt.ylabel('MSE Error')
plt.show()
    

################ Special Thanks to Agnes Johannsdottir ######################
