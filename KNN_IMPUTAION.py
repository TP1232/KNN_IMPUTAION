# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

# =============================================================================
# CSV file Read
# =============================================================================

df = pd.read_csv('data/data_csv.csv')

# =============================================================================
# Min-Max Noramlization function
# =============================================================================
def normalize(x):

##        normalization function for min max

    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

# =============================================================================

# =============================================================================
### Data Preprocessing 


col = df.columns 
df.drop([col[-1]],axis=1)    #removing unwanted columns
new = df[col[0:29]]
columns = new.columns
print(columns)
data_scaled =  normalize(new[columns[0:][:]])

#data_scaled = pd.concat([data_scaled,df[columns[-1]]],axis=1)

# df.plot(columns[2],columns[3:])
# data_scaled.plot(columns[2],columns[3:])

plt.show()

data_scaled.to_csv(r'C:\Users\Rajkp\Desktop\NLP University\big_data_uni\as1\knn-for-dummies\knn-for-dummies\data_scaled.csv', index = False)

# =============================================================================



# =============================================================================
#  Creating Artificial Missingness
# =============================================================================
random_scaled = data_scaled.sample(frac=0.50)
random_scaled_copy = random_scaled

sample_arr = np.random.choice([True, False], size=(284,29)) #creted a same shape of masking array 

final_random = random_scaled_copy.mask(sample_arr,np.NaN)
print(final_random.isna().sum().sum(),"Number of NaN instance in dataset")
print(np.sum(final_random.count()),"Number of non-NaN instance in dataset")

original_data = data_scaled[~data_scaled.index.isin(final_random.index)]  #data which is not selected during creating missingness

final_merge_data = pd.concat([original_data,final_random],axis = 0).sort_index()    # final data for performing knn

data = final_merge_data.values  #

# =============================================================================
# KNN distance Functions
# =============================================================================


def nan_distance(x,y):
    col,non_nan_elements,total_element = len(x),len(x),len(x)
    dist = 0
    
    
    for i in range(col):
        if(np.isnan(x[i]) or np.isnan(y[i])):
        # if(np.isnan(x[i]).all() or np.isnan(y[i])).all():
            non_nan_elements -= 1
            continue
        dist = dist + (x[i]-y[i])**2
        
    if non_nan_elements != 0:
        dist = dist*total_element/non_nan_elements
        
    return dist**0.5
# =============================================================================



# =============================================================================
# KNN imputation function with ecudian distance
# =============================================================================
def knn(data,n_neighbors = 2):
    count = 0
    
    nan_matrix = np.argwhere(np.isnan(data))
    n_row,n_col = data.shape
    
    final_data = data.copy()
    for index in nan_matrix:
        print(count)
        count += 1 
        distances = []
        row,col = index
        
        x = data[row]
        
        for i in range(n_row):
            if(i==row):
                continue
            
            dist = nan_distance(x,data[i])
            distances.append([dist, data[i][col]])
        
            
        distances.sort(key=lambda x: x[0])
        
        distances = np.array(distances)
        neighbors = []
        for i in range(len(distances)):
            if(np.isnan(distances[i,1])):   
                continue
            neighbors.append(distances[i,1])
            if(len(neighbors)==n_neighbors):  
                break
            
        adjusted_value = np.mean(neighbors)
        final_data[row][col] = adjusted_value
    
    return(final_data)

# =============================================================================
# knn weigted function for imputation
# =============================================================================

def inverseweight(dist, num = 1.0, const = 0.1):
    return num / (dist + const)


def create_weights(n_neighbors, distances):
  result = np.zeros(n_neighbors, dtype=np.float32)
  sum = 0.0
  for i in range(n_neighbors):
    result[i] += 1.0 / distances[i]
    sum += result[i]
  result /= sum
  return result


# =============================================================================


# =============================================================================

# =============================================================================
# calculate imputation using mean
# =============================================================================

mean_data = final_merge_data.fillna(final_merge_data.mean())   #mean coputed dataset

# =============================================================================








# =============================================================================
# Results datasets
# =============================================================================
final_1 = knn(data,1)    # data imputation for 1 neighbour knn dataset
final_3 = knn(data,3)    # data imputation for 3 neighbour knn dataset
final_5 = knn(data,5)    # data imputation for 5 neighbour knn dataset

# =============================================================================
#Note: for weighted knn coment upper 3 line and uncomment below 3 line
#      and run through the end line

# final_1 = knn_2(data,1)    # data imputation for 1 neighbour knn weighted dataset
# final_3 = knn_2(data,3)    # data imputation for 3 neighbour knn weighted dataset
# final_5 = knn_2(data,5)    # data imputation for 5 neighbour knn weighted dataset
# 
# =============================================================================




# ######## creating individual columns from dataset
nan_matrix = np.argwhere(np.isnan(data)) # matrix of NaN values and indexs of NaN

# =============================================================================
#  creating arrays for implementations 


# creating array from nan matrix which has index of all missing values 


final_pred_1 = []
final_pred_3 = []
final_pred_5 = []
original_pred = []
mean_pred = []

for i in range(len(nan_matrix)):
    
    
    final_pred_1.append(final_1[nan_matrix[i][0]][nan_matrix[i][1]])
    final_pred_3.append(final_3[nan_matrix[i][0]][nan_matrix[i][1]])
    final_pred_5.append(final_5[nan_matrix[i][0]][nan_matrix[i][1]])
    original_pred.append(data_scaled.values[nan_matrix[i][0]][nan_matrix[i][1]])
    mean_pred.append(mean_data.values[nan_matrix[i][0]][nan_matrix[i][1]])

final_pred_1 = np.array([final_pred_1])     #final knn 1 neighbor predicted values in columns
final_pred_3 = np.array([final_pred_3])     #final knn 3 neighbor predicted values in columns    
final_pred_5 = np.array([final_pred_5])     #final knn 5 neighbor predicted values in columns
original_pred = np.array([original_pred])   #original values which is replce by nan
mean_pred = np.array([mean_pred])           #mean predicted values

ar= np.array([np.arange(len(nan_matrix))])

# =============================================================================

# co = np.concatenate((ar, final_pred_1)).T
# co_3 = np.concatenate((ar, final_pred_3)).T
# co_5 = np.concatenate((ar, final_pred_5)).T
# plt.scatter(co,co_3)
# plt.plot(co_3)
# plt.plot(co_5)


# =============================================================================
# MSE Calculation
# =============================================================================

# For Knn 1 neighbour
MSE_knn_1 = np.square(np.subtract(original_pred,final_pred_1)).mean() 
MSE_knn_3 = np.square(np.subtract(original_pred,final_pred_3)).mean() 
MSE_knn_5 = np.square(np.subtract(original_pred,final_pred_5)).mean()
 
MSE_Mean = np.square(np.subtract(original_pred,mean_pred)).mean() 

MSE_wknn_1 = np.square(np.subtract(original_pred,final_pred_1).T).mean() 
MSE_wknn_3 = np.square(np.subtract(original_pred,final_pred_3).T).mean() 
MSE_wknn_5 = np.square(np.subtract(original_pred,final_pred_5).T).mean()