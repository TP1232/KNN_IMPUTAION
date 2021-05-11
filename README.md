# KNN_IMPUTAION

KNN imputation uses k -nearest neighbors in the space of genes to impute missing expression values.

## Imputation

Datasets may have missing values, and this can cause problems for many machine learning algorithms.
As such, it is good practice to identify and replace missing values for each column in your input data prior to modeling your prediction task. This is called missing data imputation, or imputing for short.

## Dataset

[Breast-cancer-Wisconsin](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)

## Usage

```bash
python KNN_Imputaion.py
```
This is a scratch program (Without ML libraries) to generate the artificial missingness and perform an imputation.

- Used total 7 imputation on the dataset.

- Imputation methods: Mean Imputation, KNN and Weighted KNN where K= 1,3,5,7

## Results

MSE Values:

MSE_mean = 0.021112318014536755

MSE_knn_1 = 0.01855509798203391

MSE_knn_3 =0.011347109502586899

MSE_knn_5 =0.009968529381389283
