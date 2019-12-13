---
title: "XGBoost Classifier"
date: 2019-12-14
tags: [data wrangling, data science, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Machine Learning Engineering, AI, Data Engineering"
mathjax: "true"
---

[Using XGBoost](https://github.com/iamgoddey/staffing_promotion) in predicting staff promotion algorithm.

**XGBoost** is one of the most popular machine learning algorithm these days. Regardless of the type of prediction task at hand; *regression or classification*.

XGBoost is well-known to provide better solutions than other machine learning algorithms. In fact, since its inception, it has become the *"state-of-the-art”* machine learning algorithm to deal with structured data.

**But what makes XGBoost so popular?**

* **Speed and performance:** Originally written in C++, it is comparatively faster than other ensemble classifiers.

+ **Core algorithm is parallelisable:** Because the core XGBoost algorithm is parallelisable it can harness the power of multi-core computers. It is also parallelizable onto GPU’s and across networks of computers making it feasible to train on very large datasets as well.

- **Consistently outperforms other algorithm methods:** It has shown better performance on a variety of machine learning benchmark datasets.

- **Wide variety of tuning parameters:** XGBoost internally has parameters for *cross-validation, regularization, user-defined objective functions, missing values, tree parameters, scikit-learn compatible API* etc.

# Using XGBoost in Python
First of all, just like what you do with any other dataset, you are going to import the dataset and store it in a variable called *"Main_Data"*. To import we use the **Pandas** python package. We import other libraries as we will have to do **Exploratory Data Analysis**

1. Importing library and Reading the dataset:
```python
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score
```
2. Loading of Dataset:
```python
    Main_Data = pd.read_csv('data.csv')
```
3. Performing an [Exploratory Data Analysis] on the Staff Promotion Data set. The Summary of a DataFrame helps to understand the type of variable, data type and presence of null values.
 * Size and Shape of Data:
```python
print('The size of the Train_Riders data is :', Main_Data.size)
print("Dimension: {}".format(Main_Data.shape))
```
 * Summary Statistics:
 ```python
    Stat_of_Main_Data = Numeric_Data.describe(include='all')
    Stat_of_Main_Data = Stat_of_Main_Data.transpose()
 ```
