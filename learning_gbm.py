# -*- coding: utf-8 -*-
import pandas as pd
pd.set_option('display.max_columns', 90)

df = pd.read_csv('train.csv')


# Describe should give a stats about a column, however there are 81 columns in 
# the dataset which makes it hard to find columns with missing values
# Writing a utility function to identify missing columns
# Also have to figure out if the missing values can be imputed or the columns
# need to be dropped as it would be hard to impute
df.describe()
cols = zip(df.columns, df.dtypes)
for name, ty in cols:
    nan_exists = sum(df[name].isnull())
    if nan_exists != 0:
        print('Column (', name, '), type (', ty, '), missing - ', nan_exists)
        values = list(set(list(df[name])))
        if len(values) <25:
            print(values)
            if ty == object:
                df[name] = df[name].fillna('no_info')
        else:
            print('values more than 25')
            print(df[name].describe())
            print(df[name].median())
            
lot_frontage_median = df['LotFrontage'].median()
mas_vnr_area_median =  df['MasVnrArea'].median()
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())      
df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].median())     
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0) 
# After reviewing the output from above check, it would be hard to 
# impute 'Alley', 'PoolQC', 'Fence', 'MiscFeature', so choices are drop the
# columns are set nan to no_info as categorical values are still less than 25

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({'price': df['SalePrice']})
prices.hist()

y = df['SalePrice']
df = df.drop(['SalePrice'], axis=1)
df = df.drop(['Id'], axis=1)
df_train = pd.get_dummies(df)

from sklearn import ensemble
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.33, random_state=77)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


estimators = []
x_train_error = []
x_test_error = []
for i in enumerate(range(1, 21)):
    print(i[1])
    estimators.append(i[1] * 100)
    #rfr = RandomForestRegressor(n_estimators=(i[1])*10)
    gb_tree = ensemble.GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=(i[1])*100, min_samples_leaf=10, random_state=77)
    gb_tree.fit(X_train, y_train)
    
    y_hat = gb_tree.predict(X_train)
    err = sqrt(mean_squared_error(y_train, y_hat))
    x_train_error.append(err)
    print(err)
    
    y_hat = gb_tree.predict(X_test)
    err = sqrt(mean_squared_error(y_test, y_hat))
    x_test_error.append(err)
    print(err)
    

fig, ax = plt.subplots()
# ax.plot(estimators, x_train_error, 'o')
ax.plot(estimators, x_test_error, 'x')
plt.show()
print(np.argmin(x_test_error))

# Based on above results will be choosing no of estimators as 260
# Next choose max depth
estimators = []
x_train_error = []
x_test_error = []
for i in enumerate (range(1, 20)):
    print(i[1])
    estimators.append(i[1] * 1)
    # rfr = RandomForestRegressor(loss='huber', learning_rate=0.05, n_estimators=130, max_depth=(i[1])*5)
    gb_tree = ensemble.GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=500, max_depth=(i[1])*1, min_samples_leaf=10, random_state=77 )
    gb_tree.fit(X_train, y_train)
    
    y_hat = gb_tree.predict(X_train)
    err = sqrt(mean_squared_error(y_train, y_hat))
    x_train_error.append(err)
    print(err)
    
    y_hat = gb_tree.predict(X_test)
    err = sqrt(mean_squared_error(y_test, y_hat))
    x_test_error.append(err)
    print(err)
    

fig, ax = plt.subplots()
# ax.plot(estimators, x_train_error, 'o')
ax.plot(estimators, x_test_error, 'x')
plt.show()
print(np.argmin(x_test_error))


gb_tree = ensemble.GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=500, max_depth=3, min_samples_leaf=10, max_features=2 )
# rfr = RandomForestRegressor(n_estimators=80, max_depth=20) # this decreased the score on leadership board
# rfr = RandomForestRegressor(n_estimators=30, max_depth=90)
gb_tree.fit(df_train, y)
len(gb_tree.feature_importances_)
feat_imp = zip(df_train.columns, gb_tree.feature_importances_)
for col, weight in feat_imp:
    print(col, '-', weight)
    
    
df_test = pd.read_csv('test.csv')
df_test = df_test.drop(['Id'], axis=1)
cols = zip(df_test.columns, df_test.dtypes)
for name, ty in cols:
    nan_exists = sum(df_test[name].isnull())
    if nan_exists != 0:
        print('Column (', name, '), type (', ty, '), missing - ', nan_exists)
        values = list(set(list(df_test[name])))
        if len(values) <25:
            print(values)
            print(list(set(list(df[name]))))
            
            if ty == object:
                df_test[name] = df_test[name].fillna('no_info')
                values_missing = list(set(list(df_test[name])) - set(list(df[name])))
                if len(values_missing) != 0:
                    print('misssing in training', values_missing)
        else:
            print('values more than 25')
            print(df_test[name].describe())
            # print(df[name].median())

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(lot_frontage_median)
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(mas_vnr_area_median)
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(0)
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(0)
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(0)
df_test['GarageArea'] = df_test['GarageArea'].fillna(0)
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(0)
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(0)
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(0)
df_test['GarageCars'] = df_test['GarageCars'].fillna(0)

df_test_converted = pd.get_dummies(df_test)

for col in df_test_converted.columns:
    if sum(df_test_converted[col].isnull()) != 0:
        print(col)
        
# Columns that are missiing in the test set
for col in list(set(df_train.columns) - set(df_test_converted.columns)):
    df_test_converted[col] = 0
    
# drop extra columns in test as these data is not available for trains
for col in list(set(df_test_converted.columns) - set(df_train.columns)):
    print('column being dropped', col)
    df_test_converted = df_test_converted.drop([col], axis=1)

predictions = gb_tree.predict(df_test_converted)
lst = list(range(1461, 2920))
output = pd.DataFrame(lst, columns=['Id'])
output['SalePrice'] = predictions

output.to_csv('predictions.csv', index=False)
