import pandas as pd
pd.set_option('display.max_columns', 90)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

train = pd.read_csv('train.csv')


train.describe()
train_ID = train['Id']
train.drop('Id', axis=1, inplace=True)

sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


y = train.SalePrice.values
train.drop(['SalePrice'], axis=1, inplace=True)

lot_frontage_median = train['LotFrontage'].median()
mas_vnr_area_median =  train['MasVnrArea'].median()

# Missing data 
cols = zip(train.columns, train.dtypes)
for name, ty in cols:
    nan_exists = sum(train[name].isnull())
    if nan_exists != 0:
        print('Column (', name, '), type (', ty, '), missing - ', nan_exists)
        values = list(set(list(train[name])))
        if len(values) <25:
            print(values)
            if ty == object:
                train[name] = train[name].fillna('no_info')
        else:
            print('values more than 25')
            print(train[name].describe())
            print(train[name].median())
            
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0) 
train['LotFrontage'] = train['LotFrontage'].fillna(lot_frontage_median)
train['MasVnrArea'] = train['MasVnrArea'].fillna(mas_vnr_area_median)

train.describe()

train.head(10)

'''
train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallQual'] = train['OverallQual'].apply(str)
train['OverallCond'] = train['OverallCond'].apply(str)
train['MoSold'] = train['MoSold'].apply(str)
train['YrSold'] = train['YrSold'].apply(str)
'''


df_test = pd.read_csv('test.csv')
test_ID = df_test['Id']
df_test = df_test.drop(['Id'], axis=1)
cols = zip(df_test.columns, df_test.dtypes)
for name, ty in cols:
    nan_exists = sum(df_test[name].isnull())
    if nan_exists != 0:
        print('Column (', name, '), type (', ty, '), missing - ', nan_exists)
        values = list(set(list(df_test[name])))
        if len(values) <25:
            print(values)
            print(list(set(list(train[name]))))
            
            if ty == object:
                df_test[name] = df_test[name].fillna('no_info')
                values_missing = list(set(list(df_test[name])) - set(list(train[name])))
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

#df_test['MSSubClass'] = df_test['MSSubClass'].apply(str)
#df_test['OverallQual'] = df_test['OverallQual'].apply(str)
#df_test['OverallCond'] = df_test['OverallCond'].apply(str)
#df_test['MoSold'] = df_test['MoSold'].apply(str)
#df_test['YrSold'] = df_test['YrSold'].apply(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    # lbl.fit(list(train[c].values)) 
    lst = list(train[c].values)
    lst.extend(list(df_test[c].values))
    lbl.fit(lst) 
    train[c] = lbl.transform(list(train[c].values))
    df_test[c] = lbl.transform(list(df_test[c].values))

# shape        
print('Shape train: {}'.format(train.shape))

df_train = pd.get_dummies(train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.3, random_state=77)
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor

estimators = []
x_train_error = []
x_test_error = []
for i in enumerate (range(1, 20)):
    estimators.append(i[1] * 100)
    xgbr = XGBRegressor(learning_rate=0.05, n_estimators=(i[1])*100, random_state=77, verbose=False )
    xgbr.fit(X_train, y_train)
    
    y_hat = xgbr.predict(X_train)
    err = sqrt(mean_squared_error(y_train, y_hat))
    x_train_error.append(err)
    print(err)
    
    y_hat = xgbr.predict(X_test)
    err = sqrt(mean_squared_error(y_test, y_hat))
    x_test_error.append(err)
    print(err)
    

    fig, ax = plt.subplots()
    ax.plot(estimators, x_train_error, 'o')
    ax.plot(estimators, x_test_error, 'x')
    plt.show()
    print(np.argmin(x_test_error))
    print(estimators[np.argmin(x_test_error)])
    
# Based on above results will be choosing no of estimators as 1000
# Next choose max depth
estimators = []
x_train_error = []
x_test_error = []
for i in enumerate (range(2, 10)):
    estimators.append(i[1] * 1)
    xgbr = XGBRegressor(learning_rate=0.05, n_estimators=700, max_depth=(i[1])*1, random_state=77, verbose=False )
    xgbr.fit(X_train, y_train)
    
    y_hat = xgbr.predict(X_train)
    err = sqrt(mean_squared_error(y_train, y_hat))
    x_train_error.append(err)
    print(err)
    
    y_hat = xgbr.predict(X_test)
    err = sqrt(mean_squared_error(y_test, y_hat))
    x_test_error.append(err)
    print(err)
    

    fig, ax = plt.subplots()
    ax.plot(estimators, x_train_error, 'o')
    ax.plot(estimators, x_test_error, 'x')
    plt.show()
    print(np.argmin(x_test_error))
    print(estimators[np.argmin(x_test_error)])
    

xgbr = XGBRegressor(learning_rate=0.05, n_estimators=700, max_depth=3, random_state=77, verbose=False )
xgbr.fit(df_train, y)
len(xgbr.feature_importances_)
feat_imp = zip(df_train.columns, xgbr.feature_importances_)
for col, weight in feat_imp:
    print(col, '-', weight)
    

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
    
df_test_updated = pd.DataFrame(df_test_converted[df_train.columns[0]], columns=[df_train.columns[0]])
for col in df_train.columns:
    df_test_updated[col] = df_test_converted[col]

predictions = xgbr.predict(df_test_updated)
predictions = np.expm1(predictions)


output = pd.DataFrame(test_ID)
output['SalePrice'] = predictions

output.to_csv('predictions.csv', index=False)
