import pandas as pd
pd.set_option('display.max_columns', 90)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from scipy.special import boxcox1p

from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.9f}'.format(x)) #Limiting floats output to 3 decimal points

train = pd.read_csv('train.csv')

train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000))]
# train.drop(train['GarageArea'], inplace=True)
train.drop(train[train.TotalBsmtSF>3000].index, inplace=True)
train.drop(train[train.YearBuilt<1900].index, inplace=True)
train.reset_index(drop=True, inplace=True)


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
                train[name] = train[name].fillna('None')
        else:
            print('values more than 25')
            print(train[name].describe())
            print(train[name].median())
            
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0) 
train['LotFrontage'] = train['LotFrontage'].fillna(lot_frontage_median)
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)

train.describe()

train.head(10)

train['MSSubClass'] = train['MSSubClass'].apply(str)
# train['OverallQual'] = train['OverallQual'].apply(str)
# train['OverallCond'] = train['OverallCond'].apply(str)
# train['BsmtFullBath'] = train['BsmtFullBath'].apply(str)
# train['BsmtHalfBath'] = train['BsmtHalfBath'].apply(str)
# train['FullBath'] = train['FullBath'].apply(str)
# train['HalfBath'] = train['HalfBath'].apply(str)
train['MoSold'] = train['MoSold'].apply(str)
train['YrSold'] = train['YrSold'].apply(str)
# train['BedroomAbvGr'] = train['BedroomAbvGr'].apply(str)
# train['KitchenAbvGr'] = train['KitchenAbvGr'].apply(str)
# train['Fireplaces'] = train['Fireplaces'].apply(str)

print(train['SaleType'].mode()[0])


df_test = pd.read_csv('test.csv')
test_ID = df_test['Id']
df_test = df_test.drop(['Id'], axis=1)
df_test['BsmtQual'] = df_test['BsmtQual'].fillna('None')
df_test['BsmtCond'] = df_test['BsmtCond'].fillna('None')
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna('None')
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna('None')
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna('None')
df_test['GarageType'] = df_test['GarageType'].fillna('None')
df_test['GarageFinish'] = df_test['GarageFinish'].fillna('None')
df_test['GarageQual'] = df_test['GarageQual'].fillna('None')
df_test['GarageCond'] = df_test['GarageCond'].fillna('None')
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
                print('filling nan with ', train[name].mode()[0])
                df_test[name] = df_test[name].fillna(train[name].mode()[0])
                values_missing = list(set(list(df_test[name])) - set(list(train[name])))
                if len(values_missing) != 0:
                    print('misssing in training', values_missing)
        else:
            print('values more than 25')
            print(df_test[name].describe())
            # print(df[name].median())

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(lot_frontage_median)
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)
df_test['BsmtFinSF1'] = df_test['BsmtFinSF1'].fillna(0)
df_test['BsmtFinSF2'] = df_test['BsmtFinSF2'].fillna(0)
df_test['TotalBsmtSF'] = df_test['TotalBsmtSF'].fillna(0)
df_test['BsmtUnfSF'] = df_test['BsmtUnfSF'].fillna(0)
df_test['GarageArea'] = df_test['GarageArea'].fillna(0)
df_test['BsmtFullBath'] = df_test['BsmtFullBath'].fillna(0)
df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(0)
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(0)
df_test['GarageCars'] = df_test['GarageCars'].fillna(0)
df_test = df_test.replace({'GarageCars': 5}, 4)
# df_test['SaleType'] = df_test['SaleType'].fillna(train['SaleType'].mode()[0])
# df_test['MSZoning'] = df_test['MSZoning'].fillna(train['MSZoning'].mode()[0])
# df_test['Utilities'] = df_test['Utilities'].fillna(train['Utilities'].mode()[0])

df_test['MSSubClass'] = df_test['MSSubClass'].apply(str)
# df_test['OverallQual'] = df_test['OverallQual'].apply(str)
# df_test['OverallCond'] = df_test['OverallCond'].apply(str)
# df_test['BsmtFullBath'] = df_test['BsmtFullBath'].astype(float).astype(int).apply(str)
# df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].astype(float).astype(int).apply(str)
# df_test['FullBath'] = df_test['FullBath'].apply(str)
# df_test['HalfBath'] = df_test['HalfBath'].apply(str)
df_test['MoSold'] = df_test['MoSold'].apply(str)
df_test['YrSold'] = df_test['YrSold'].apply(str)
# df_test['BedroomAbvGr'] = df_test['BedroomAbvGr'].apply(str)
# df_test['KitchenAbvGr'] = df_test['KitchenAbvGr'].apply(str)
# df_test['Fireplaces'] = df_test['Fireplaces'].apply(str)


numeric_types = [] 
for c  in train.dtypes.index:
    if train.dtypes[c] != object:
        numeric_types.append(c)
        
for n_type in numeric_types:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    # print(axes)
    axes[0].set_title(n_type + ' distribution')
    # axes[0].set_legend('Normal dist. ($\mu=$ {:.2f}, $\sigma=$ {:.2f} and $\skew={:.2f} )'.format(mu, sigma, skewed_feats), loc='best')
    
    # Get the fitted parameters used by the function
    skewed_feats = skew(train[n_type]) #compute skewness
    sns.distplot(train[n_type],fit=norm, ax=axes[0], label='Normal dist. ($\mu=$ {:.2f}, $\sigma=$ {:.2f} and $\skew={:.2f} )'.format(mu, sigma, skewed_feats))
    # print('skew = {:.2f}'.format(skewed_feats))
    (mu, sigma) = norm.fit(train[n_type])
    print( '\n normal' + n_type + ' mu = {:.2f}, skew={:.2f} and sigma = {:.2f}\n'.format(mu, skewed_feats, sigma))
    
    #Now plot the distribution
    # plt.legend(['Normal dist. ($\mu=$ {:.2f}, $\sigma=$ {:.2f} and $\skew={:.2f} )'.format(mu, sigma, skewed_feats)],
    #             loc='best')
    # plt.ylabel('Frequency')
    # plt.title(n_type + ' distribution')
    
    lam = 0.15
    if abs(skewed_feats) > 0.5:
        # skewed_feats_log1p = skew(np.log1p(train[n_type]))
        skewed_feats_log1p = skew(boxcox1p(train[n_type], lam))
        if abs(skewed_feats_log1p) < abs(skewed_feats):
            train[n_type] = boxcox1p(train[n_type], lam)
            axes[1].set_title(n_type + ' log1p distribution')
            # axes[1].set_legend('Normal dist. ($\mu=$ {:.2f}, $\sigma=$ {:.2f} and $\skew={:.2f} )'.format(mu, sigma, skewed_feats), loc='best')
            df_test[n_type] = boxcox1p(df_test[n_type], lam)
            
            skewed_feats = skew(train[n_type]) #compute skewness
            # print('skew = {:.2f}'.format(skewed_feats))
            (mu, sigma) = norm.fit(train[n_type])
            print( '\n log1p' + n_type + ' mu = {:.2f}, skew={:.2f} and sigma = {:.2f}\n'.format(mu, skewed_feats, sigma))
            axes[1] = sns.distplot(train[n_type], fit=norm, ax=axes[1])
            
            # plt.legend(['Normal dist. ($\mu=$ {:.2f}, $\sigma=$ {:.2f} and $\skew={:.2f})'.format(mu, sigma, skewed_feats)], loc='best')
            plt.ylabel('Frequency')
            plt.title(n_type + ' distribution')
        
    # fig.tight_layout()
    plt.show()
    
'''
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'BedroomAbvGr', 'BedroomAbvGr', 'Fireplaces')
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
'''
train.LotShape.replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)
train.LandContour.replace(to_replace = ['Low', 'Bnk', 'HLS', 'Lvl'], value = [0, 1, 2, 3], inplace = True)
train.Utilities.replace(to_replace = ['NoSeWa', 'AllPub'], value = [0, 1], inplace = True)
train.LandSlope.replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)
train.ExterQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
train.ExterCond.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
train.BsmtQual.replace(to_replace = ['None', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
train.BsmtCond.replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
train.BsmtExposure.replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
train.BsmtFinType1.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
train.BsmtFinType2.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
train.HeatingQC.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
train.Electrical.replace(to_replace = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)
train.KitchenQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
train.Functional.replace(to_replace = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
train.FireplaceQu.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
train.GarageFinish.replace(to_replace =  ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)
train.GarageQual.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
train.GarageCond.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
train.PavedDrive.replace(to_replace =  ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)
train.PoolQC.replace(to_replace =  ['None', 'Fa', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
train.Fence.replace(to_replace =  ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)

df_test.LotShape.replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)
df_test.LandContour.replace(to_replace = ['Low', 'Bnk', 'HLS', 'Lvl'], value = [0, 1, 2, 3], inplace = True)
df_test.Utilities.replace(to_replace = ['NoSeWa', 'AllPub'], value = [0, 1], inplace = True)
df_test.LandSlope.replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)
df_test.ExterQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
df_test.ExterCond.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df_test.BsmtQual.replace(to_replace = ['None', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df_test.BsmtCond.replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
df_test.BsmtExposure.replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
df_test.BsmtFinType1.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
df_test.BsmtFinType2.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
df_test.HeatingQC.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
df_test.Electrical.replace(to_replace = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)
df_test.KitchenQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
df_test.Functional.replace(to_replace = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
df_test.FireplaceQu.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df_test.GarageFinish.replace(to_replace =  ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)
df_test.GarageQual.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df_test.GarageCond.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
df_test.PavedDrive.replace(to_replace =  ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)
df_test.PoolQC.replace(to_replace =  ['None', 'Fa', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
df_test.Fence.replace(to_replace =  ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)

df_train = pd.get_dummies(train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y, test_size=0.3, random_state=77)
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

xgb1 = XGBRegressor()
parameters = {'nthread':[1], #when use hyperthread, xgboost may become slower
              # 'objective':['reg:squarederror'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [3],
              'min_child_weight': [1.5, 1.6, 1.7],
              'silent': [0],
              'subsample': [0.5, 0.6],
              'gamma': [0.04, 0.05],
              'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
              'reg_alpha': [0.4, 0.45, 0.5],
              'reg_lambda': [0.8, 0.87, 0.94, 1],
              'seed': [77],
              'n_estimators': range(2000, 2501, 100)}

fit_params = {'early_stopping_rounds': 42,
              'eval_metric': 'rmse',
              'eval_set': [(X_train, y_train), (X_test, y_test)]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs = 4,
                        scoring='r2',
                        verbose=0)

xgb_grid.fit(X_train,y_train, **fit_params)
# changing to df_train, y because of CV
# xgb_grid.fit(df_train, y)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
print(xgb_grid.best_estimator_)

'''
estimators = []
x_train_error = []
x_test_error = []
for i in enumerate (range(4, 30)):
    estimators.append(i[1] * 100)
    xgbr = XGBRegressor(learning_rate=0.05, n_estimators=(i[1])*100, random_state=77, nthread = -1, max_depth=3, subsample=0.5 )
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
'''

# xgbr = XGBRegressor(learning_rate=0.05, n_estimators=1000, max_depth=3, random_state=77, subsample=0.5, verbose=True)

# 0.9067720490811773 {'colsample_bytree': 0.4, 'gamma': 0.04, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 2400, 'nthread': -1, 'silent': 1, 'subsample': 0.5}
# xgbr = XGBRegressor(colsample_bytree= 0.7, learning_rate= 0.05, max_depth= 3, n_estimators= 1400, nthread= -1, objective= 'reg:squarederror', subsample= 0.7, verbose=10)


'''
xgbr = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, verbosity=1,
                             random_state =77, nthread = -1)


xgbr = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.04, learning_rate=0.05,
       max_delta_step=0, max_depth=3, min_child_weight=1.5, missing=None,
       n_estimators=2000, n_jobs=1, nthread=1, objective='reg:linear',
       random_state=0, reg_alpha=0.4, reg_lambda=0.8, scale_pos_weight=1,
       seed=77, silent=1, subsample=0.5)
'''
xgbr = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.6, gamma=0.04, learning_rate=0.05,
       max_delta_step=0, max_depth=3, min_child_weight=1.5, missing=None,
       n_estimators=2000, n_jobs=1, nthread=1, objective='reg:linear',
       random_state=0, reg_alpha=0.45, reg_lambda=0.86, scale_pos_weight=1,
       seed=77, silent=0, subsample=0.6)
xgbr.fit(df_train, y)
len(xgbr.feature_importances_)
feat_imp = zip(df_train.columns, xgbr.feature_importances_)
for col, weight in feat_imp:
    print(col, '-', weight)
    
import xgboost as xgb
xgb.plot_importance(xgbr, max_num_features=20)
    

df_test_converted = pd.get_dummies(df_test)

for col in df_test_converted.columns:
    if sum(df_test_converted[col].isnull()) != 0:
        print(col)
        
# Columns that are missiing in the test set
for col in list(set(df_train.columns) - set(df_test_converted.columns)):
    print('adding', col)
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

predictions = np.ceil(predictions/1000) * 1000
output['SalePrice'] = predictions
output.to_csv('predictions_rounded.csv', index=False)
