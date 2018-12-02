import pandas as pd
import numpy as np
import math
from scipy import stats

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

test.drop('id',axis=1,inplace=True)
train.drop('id',axis=1,inplace=True)

train['time']=np.log1p(train['time'])

ntrain=train.shape[0]
ntest=test.shape[0]

trainlabel=train.time.values

allData=pd.concat((train,test)).reset_index(drop=True)
allData.drop(['time'],axis=1,inplace=True)
allData['n_jobs'] = allData['n_jobs'].replace(-1,16)
allData['new']=allData['n_samples']*allData['n_features']/allData['n_jobs']
allData['alpha']=allData['alpha'].apply(lambda x:-math.log(x,10))
allData['n_jobs']=allData['n_jobs'].apply(str)



allData=pd.get_dummies(allData)
for i in list(allData.columns):
    allData[i]=stats.zscore(allData[i])

train=allData[:ntrain]
test=allData[ntrain:]

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb

n_folds =5
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.7, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='poly', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=12, min_samples_split=30, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)

stacked_averaged_models.fit(train.values, trainlabel)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

model_xgb.fit(train, trainlabel)
xgb_pred = np.expm1(model_xgb.predict(test))

model_lgb.fit(train, trainlabel)
lgb_pred = np.expm1(model_lgb.predict(test.values))

result=stacked_pred
#result=stacked_pred*0.7+xgb_pred*0.2+lgb_pred*0.1

for i in range(len(result)):
    if result[i]<0:
        result[i]=0.07

result=pd.DataFrame(result)
result.columns=['time']
result['id']=result.index
result=result[['id','time']]
result.to_csv('ttsubmission.csv',index=0)

