#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-6-10下午4:12'
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

df=pd.read_excel('/media/jerry/JERRY/MASTER_project/us_MACRO.xls').iloc[6:-2,:]
index=df.iloc[:,0]
df.index=pd.DatetimeIndex(index)
a=df['LIBOR:美元:隔夜'][:]
b=df['美国:个人消费支出物价指数:季调'][:].fillna(method='ffill')
c=df['美国:联邦基金利率(日)'][:]
d=df['美国:国债长期平均实际利率'][:]
a=a.fillna(0)
b=b.fillna(0)
c=c.fillna(0)
d=d.fillna(0)
ds=pd.DataFrame()
ds['A'],ds['C'],ds['B']=a,c,b

X_train, X_test, y_train, y_test = train_test_split(
    ds.values, d.values, test_size=0.25, random_state=2)

# 设置gridsearch的参数
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3],
                     'C': [10, 100]},
                    {'kernel': ['linear'], 'C': [100, 1000]}]

#设置模型评估的方法.如果不清楚,可以参考上面的k-fold章节里面的超链接
scores = [ 'neg_mean_squared_error']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #构造这个GridSearch的分类器,5-fold
    clf = GridSearchCV(SVR(), tuned_parameters, cv=2,
                       scoring='%s' % score)
    #只在训练集上面做k-fold,然后返回最优的模型参数
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    #输出最优的模型参数
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    #在测试集上测试最优的模型的泛化能力.
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()





ols=sm.OLS(d,sm.add_constant(ds)).fit()
print(ols.summary())
svr=SVR()
svr.fit(X_train,X_test)
print(svr.score(ds.values,d.values))