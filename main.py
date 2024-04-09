import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from xgboost import XGBRegressor
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
holidays=pd.read_csv('holidays_events.csv')
oil=pd.read_csv('oil.csv')
stores=pd.read_csv('stores.csv')
test_copied=test.copy()
id=test['id']
train=train[['id', 'date', 'store_nbr', 'family',  'onpromotion','sales']]

df=pd.concat([train,test])
def lag_feature(df, lags):
    for i in lags:
        df['sales_lag_'+str(i)] = df.groupby(["store_nbr", "family"])['sales'].shift(i)
    return df

lags=[16,30,90,180,360]
a=lag_feature(df,lags)

#a=pd.merge(a, stores, on='store_nbr',how='left')
oil.fillna(method='bfill',inplace=True)
a=pd.merge(a,oil,on='date',how='left')
#a['dcoilwtico'].fillna(method='bfill',inplace=True)
a['dcoilwtico'].fillna(method='bfill',inplace=True)
a['date']=pd.to_datetime(a['date'])
a.set_index(a['date'],inplace=True)
a['day']=a['date'].dt.day
a['weekends']=a['day']>4
le=LabelEncoder()
cat_col=a.select_dtypes('object')
for i in cat_col:
    a[i]=le.fit_transform(a[i])
y=a['sales'].loc[a.index<='2017-06-16']
print()
val_y=a['sales'].loc[(a['date'] > '2017-06-16') & (a['date'] < '2017-08-16')]
a.drop(columns=['sales'],inplace=True)
scaler=StandardScaler()
col=a.columns.to_list()
col.remove('date')
print(col)
for i in col:
    a[i]=scaler.fit_transform(a[i].to_frame())
print(a)
a.drop(columns=['id'],inplace=True)
x=a.loc[a.index<='2017-06-16']
test=a.loc[a.index>='2017-08-16']
val_x=a.loc[(a['date'] > '2017-06-16') & (a['date'] < '2017-08-16')]
print(x.info(),val_x.info())
x.drop(columns=['date'],inplace=True)
val_x.drop(columns=['date'],inplace=True)
test.drop(columns=['date'],inplace=True)

model = XGBRegressor(
    max_depth=8,
    n_estimators=200,
    min_child_weight=300,
    colsample_bytree=0.75,
    eta=0.3,
    seed=42
    )

model.fit(
    x,
    y,
    eval_metric="rmse",
    eval_set=[(x, y),[val_x,val_y]],
    verbose=True,
    early_stopping_rounds = 10)
y=pd.DataFrame(model.predict(test),columns=['sales'])
id=pd.DataFrame(id)

id=id.join(y)
print(id)
id.to_csv(r'C:\Users\alesha\Desktop\stock.csv',index=False)