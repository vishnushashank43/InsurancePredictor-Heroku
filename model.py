# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split 
import pickle
#importing
df = pd.read_csv('insurance.csv')
df.head()
 #variables
X = X = df.drop(columns='charges').values
y = df['charges'].values
#Labelencoding
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])


# train test split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)


# buliding model
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state = 0)


# buliding model
regressor = xgb.XGBRegressor(n_estimators=100)
                    
regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)



pickle.dump(regressor , open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
