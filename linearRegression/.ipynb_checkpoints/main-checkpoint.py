# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn import preprocessing;
from sklearn.model_selection import train_test_split;
from sklearn import linear_model;
import matplotlib.pyplot as plt


prices_df = pd.read_csv('prices.csv')

close_google = pd.DataFrame(prices_df.low[prices_df.symbol=='GOOG'])
#close_google.plot()



forecast_out = 5 
test_size = 0.2; 
df = close_google

label = df.shift(-forecast_out);
X = np.array(df); 
X = preprocessing.scale(X) 
#plt.plot(X)


X_lately = X[-forecast_out:] 
X = X[:-forecast_out] 
label.dropna(inplace=True)
y = np.array(label) 

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) 
plt.scatter(X_train, Y_train)
plt.show()

learner = linear_model.LinearRegression() 

learner.fit(X_train,Y_train); 
score=learner.score(X_test,Y_test)

forecast = learner.predict(X_lately) 

response = {}
response['test_score'] = score
response['forecast_set'] = forecast

print(response);

