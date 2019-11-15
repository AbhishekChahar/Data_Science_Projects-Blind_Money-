#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


boston = datasets.load_boston()
bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos['PRICE'] = boston.target
print(bos.head(7))


# In[21]:


bos.isnull().sum()


# In[24]:


print(bos.describe())


# In[33]:


plt.hist(bos['PRICE'], bins =30)
plt.show()


# In[35]:


X_rooms =bos.RM
Y_price = bos.PRICE
X_rooms = np.array(X_rooms).reshape(-1,1)
Y_price = np.array(Y_price).reshape(-1,1)
print(X_rooms.shape)
print(Y_price.shape)


# In[44]:


X_train_1 ,X_test_1 , Y_train_1, Y_test_1 = train_test_split(X_rooms , Y_price , test_size = 0.2 , random_state =5)
print(X_train_1.shape) 
print(X_test_1.shape)
print(Y_train_1.shape)
print(Y_test_1.shape)


# In[49]:


reg_1 = LinearRegression()
reg_1.fit(X_train_1,Y_train_1)
y_train_predict_1 = reg_1.predict(X_train_1)
rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
r2 = round(reg_1.score(X_train_1,Y_train_1),2)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[53]:


prediction_Space= np.linspace(min(X_rooms),max(X_rooms)).reshape(-1,1)
plt.scatter(X_rooms,Y_price)
plt.plot(prediction_space , reg_1.predict(prediction_space),color = 'black' , linewidth =5)
plt.ylabel('Value of house/1000($)')
plt.xlabel('number of rooms')
plt.show()


# In[4]:


X= bos.drop('PRICE', axis =1)
y=bos['PRICE']
X_train , X_test, y_train , y_test = train_test_split(X ,y , test_size = 0.2 , random_state = 42)
reg_all = LinearRegression()
reg_all.fit(X_train , y_train)

y_train_predict = reg_all.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train , y_train_predict)))
r3 = round(reg_all.score(X_train, y_train),3)
print("The model performance for training set")
print("---------------")
print('RMSE is {}'.format(rmse))
print('R3 score is {}'.format(r3))


# In[6]:


#evaluation for test data

y_pred = reg_all.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_pred , y_test)))
r4 = round( reg_all.score(X_test , y_test),4)
print("Model performance for test data is:  ")
print('RMSE is{} '.format(rmse))
print('r4 score is {}'.format(r4))


# In[8]:


plt.scatter(y_test , y_pred)
plt.xlabel("Actual House Prices($1000)")
plt.ylabel("Predicted House Prices: ($1000)")
plt.xticks(range(0,int(max(y_test)),4))
plt.yticks(range(0,int(max(y_test)),4))
plt.title("Actual Prices vs predicted Prices")


# In[ ]:




