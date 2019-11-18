#!/usr/bin/env python
# coding: utf-8

# In[39]:


from sklearn import datasets 
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

boba = datasets.load_breast_cancer()

print(boba.data.shape)
train_data , test_data , train_target , test_target = train_test_split( boba.data , boba.target , 
                                                                       test_size =0.33 , random_state = 42 )
print(train_data.shape)

sexy = GaussianNB()
sexy.fit( train_data , train_target )
print( sexy.predict(train_data) )
print( sexy.score( train_data , train_target))

predicted = sexy.predict(boba.data)
predicted1 =sexy.predict(test_data)
print(sexy.score(test_data, test_target))
print(test_target)
print(confusion_matrix( test_target, predicted1))
print(confusion_matrix( boba.target, predicted))


# In[ ]:




