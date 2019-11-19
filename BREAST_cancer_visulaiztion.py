#!/usr/bin/env python
# coding: utf-8

# In[69]:


from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
breast = datasets.load_breast_cancer()
#print(breast.data.shape)
#print(breast)

#print(breast.target_names)
df = pd.DataFrame(breast.data , columns =  breast.feature_names)
print(df)
df['target'] = breast.target
print(df['target'])
plt.xlabel('feature = mean radius ')
plt.ylabel( 'target')
plt.yticks( ticks = [ 0 , 1 ])
plt.tight_layout()
#plt.xscale('log')
plt.scatter(df['mean radius'] , df['target'] , s = 10 , marker = 'o' , label ='Mean Radius')
plt.scatter(df['mean texture'] , df['target']-0.1 , s = 10, marker = 'X', label = 'mean texture')
plt.legend()
#plt.scatter(df['mean smoothness'] ,df['target'], color = 'c', marker = '*')


# In[ ]:





# In[ ]:




