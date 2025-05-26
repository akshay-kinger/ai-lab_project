#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp


# In[4]:


df=pd.read_csv(r"C:\Users\hp\Downloads\data.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[9]:


df.describe()


# In[10]:


df['Grade'].value_counts()


# In[12]:


df.hist(column='pH',bins=50)


# In[14]:


df.hist(column='Temprature',bins=50)


# In[27]:


X=df[['pH','Temprature','Taste','Odor','Fat','Turbidity','Color']].values
y=df[['Grade']].values
X[0:5]


# In[31]:


X=pp.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[33]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y =train_test_split(X,y,test_size=0.2,random_state=4)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
k=3
kcls=KNeighborsClassifier(n_neighbors=k)
kcls.fit(train_x,train_y)


# In[39]:


from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

ks = 10
mean_acc = np.zeros((ks - 1))
std_acc = np.zeros((ks - 1))

for n in range(1, ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(train_x, train_y.ravel())
    yhat = neigh.predict(test_x)
    mean_acc[n - 1] = accuracy_score(test_y, yhat)
    std_acc[n - 1] = np.std(yhat == test_y) / np.sqrt(yhat.shape[0])

print("Mean Accuracies:", mean_acc)
print("Standard Deviations:", std_acc)


# In[41]:


plt.plot(range(1,ks),mean_acc,'b--')
plt.fill_between(range(1,ks),mean_acc -1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,ks),mean_acc -3 * std_acc,mean_acc +3 * std_acc,alpha=0.10,color="blue")
plt.legend(('Accuracy','+/-1xstd','+/-3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()

print("the best accuracy was",mean_acc.max(),"with k=",mean_acc.argmax()+1)
                


# In[42]:


train_y_hat = kcls.predict(train_x)
test_y_hat = kcls.predict(test_x)
print('train accuracy score:', accuracy_score(train_y,train_y_hat))
print('test accuracy score:' , accuracy_score(test_y,test_y_hat))


# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold

model = DecisionTreeClassifier()
-x


# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dtc = DecisionTreeClassifier(criterion="entropy", max_depth=10)
dtc.fit(train_x, train_y)
print("DecisionTree's Accuracy:", accuracy_score(test_y, dtc.predict(test_x)))



# In[47]:


model.fit(train_x,train_y)


# In[48]:


model.score(train_x,train_y)


# In[ ]:




