
# coding: utf-8

# In[2]:


#Import required packages
import pandas as pd  
import numpy as np 


# In[6]:


import seaborn as sns  
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression


# In[7]:


#read Iris Data Set in Pandas DataFrame
df = pd.read_csv("iris.csv")


# In[9]:


#Print first 5 rows of Iris data set
print (df.head())


# In[11]:


#Drop the first column, which represent nothing but row numbers
df.drop("Unnamed: 0", inplace=True, axis=1)
df


# In[12]:


sns.lmplot("Sepal.Length", "Sepal.Width", data=df, hue="Species", fit_reg=False)


# In[13]:


# view data set for petal length and petal width
sns.lmplot("Petal.Length", "Petal.Width", data=df, hue="Species", fit_reg=False) 


# In[14]:


sns.pairplot(data=df, hue="Species")


# In[24]:


x = np.array(df.drop(["Species"],1))  
y = np.array(df["Species"])  


# In[30]:


#split the data set to train n test set
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)


# In[31]:


classifier = LogisticRegression()


# In[32]:


#train classifier
classifier.fit(x_train, y_train)


# In[33]:


#chk accuracy on test data set
conf = classifier.score(x_test, y_test)


# In[34]:


print(conf)

