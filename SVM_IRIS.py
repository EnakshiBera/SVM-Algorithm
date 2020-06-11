#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines(SVM)

# ### SVM offers very high accuracy compared to other classifiers such as logistic regression, and decision trees. It is known for its kernel trick to handle nonlinear input spaces. It is used in a variety of applications such as face detection, intrusion detection, classification of emails, news articles and web pages, classification of genes, and handwriting recognition.

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris= load_iris()
iris.feature_names


# In[2]:


df = pd.DataFrame(iris.data, columns= iris.feature_names)
df.head()


# In[3]:


df['target']= iris.target
df.head()


# In[4]:


iris.target_names


# In[5]:


df.dtypes


# In[6]:


# 0 means setosa
df[df.target==0].head()


# In[7]:


# 1 means versicolor
df[df.target==1].head()


# In[8]:


# 2 means virginica
df[df.target==2].head()


# In[9]:


df['flower_names'] = df.target.apply(lambda x: iris.target_names[x])


# In[10]:


df.head()


# In[11]:


df[df.target==1].head()


# ## Visualizing the Data

# In[12]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]


# In[14]:


plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='g', marker='*')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='r', marker='.')


# In[15]:


plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='g', marker='*')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='r', marker='.')


# ## Splitting Data

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x=df.drop(['target', 'flower_names'], axis='columns')
x.head()


# In[18]:


y=df.target
y.head()


# In[19]:


x_train,x_test, y_train, y_test  = train_test_split(x,y,train_size=0.8)


# In[20]:


len(x_train)


# In[21]:


len(x_test)


# # TRAINING  THE ALGORITHM

# # Model without Kernel.

# In[30]:


# Fitting SVC Classification to the Training set 
from sklearn.svm import SVC
svc= SVC()
svc.fit(x_train, y_train)

#Predict the response fot test dataset.
y_pred=svc.predict(x_test)
print(y_pred)

# Evaluating the Model
# Accuracy can be computed by comparing actual test set values and predicted values.
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
svc.score(x_test, y_test)
svc.score(x_test, y_test)


# # SVM KERNELS
# The SVM algorithm is implemented in practice using a kernel. A kernel transforms an input data space into the required form. SVM uses a technique called the kernel trick. Here, the kernel takes a low-dimensional input space and transforms it into a higher dimensional space. In other words, you can say that it converts nonseparable problem to separable problems by adding more dimension to it. It is most useful in non-linear separation problem. Kernel trick helps you to build a more accurate classifier.

# ## 1. Using Linear Kernel

# In[41]:


# Fitting SVC Classification to the Training set with linear kernel
from sklearn.svm import SVC
svc_linear= SVC(kernel='linear', random_state = 0)
svc_linear.fit(x_train, y_train)

#Predict the response fot test dataset.
y_pred=svc_linear.predict(x_test)
print(y_pred)

# Evaluating the Model
# Accuracy can be computed by comparing actual test set values and predicted values.
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# #### The model accuracy_score is 93.3%.

# ## Using Polynomial Kernel.

# In[37]:


# Fitting SVC Classification to the Training set with Polynomial kernel
from sklearn.svm import SVC
svc_polynomial= SVC(kernel = 'poly', random_state = 0)
svc_polynomial.fit(x_train, y_train)

# Predicting the Test set results
y_pred = svc_polynomial.predict(x_test)
print(y_pred)

# Evaluating the Model
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# ## Using RBF Kernel.

# In[38]:


# Fitting SVC Classification to the Training set with linear kernel
from sklearn.svm import SVC
svc_rbf = SVC(kernel = 'rbf', random_state = 0)
svc_rbf.fit(x_train, y_train)

# Predicting the Test set results
y_pred = svc_rbf.predict(x_test)
print(y_pred)

# Evaluating the Model
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)


# ## Using Sigmoid Kernel. 

# In[39]:


# Fitting SVC Classification to the Training set with sigmoid kernel
from sklearn.svm import SVC
svc_sigmoid = SVC(kernel = 'sigmoid', random_state = 0)
svc_sigmoid.fit(x_train, y_train)

# Predicting the Test set results
y_pred = svc_sigmoid.predict(x_test)
print(y_pred)

# Evaluating the Model
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)

