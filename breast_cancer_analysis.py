#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Analysis of features (measurement of cell nuclei) to predict whether the breast cancer tissue is malignant(M) or benign(B)

#Importing all libraries for data analysis and machine learning
import numpy as np #math
import pandas as pd #processing and manipulating data
import matplotlib.pyplot as plt #plotting the data (graph)
import seaborn as sns #plot data (interactive graph)
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split # split data into train and test sets
from sklearn.linear_model import LogisticRegression # Logistic regression
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier
from sklearn.neighbors import KNeighborsClassifier #K-Neighbors classifier
from sklearn.tree import DecisionTreeClassifier #Decision Tree classifier
from sklearn.svm import SVC # Support Vector Machines classifier
from sklearn import metrics # to check for error and accuracy of the model


# In[2]:


#Import the dataset
#Dataset is publically available and is downloaded from the UCI Machine Learning Repository

data = pd.read_csv("breastcancer_diagnostic/data.csv", header = 0)


# In[3]:


#Data Exploration

data.head()


# In[4]:


data.info()


# In[5]:


#Dropping 'Unnamed' and 'id' columns 

data = data.drop("Unnamed: 32", axis=1)
data = data.drop("id", axis=1)


# In[6]:


#Descriptive Statistics

data.describe()


# In[7]:


#Independent variables
x = data.iloc[:, 2:31].values
#Dependent variables
y= data['diagnosis'].values


# In[8]:


#The variable, 'diagnosis' is a class label, so convert it to integer

#data_diagnosis =data['diagnosis'].map({'M':1,'B':0})
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)


# In[9]:


#Determining total number of malignant and benign cells

ct = sns.countplot(data['diagnosis'], label = "Count")
B, M = data['diagnosis'].value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)


# In[10]:


#Visualization of the dataset
#Data distribution of the different features 

data.groupby('diagnosis').size()
data.groupby('diagnosis').hist(figsize=(15, 15))


# In[11]:


#Correlation - exploring the relationship between the features 

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[12]:


#Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[13]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


#Create Function to fit and predict model

def fitPredictModel(model):
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    print(metrics.accuracy_score(prediction,Y_test))
    
    #Generate the confusion matrix and prediction accuracy
    cm = metrics.confusion_matrix(Y_test,prediction)
    
    #Visualize the confusion matrix
    #sns.heatmap(cm,annot=True,fmt="d")
    


# In[15]:


#Support Vector Machines Classifier

SVC_model = SVC(kernel = 'rbf', random_state = 0)
fitPredictModel(SVC_model)


# In[16]:


#Random Forest Classifier

RF_model=RandomForestClassifier(n_estimators=100)
fitPredictModel(RF_model)


# In[17]:


#Logistic Regression Classifier

LR_model = LogisticRegression(random_state = 0)
fitPredictModel(LR_model)


# In[18]:


#K-Nearest Neighbors Classifier

KNN_classifier = KNeighborsClassifier(n_neighbors = 10)
fitPredictModel(KNN_classifier)


# In[19]:


#Using ANN (Artificial Neural Network) to predict Heart Disease or not

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[20]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)


# In[21]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = metrics.confusion_matrix(Y_test, y_pred)
print(metrics.accuracy_score(y_pred,Y_test))

#Visualize the confusion matrix
sns.heatmap(cm,annot=True,fmt="d")


# In[ ]:




