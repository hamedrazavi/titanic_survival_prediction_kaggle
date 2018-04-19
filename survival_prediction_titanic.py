
# coding: utf-8

# In[577]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[578]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.tail()


# # Cleaning
# ## The columns which are not numeric will be converted to numeric and the columns with too many NaNs will be removed (e.g., the Cabin)

# In[579]:


print("Total number of samples in train set is:", len(train))
print("-"*45)
print("The number of null (NaN) values in each column of the train set is:")
print(train.isnull().sum())


# ### From the above cell output, the cabin column has too many NaN values, so we will remove it. Also, we will replace the other NaN values in the Age and Embarked columns with their average values (or most frequent values in the case of discrete distribution)

# In[580]:


trData = train.drop('Cabin', 1)
# Name and Ticket number are also dropped. Name is irelevant and Ticket number is unique for each person (i.e. 891 different tickets!)
trData = trData.drop('Name', 1)
trData = trData.drop('Ticket', 1)

testData = test.drop('Cabin', 1)
# Name and Ticket number are also dropped. Name is irelevant and Ticket number is unique for each person (i.e. 891 different tickets!)
testData = testData.drop('Name', 1)
testData = testData.drop('Ticket', 1)


# In[581]:


testData.head()


# In[582]:


trData.head()


# In[583]:


testData[testData['Fare'].isnull()]


# In[584]:


testData['Fare'] = testData['Fare'].fillna(trData['Fare'].mean())
testData.head()


# In[585]:


# Age has a lot of missing data (177 out of 891), we replace the missing ages with the average value
trData['Age']  = trData['Age'].fillna(trData['Age'].mean());
testData['Age']  = testData['Age'].fillna(trData['Age'].mean());


# In[586]:


trData['Embarked'].value_counts()


# In[587]:


trData[trData['Embarked'].isnull()]


# In[588]:


# remove the two rows without Embarked Info
trData = trData.drop(trData.index[[61, 829]]);
len(trData)


# In[589]:


# reset the index
trData = trData.reset_index()


# In[590]:


trData.head()


# In[591]:


trData = trData.drop('index', 1);


# In[592]:


trData['Sex'] = trData['Sex'].replace(['female', 'male'],[0,1])
testData['Sex'] = testData['Sex'].replace(['female', 'male'],[0,1])


# In[593]:


print(trData['Embarked'].unique())
print(testData['Embarked'].unique())


# In[594]:


trData['Embarked'] = trData['Embarked'].replace(['S','C','Q'],[0,1,2])
testData['Embarked'] = testData['Embarked'].replace(['S','C','Q'],[0,1,2])
trData.head()
testData.head()


# In[595]:


trData['Nfamily'] = trData['Parch'] + trData['SibSp']
testData['Nfamily'] = testData['Parch'] + testData['SibSp']


# # Statistical Analysis and feature engineering

# In[596]:


trData[['Nfamily', 'Survived']].groupby(['Nfamily'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[597]:


trData[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[598]:


trData[['Sex','Survived']].groupby(['Sex'], as_index = False).mean()


# In[599]:


trData[['SibSp','Survived']].groupby(['SibSp'], as_index = False).mean()


# In[600]:


trData['SibSp'].value_counts()


# In[601]:


trData[['Pclass','Fare','Survived']].groupby(['Pclass'], as_index = False).mean()


# In[602]:


pd.crosstab(trData['Sex'], trData['Survived'])


# In[603]:


X = trData[['Pclass', 'Sex','Age', 'Fare', 'Nfamily', 'Embarked']]
y = trData['Survived']
X.head()
testData.head()


# In[604]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)


# In[617]:


# Knearest neighbor
score = []
for n in range(1, 50):
    kneighbor = KNeighborsClassifier(n_neighbors=n)
    kneighbor.fit(Xtrain, ytrain)
    ypredict = kneighbor.predict(Xtest)
    score.append(metrics.accuracy_score(ytest, ypredict))
max(score)


# In[606]:


plt.plot(range(1,50), score,'.-')
plt.show()


# In[607]:


# logistic regression
logreg = LogisticRegression()
logreg.fit(Xtrain, ytrain)
ypredict = logreg.predict(Xtest)
lgscore = (metrics.accuracy_score(ytest, ypredict))
lgscore


# In[608]:


# Naive Baise
from sklearn.naive_bayes import GaussianNB
naiveB = GaussianNB()
naiveB.fit(Xtrain, ytrain)
ypredict = naiveB.predict(Xtest)
nbscore = (metrics.accuracy_score(ytest, ypredict))
nbscore


# In[609]:


# SVM
svmclf = SVC()
svmclf.fit(Xtrain, ytrain)
ypredict = svmclf.predict(Xtest)
svmscore = metrics.accuracy_score(ytest, ypredict)
svmscore


# ## From the above comparisons Logistic Regression with an accuracy of 0.83 is the best predictor 

# In[610]:


score = []
logreg = LogisticRegression()
logreg.fit(X, y)
metrics.accuracy_score(ytest, arpredict)


# In[611]:


testDataTemp = testData[['Pclass','Sex','Age', 'Fare','Nfamily','Embarked']]
arPredict = logreg.predict(testDataTemp)


# In[612]:


yPredict = pd.DataFrame({'PassengerId':testData['PassengerId'], 'Survived': arPredict})


# In[613]:


yPredict.head()


# In[614]:


yPredict.to_csv('../predictions.csv', index = False)
yPredict.shape


# In[615]:


X.head()


# In[616]:


testDataTemp.head()

