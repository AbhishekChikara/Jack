import pandas as pd 
import numpy as np 

df1 = pd.read_csv('train.csv')

df1 = df1.drop(['Name','Ticket','Cabin'],axis =1)

age_mean = df1['Age'].mean()


df1['Age'] = df1['Age'].fillna(age_mean)

df1['Sex'] = df1['Sex'].map({'female':0,'male':1})

df1['Embarked'] = df1['Embarked'].fillna('S')

df1 = pd.concat([df1,pd.get_dummies(df1['Embarked'], prefix = 'Embarked')],axis = 1)

df1 = df1.drop(['Embarked'],axis =1)

# print(df1.head())

X_train = df1.iloc[ : , 2 : ].values
y_train = df1['Survived']

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

parameter_grid = {
	'max_features': [0.5,1.],
	'max_depth' :  [5.,None]
}   

grid_search = GridSearchCV(RandomForestClassifier(n_estimators = 100),parameter_grid,cv = 5 , verbose = 3)

grid_search.fit(X_train,y_train)

# print(grid_search.best_params_)

model = RandomForestClassifier(n_estimators=100,max_features =0.5,max_depth = 5.0,random_state =0)
model =model.fit(X_train,y_train)

#getting the test data
df2 = pd.read_csv('test.csv')

df2 = df2.drop(['Name','Ticket','Cabin'],axis =1)

age_mean2 = df2['Age'].mean()


df2['Age'] = df2['Age'].fillna(age_mean2)


df2['Sex'] = df2['Sex'].map({'female':0,'male':1})

df2['Embarked'] = df2['Embarked'].fillna('S')

df2 = pd.concat([df2,pd.get_dummies(df2['Embarked'], prefix = 'Embarked')],axis = 1)

df2 = df2.drop(['Embarked'],axis =1)
df2 = df2.dropna()
X_test = df2.iloc[:, 1:]


y_prediction = model.predict(X_test)
print(np.sum(y_prediction)/len(y_prediction))