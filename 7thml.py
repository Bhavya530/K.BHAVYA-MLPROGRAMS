import pandas as pd
from sklearn.linear_model import LogisticRegression

d=pd.read_csv("logistic.csv")

X=d[['Age','Salary']]
y=d['Buy']

m=LogisticRegression()
m.fit(X,y)

print(m.predict([[48,55000]]))
