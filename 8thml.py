import pandas as pd
from sklearn.linear_model import LinearRegression

d=pd.read_csv("linear.csv")

X=d[['x']]
y=d['y']

m=LinearRegression()
m.fit(X,y)

print(m.predict([[6]]))
