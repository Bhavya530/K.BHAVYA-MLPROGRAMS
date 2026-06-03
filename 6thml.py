import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score

d=pd.read_csv("nb.csv")

X=d[['Age','Income']]
y=d['Buy']

m=GaussianNB()
m.fit(X,y)

p=m.predict(X)

print("Confusion Matrix")
print(confusion_matrix(y,p))

print("Accuracy =",accuracy_score(y,p))
