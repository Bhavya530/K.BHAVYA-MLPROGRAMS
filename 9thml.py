#Q9

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = {
    'Level':[1,2,3,4,5,6,7,8,9,10],
    'Salary':[45000,50000,60000,80000,110000,
              150000,200000,300000,500000,1000000]
}

df = pd.DataFrame(data)

X = df[['Level']]
y = df['Salary']

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

print("Linear Prediction for Level 6.5:")
print(lin_reg.predict([[6.5]])[0])

print("Polynomial Prediction for Level 6.5:")
print(poly_reg.predict(poly.transform([[6.5]]))[0])
