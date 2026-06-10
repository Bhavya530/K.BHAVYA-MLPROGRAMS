#Q11
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('credit_score.csv')

X = data[['Age', 'Income', 'Loan_Amount']]
y = data['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_customer = [[32, 55000, 16000]]
prediction = model.predict(new_customer)

print("\nPredicted Credit Score:", prediction[0])
