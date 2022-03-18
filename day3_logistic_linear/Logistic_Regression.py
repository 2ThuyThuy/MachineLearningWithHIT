import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('data_Logistic.csv')
X = dataset.iloc[:,-3:-2].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)

def sigmoid(s):
    return 1/(1 + np.exp(-s))



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

print(model.coef_, model.intercept_, model.n_iter_)
coef = model.coef_
xx = np.linspace(15,65,100)
y_pred = model.predict(X_test)

#plt.scatter(X_test,y_pred)
#plt.xlabel('Age')
plt.show()