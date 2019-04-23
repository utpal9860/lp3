import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

dataset=pd.read_csv("ml1.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
print(x)
print(y)
from sklearn.linear_model import LinearRegression
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                 #   random_state
reg = LinearRegression()
reg.fit(x,y)
reg.coef_
reg.intercept_

print ("Accuracy : "+ str(reg.score(x,y)*100))

hours = int(input('Enter the number of hours : '))
eq = reg.coef_*hours+reg.intercept_
print ('y = %f*%f+%f' % (reg.coef_,hours,reg.intercept_))
print ("Risk Score: ", eq[0])
plt.plot(x,y,'o')
plt.plot(x,reg.predict(x))
plt.show()
