# Question 1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
#importing the data set
df = pd.read_csv("kc_house_data.csv")
df
df.shape
df.size
df.isnull().sum()
df.dtypes
del df["date"]
df.columns
sn.kdeplot(df["price"], shade=True)
ab=np.log(df["price"]) # to remove skewness
ab
sn.kdeplot(ab, shade=True)
plt.figure(figsize=(10,8)) # it describes every variable mapping with every variable
corr=df.corr()
sn.heatmap(corr,annot=True)
X=np.array(df.price).reshape(21613,1)
X
Y=np.array(df.sqft_living).reshape(21613,1)
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
regression=LinearRegression()
regression.fit(Y_train,X_train)
regression.coef_
regression.intercept_
x_pred = regression.predict(Y_test)
x_pred
a=pd.DataFrame({'actual':[X_test],'prediction':[x_pred]})
a
