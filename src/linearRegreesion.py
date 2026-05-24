import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import matplotlib.pyplot as plt


df=pd.read_csv(r"C:\Users\ivans\Desktop\All\MSC\Supervised_Machine_learning\Assignment\AIML_assingment_1\study-hours-score-predictor\data\marks_v2.csv")

x_train, x_test ,y_train , y_test = train_test_split(df['hours'],df['marks'],test_size=0.2,random_state=42)

x_train = np.array(x_train).reshape((-1,1))
x_test = np.array(x_test).reshape((-1,1))
y_train=np.array(y_train).reshape((-1,1))
y_test=np.array(y_test).reshape((-1,1))



lrm=linear_model.LinearRegression()
lrm.fit(x_train,y_train)

predictions =lrm.predict(x_test)



plt.scatter(x_train,y_train)
plt.plot(x_test,predictions,color='yellow',linewidth=3)
plt.xlabel('hours')
plt.xlabel('marks')
plt.title('Linear Regression')
plt.show()
