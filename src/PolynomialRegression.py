import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r"C:\Users\ivans\Desktop\All\MSC\Supervised_Machine_learning\Assignment\AIML_assingment_1\study-hours-score-predictor\data\marks_v2.csv") #importing datasets using pandas

x=df[['hours']].values

y=df['marks'].values

poly = PolynomialFeatures(degree=3 , include_bias=True)
x_transform=poly.fit_transform(x)


print(x_transform.shape)
print(x_transform)

model = LinearRegression(fit_intercept=False)
model.fit(x_transform,y)

y_pred=model.predict(x_transform)
print('Predicted response:')
print(y_pred)

# Plot
plt.figure(figsize=(8,5))

# Actual data points
plt.scatter(x, y, color='blue', s=50, label='Actual Data')

# Polynomial regression curve
sorted_index = x.flatten().argsort()
x_sorted = x.flatten()[sorted_index]
y_sorted = y_pred.flatten()[sorted_index]

plt.plot(x_sorted, y_sorted, color='red', linewidth=2,
         label='Polynomial Regression')

# Labels and title
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Polynomial Regression (Degree 2)')

# Styling
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()