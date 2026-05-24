import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

#Import dataset using Pandas
df = pd.read_csv(r"..\study-hours-score-predictor\data\marks_v2.csv")

x = df[['hours']].values # Extract feature (independent variable) - study hours
y = df['marks'].values # Extract target variable (dependent variable) - marks
 

# TRAIN / TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.45,  # 55% data for training and 45% for testing
    random_state=42
)

poly = PolynomialFeatures(degree=3) # Using a 3 degree

# Transform features
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Train model
model = LinearRegression()
model.fit(x_train_poly, y_train)

# Predict on test set
y_pred = model.predict(x_test_poly)


print("MAE:", metrics.mean_absolute_error(y_test, y_pred))  # average absolute error
print("MSE:", metrics.mean_squared_error(y_test, y_pred))   # squared error penalty
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  # error in original scale
print("R2 Score:", metrics.r2_score(y_test, y_pred))  # how well model explains variance

# visualisation Result

plt.figure(figsize=(10,6))

plt.scatter(x, y, color='blue', label='Training Data')

#Creating Smooth curve for polynomial regression
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_line_poly = poly.transform(x_line)
y_line = model.predict(x_line_poly)

plt.plot(x_line, y_line, color='red', label='Polynomial Regression (Degree 3)') # Plot regression curve

# Labels
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Polynomial Regression (Degree 3)')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()