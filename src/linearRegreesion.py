import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import matplotlib.pyplot as plt

# Load the dataset from CSV file using pandas
df = pd.read_csv(
    r"C:\Users\ivans\Desktop\All\MSC\Supervised_Machine_learning\Assignment\AIML_assingment_1\study-hours-score-predictor\data\marks_v2.csv"
)

<<<<<<< HEAD
# Split the dataset into:
# - Features (hours studied)
# - Target variable (marks obtained)
x_train, x_test, y_train, y_test = train_test_split(
    df['hours'],
    df['marks'],
    test_size=0.2, # test_size=0.2  80% data for training and 20% for testing
    random_state=42 # random_state=42 ensures reproducible results
)
=======
df=pd.read_csv(r"..\study-hours-score-predictor\data\marks_v2.csv")
>>>>>>> 420395c9bcdf53f98f99531a21fd257ff3a70f67

# Convert pandas Series into NumPy arrays
# reshape((-1,1)) converts the data into a 2D array
x_train = np.array(x_train).reshape((-1, 1))
x_test = np.array(x_test).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))
y_test = np.array(y_test).reshape((-1, 1))

# Create the Linear Regression model
lrm = linear_model.LinearRegression()

# Train the model using the training dataset
# The model learns the relationship between study hours and marks
lrm.fit(x_train, y_train)

# Use the trained model to predict marks for test data
predictions = lrm.predict(x_test)

# Create a figure for visualization (width of figure = 8 && height of the figure= 5)
plt.figure(figsize=(8, 5)) 

# Plot the training dataset points
plt.scatter(
    x_train,
    y_train,
    color='blue', #Blue represent actual student data
    alpha=0.6,
    label='Training data'
)

# Sort x values before plotting the regression line
# This ensures the line appears smooth and correctly ordered
sorted_index = x_test.flatten().argsort()
x_sorted = x_test.flatten()[sorted_index]
y_sorted = predictions.flatten()[sorted_index]

# Plot the regression line
plt.plot(
    x_sorted,
    y_sorted,
    color='red', # Red line represents the predicted trend by the model
    linewidth=2,
    label='Regression line'
)

# labels and title for better readability
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Linear Regression: Hours vs Marks')

# Display legend on the graph
plt.legend()

# Show the final visualization
plt.show()