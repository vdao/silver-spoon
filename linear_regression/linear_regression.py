import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = dataframe[['BMI']]
y_values = dataframe[['Life expectancy']]

#train model on data
bmi_regression = linear_model.LinearRegression()
bmi_regression.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, bmi_regression.predict(x_values))
plt.show()
