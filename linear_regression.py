import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

#import dataset and acquire needed variables
dataset = pd.read_csv("/Users/stevesahayadarlin/Documents/UdemyML/Datasets/Datasets/house_prices.csv")
size = dataset['sqft_living']
price = dataset['price']

print(size)

#machine learning only handles arrays not dataframes
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

#Use these two lines to train the data
model = LinearRegression()
model.fit(x,y)

#MSE and R value
regression_model_mse = mean_squared_error(x,y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x,y))

#print out b1 and b2e
print(model.coef_[0])
print(model.intercept_[0])

plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color = "black")
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

#predict a price
print("Prediction by the model: ", model.predict([[2500]]))
