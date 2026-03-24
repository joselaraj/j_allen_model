'''
This file contains the code for the model.

first, im going to start off by importing the necessary libraries and loading the data.
'''
import sklearn as sk
from sklearn.linear_model import LinearRegression 
import pandas as pd 
import numpy as np
from data_src import data_src
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle 
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

#import our data
data = pd.read_csv(data_src) 

#this is where Im going to choose what attributes are going to be used
'''
avg time to throw: The more time the higher chance for sack/scramble. But also chance for a deep ball
pass tds: a lot of tds means receivers are open and qb is in good rhythm
comp: more completions means more yards
int: can impact the qb confidence and also the defense confidence 
aggressiveness: how aggressively the qb plays
wind: wind conditions can affect passing (0 wind if playing in dome)
temp: temperature can impact player performance 
dome: playing indoors vs outdoors
'''
data = data[['avg_time_to_throw',
             'pass_touchdowns',
             'attempts',
             'pass_yards',
             'completions',
             'interceptions',
             'aggressiveness',
             'wind',
             'temp',
             'dome']]

# now we are going to enter what we want to predict which is the passing yards and attempts
# returns a new data frame 
X = np.array(data.drop(['attempts'], axis=1))
y = np.array(data['attempts'])

# now we are going to split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
'''
best = 0 
for _ in range(30):

    lin = LinearRegression() 

    lin.fit(X_train, y_train)

    acc = lin.score(X_test, y_test)
    print(acc)

    if acc > best:
        best = acc 
        with open('attempts_model.pickle', 'wb') as f:
            pickle.dump(lin, f)

''' 
pickle_in = open('attempts_model.pickle', 'rb')
lin = pickle.load(pickle_in)

style.use('ggplot')

y_pred = lin.predict(X_test)

# calculate the best fit line
m, b = np.polyfit(y_test, y_pred, 1)
best_fit_x = np.linspace(y_test.min(), y_test.max(), 100)
best_fit_y = m * best_fit_x + b

print(lin.score(X_test,y_test))

plt.scatter(y_test, y_pred)
plt.plot(best_fit_x, best_fit_y, '--r', label=f'y = {m:.2f}x + {b:.2f}')
plt.xlabel('Actual')
plt.ylabel('Predicted ')
plt.title('Attempts - Actual vs Predicted ')
plt.legend()
plt.show()