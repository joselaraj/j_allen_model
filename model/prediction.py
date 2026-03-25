import numpy as np
import pickle
from model.model import lin

# we are going to train a linear regression model to predict the number of attempts based on the other features
with open('attempts_model.pickle', 'rb') as f:
    lin = pickle.load(f) 

#we can let the user enter their own data to predict the number of attempts
avg_time_to_throw = float(input("Enter the average time to throw (in seconds): "))
attemps = float(input("Enter the number of attempts: "))
pass_touchdowns = float(input("Enter the number of passing touchdowns: "))
completions = float(input("Enter the number of completions: "))
interceptions = float(input("Enter the number of interceptions: "))
aggressiveness = float(input("Enter the aggressiveness rating (0-10): "))
dome = int(input("Dome (1 for yes, 0 for no): "))
temp = float(input("Enter the temperature (in degrees Fahrenheit): "))
wind = float(input("Enter the wind speed (in mph): "))


attempts_predict = np.array([[avg_time_to_throw,attemps, pass_touchdowns, completions, interceptions, aggressiveness, wind, temp, dome]])
lin.predict(attempts_predict)
print(f"Predicted number of attempts: {attempts_predict[0]}")
