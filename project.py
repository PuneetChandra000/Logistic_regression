import pandas as pd
import plotly_express as pe
import numpy as np
import matplotlib.pyplot as mp
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------

data = pd.read_csv("project.csv")

score = data["Velocity"].tolist()

accepted = data["Escaped"].tolist()

graph = pe.scatter(x = score , y = accepted) 

# --------------------------------------------------------------------------

x = np.reshape(score , (len(score) , 1))
y = np.reshape(accepted , (len(accepted) , 1))

lr = LogisticRegression()
lr.fit(x,y)

mp.figure()
mp.scatter(x.ravel() , y , color= 'black' , zorder = 20)

def model(x):
    return 1/(1+np.exp(-x))


x_test = np.linspace(0 , 100 , 200) 

chance = model(x_test * lr.coef_ + lr.intercept_).ravel()

mp.plot(x_test, chance, color='red', linewidth=3)

# axhline will plot the horizontal line on x axis based on the value of y

mp.axhline(y=0, color='k', linestyle='-')
mp.axhline(y=1, color='k', linestyle='-')
mp.axhline(y=0.5, color='b', linestyle='--')
mp.axvline(x=x_test[23], color='b', linestyle='--')

mp.ylabel('y')
mp.xlabel('X')
mp.xlim(0, 30)
mp.show()

print("-----------------------------")
