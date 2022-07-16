import pdb
import numpy as np
import pandas as pd
import json
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds

# Data processing
f = open('data.json', 'r')
s = f.read()
f.close()
dic = json.loads(s)

namesList = []
pricesList = []
fatsList = []
proteinList = []
carbsList = []

for key, value in dic.items():
    print(f"Processing {key}")
    try:
        name = key
        price = value['price_per_g']
        fats = value['nutritional_value']['fat']
        protein = value['nutritional_value']['protein']
        carbs = value['nutritional_value']['carb']
        if not (price and fats and protein and carbs):
            raise Exception("Something is missing")
        namesList.append(name)
        pricesList.append(price)
        fatsList.append(fats)
        proteinList.append(protein)
        carbsList.append(carbs)
        print(f"  Successfully processed {name}")
    except:
        print(f"  Could not process {name}")

# pdb.set_trace()


namesList = np.array(namesList)
pricesList = np.array(pricesList)
fatsList = np.array(fatsList) / 100
proteinList = np.array(proteinList) / 100
carbsList = np.array(carbsList) / 100

# Define objective function (cost)
def obj(x):
    return np.dot(x,pricesList)

c = np.array([400,200,60])
A = np.vstack((carbsList,proteinList,fatsList))
linear_constraint = LinearConstraint(A, c, 1.1*c)
bounds = Bounds(0, np.inf)
x0 = np.ones(namesList.size)

res = minimize(obj, x0, constraints=[linear_constraint], bounds=bounds)
pdb.set_trace()

# Print results
mask = res.x > 1
