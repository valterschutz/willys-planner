import sys
import pdb
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.optimize import minimize
# from scipy.optimize import LinearConstraint
# from scipy.optimize import Bounds
from scipy.optimize import linprog

CARB_GOAL = 300
PROTEIN_GOAL = 200
FAT_GOAL = 60
N_RESULTS = int(sys.argv[2])  # Maximum number of food combinations to show
# Only show items in solution over threshold (in grams)
THRESHOLD = 1
# Accept macros up to some ratio of their specified value
MACRO_CONSTRAINT = 1.1

sns.set_theme()

# Data processing
data_name = sys.argv[1]
f = open(data_name, 'r')
s = f.read()
f.close()
products = json.loads(s)

namesList = []
pricesList = []
fatsList = []
proteinList = []
carbsList = []

n = len(products)
for i, product in enumerate(products):
    print(f"Product {i}/{n}")
    try:
        name = product["name"]
        if product["comparePriceUnit"] == "kg":
            comparePrice = product["comparePrice"] / 1000
        else:
            raise Exception(f"comparePriceUnit was not kg for {name}")
        fats = float(product["fats"]) / 100
        protein = float(product["protein"]) / 100
        carbs = float(product["carbs"]) / 100
        namesList.append(name)
        pricesList.append(comparePrice)
        fatsList.append(fats)
        proteinList.append(protein)
        carbsList.append(carbs)
    except Exception as e:
        print(f"  Could not process {name}, error: {e}")

namesArray = np.array(namesList)
pricesArray = np.array(pricesList)
fatsArray = np.array(fatsList)
proteinArray = np.array(proteinList)
carbsArray = np.array(carbsList)

results = []

# As we generate more food combinations, the foods to choose from
# become more limited.
keep_mask = np.ones(namesArray.size)  

for i in range(N_RESULTS):
    # Define objective function (cost)
    def obj(x):
        return np.dot(x,pricesArray)
    b = np.array([CARB_GOAL,PROTEIN_GOAL,FAT_GOAL])
    A = np.vstack((carbsArray,proteinArray,fatsArray)) * keep_mask
    A_ub = np.vstack((A, -A))
    b_ub = np.concatenate((2*b, -b))
    res = linprog(pricesArray, A_ub, b_ub)
    # We are not guaranteed to find N_RESULTS solutions, especially
    # if we don't have many foods to choose from.
    if res.success:
        keep_mask = keep_mask * (res.x < 0.1)
        results.append(res)
    else:
        break

# Print results
print()
for j, res in enumerate(results):
    print(f"Option #{j+1}:")
    mask = res.x > 1
    res_names = namesArray[mask]
    res_grams = res.x[mask]
    res_prices = res.x[mask] * pricesArray[mask]
    for i in range(len(res_names)):
        print(f"{round(res_grams[i],2)}g of {res_names[i]} ({round(res_prices[i],2)} kr)")
    tot_cost = np.sum(res_prices)
    print(f"Total cost: {round(tot_cost,2)} kr")
    print()
