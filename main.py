import re
import sys
import numpy as np
import json
import argparse
from scipy.optimize import linprog

# Only show items in solution over threshold (in grams)
THRESHOLD = 1
# Accept macros up to some ratio of their specified value
MACRO_CONSTRAINT = 1.1

DATA_FILENAME = "data.json"
BLACKLIST_FILENAME = "blacklist.txt"

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("carbs", help = "Amount of carbohydrates (in grams) in a day", type = float)
parser.add_argument("protein", help = "Amount of protein (in grams) in a day", type = float)
parser.add_argument("fats", help = "Amount of fats (in grams) in a day", type = float)
parser.add_argument("-n", help = "Number of food combinations to generate. Default is 10.", type = int, default = 10)
parser.add_argument("--fluids", help = "Whether to include fluids in the result. Note that this will make the results unreliable and will require modification of the blacklist file to get decent output.", action = "store_true")
parser.add_argument("-v", "--verbose", help = "Show verbose output. Helpful to see which items were unable to be processed.", action = "store_true") 
parser.add_argument("-d", "--data", help = "File to get JSON data from. Default is \"data.json\".", default = "data.json")
parser.add_argument("-b", "--blacklist", help = "File to get blacklist names from. List one entry per line, product names that match a blacklist entry will be excluded from the results. Default is \"blacklist.txt\".", default = "blacklist.txt")
args = parser.parse_args()
carb_goal = args.carbs
protein_goal = args.protein
fat_goal = args.fats
n_results = args.n
include_fluids = args.fluids
is_verbose = args.verbose
data_filename = args.data
blacklist_filename = args.blacklist

# Read JSON data
f = open(data_filename, 'r')
s = f.read()
f.close()
products = json.loads(s)

namesList = []
pricesList = []
fatsList = []
proteinList = []
carbsList = []

# Get blacklisted words
blacklisted = []
f = open(blacklist_filename, "r")
for line in f:
    blacklisted.append(line.strip())
f.close()

n = len(products)
for i, product in enumerate(products):
    if is_verbose:
        print(f"Product {i}/{n}")
    try:
        name = product["name"]
        for blacklisted_name in blacklisted:
            if re.search(f'\\b{blacklisted_name.lower()}\\b', name.lower()):
                raise Exception(f"{name} was blacklisted")
        # Comparison price will be in units kr/g
        if product["comparePriceUnit"] == "kg":
            comparePrice = product["comparePrice"] / 1000
        elif include_fluids and product["comparePriceUnit"] == "l":
            # Assume 1 l = 1 kg
            comparePrice = product["comparePrice"] / 1000
        else:
            raise Exception(f"Invalid comparePriceUnit \"{product['comparePriceUnit']}\".")
        fats = float(product["fats"]) / 100
        protein = float(product["protein"]) / 100
        carbs = float(product["carbs"]) / 100
        namesList.append(name)
        pricesList.append(comparePrice)
        fatsList.append(fats)
        proteinList.append(protein)
        carbsList.append(carbs)
    except Exception as e:
        if is_verbose:
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

for i in range(n_results):
    # Define objective function (cost)
    def obj(x):
        return np.dot(x,pricesArray)
    b = np.array([carb_goal,protein_goal,fat_goal])
    A = np.vstack((carbsArray,proteinArray,fatsArray)) * keep_mask
    A_ub = np.vstack((A, -A))
    b_ub = np.concatenate((MACRO_CONSTRAINT*b, -b))
    res = linprog(pricesArray, A_ub, b_ub)
    # We are not guaranteed to find n_results solutions, especially
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
