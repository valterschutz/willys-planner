# willys-planner
A python script for calculating the cheapest food combinations, given macros (how many grams of carbohydrates, proteins and fats to eat each day). Data generated with [willys-json](https://github.com/valterschutz/willys-json).

## Usage
```
python3 main.py [-h] [-n N] [--fluids] [-v] [-d DATA] [-b BLACKLIST] carbs protein fats
```
Run ```python3 main.py --help``` for info about each option.

## Dependencies
NumPy and SciPy. Install these by running ```python3 -m pip install numpy scipy```.
