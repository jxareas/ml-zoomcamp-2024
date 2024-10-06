# %% Imports

import pandas as pd

# %% Dataframes

data = [
    ['Nissan', 'Stanza', 1991, 138, 4, 'MANUAL', 'sedan', 2000],
    ['Hyundai', 'Sonata', 2017, None, 4, 'AUTOMATIC', 'Sedan', 27150],
    ['Lotus', 'Elise', 2010, 218, 4, 'MANUAL', 'convertible', 54990],
    ['GMC', 'Acadia', 2017, 194, 4, 'AUTOMATIC', '4dr SUV', 34450],
    ['Nissan', 'Frontier', 2017, 261, 6, 'MANUAL', 'Pickup', 32340],
]

columns = [
    'Make', 'Model', 'Year', 'Engine HP', 'Engine Cylinders',
    'Transmission Type', 'Vehicle_Style', 'MSRP'
]

# Creating a dataframe
df = pd.DataFrame(data, columns=columns)
print(df)

data = [
    {
        "Make": "Nissan",
        "Model": "Stanza",
        "Year": 1991,
        "Engine HP": 138.0,
        "Engine Cylinders": 4,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "sedan",
        "MSRP": 2000
    },
    {
        "Make": "Hyundai",
        "Model": "Sonata",
        "Year": 2017,
        "Engine HP": None,
        "Engine Cylinders": 4,
        "Transmission Type": "AUTOMATIC",
        "Vehicle_Style": "Sedan",
        "MSRP": 27150
    },
    {
        "Make": "Lotus",
        "Model": "Elise",
        "Year": 2010,
        "Engine HP": 218.0,
        "Engine Cylinders": 4,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "convertible",
        "MSRP": 54990
    },
    {
        "Make": "GMC",
        "Model": "Acadia",
        "Year": 2017,
        "Engine HP": 194.0,
        "Engine Cylinders": 4,
        "Transmission Type": "AUTOMATIC",
        "Vehicle_Style": "4dr SUV",
        "MSRP": 34450
    },
    {
        "Make": "Nissan",
        "Model": "Frontier",
        "Year": 2017,
        "Engine HP": 261.0,
        "Engine Cylinders": 6,
        "Transmission Type": "MANUAL",
        "Vehicle_Style": "Pickup",
        "MSRP": 32340
    }
]

df = pd.DataFrame(data)
print(df.head(n=2))

# %% Pandas Series

print(df.Make)
print(df['Make'])

# Subset of the dataframe
print(df[['Make', 'Model', 'MSRP']])

# Creating a new column
df['id'] = [1, 2, 3, 4, 5]

del df['id']

# %% Index
print(df.index)

df.index = ['a', 'b', 'c', 'd', 'e']

print(df.loc[['b', 'c']])
print(df.iloc[1:3, ])

df = df.reset_index(drop=True)

# %% Element-wise operations

df['Engine HP'] / 100

# %% Filtering

df[df['Year'] >= 2015]

# %% String operations

df['Vehicle_Style'] = df['Vehicle_Style'] \
    .str.lower() \
    .str.replace(' ', '_')

# %% Summarizing operators

df['MSRP'].min()
df['MSRP'].max()
df['MSRP'].mean()
df['MSRP'].describe()

df.describe()

# %% Missing values

df.isnull().sum()

# %% Grouping

# """
# SELECT transmission_type, AVG(MSRP)
# FROM cars
# GROUP BY transmission_type
# """

df.groupby('Transmission Type')['MSRP'].mean()


# %% Getting the ndarrays

# Getting the underlying numpy array
df['MSRP'].values

# Converting a pandas dataframes back to a list of dictionaries
df.to_dict(orient='records')
