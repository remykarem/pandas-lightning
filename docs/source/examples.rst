.. pandas-addons documentation master file, created by
   sphinx-quickstart on Thu Jul 23 23:45:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Examples
========

Titanic
-------

.. code-block:: python

   import string
   import pandas as pd
   from pandas import CategoricalDtype
   import pandas_pipeline

Once you have imported, :code:`DataFrame.lambdas` will be available to you.

Reading
*******

>>> df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")

Changing types
**************

>>> df = df.lambdas.astype({
...     "PassengerId": "index",
...     "Name": str,
...     "Sex": "category",
...     "Embarked": "category",
...     "Pclass": [3, 2, 1]})

This is the same as

>>> df = df.set_index("PassengerId")
>>> df["Name"] = df["Name"].astype(str)
>>> df["Sex"] = df["Sex"].astype("category")
>>> df["Embarked"] = df["Embarked"].astype("category")
>>> df["Pclass"] = df["Pclass"].astype(CategoricalDtype([3, 2, 1], ordered=True)

Creating new features
*********************

>>> df = df.lambdas.sapply({
...   "Cabin": lambda s: s.str[0],
...   ("HasCabinCode", "Cabin"): lambda s: ~s.isna(),
...   ("HasDep", ("SibSp", "Parch")): lambda s, t: (s+t) > 0,
...   ("HasLetters", "Ticket"): lambda s: s.str.startswith(tuple(string.ascii_letters)) })

which is the same as

>>> df["Cabin"] = df["Cabin"].str[0]
>>> df["HasCabinCode"] = ~df["Cabin"].isna()
>>> df["HasDep"] = df["SibSp"] + df["Parch"] > 0
>>> df["HasLetters"] = df["Ticket"].str.startswith(tuple(string.ascii_letters))

Binning
*******

For numerical values,

>>> df = df.lambdas.map_numerical_binning({
...    "Age": range(0, 100, 10),
...    "Fare": ("quartile", 4)
... }, ordered=True)

which is the same as

>>> df["Age"] = pd.cut(df["Age"], range(0, 100, 10))
>>> df["Fare"] = pd.qcut(df["Fare"], 4)

For categorical values,

>>> df = df.lambdas.map_categorical_binning({
...    "Age": range(0, 100, 10),
...    "Fare": ("quartile", 4)
... }, ordered=True)

which is the same as

>>> df["Age"] = pd.cut(df["Age"], range(0, 100, 10))
>>> df["Fare"] = pd.qcut(df["Fare"], 4)

Plotting
********
