.. pandas-lightning documentation master file, created by
   sphinx-quickstart on Thu Jul 23 23:45:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting started
===============

Install
-------

.. code-block:: bash

   pip install git+https://github.com/remykarem/pandas-lightning#egg=pandas-lightning

Import
------

>>> import pandas as pd
>>> import pandas_lightning

Dataframe accessors like :code:`.lambdas` and series accessors like :code:`scaler`
will be available to your DataFrame and Series objects.

.. note::
   Accessors are like :code:`str`

Use df.lambdas
--------------

**df.transform_columns**

>>> df = pd.util.testing.makeMissingDataframe()
>>> def squared(x):
...     return x*x

What used to be

>>> df["A"] = abs(df["A"])
>>> df["B"] = df["B"] * 10
>>> df["C_squared"] = squared(df["C"])
>>> df["AB"] = df["A"] + df["B"]

can be rewritten as

>>> df = df.transform_columns(
...     A=abs,
...     B=lambda b: b * 10,
...     C_squared=("C", squared),
...     AB=(["A", "B"], lambda a, b: a+b),
... )

**df.cast**

What used to be

>>> df["A"] = df["A"].astype("category")
>>> df["B"] = df["B"].astype(int)
>>> df["C_abs"] = df["C_abs"].astype(str)
>>> df["D"] = pd.to_datetime(df["D"])

can be rewritten as

>>> df = df.cast(
...      A="category",
...      B=int,
...      C_abs=str,
...      D="datetime",
... )

See the full set of dataframe accessors :ref:`here <dataframe:DataFrame accessors>`.

Use df[col].scaler
------------------

>>> sr = pd.Series([1, 2, 3, 4, 5])
>>> sr.scaler.standardize()
0   -1.264911
1   -0.632456
2    0.000000
3    0.632456
4    1.264911
dtype: float64
>>> sr.scaler.minmax()
0    0.00
1    0.25
2    0.50
3    0.75
4    1.00
dtype: float64

Use df[col].pctg
----------------

>>> sr = pd.Series([1, None, 0, 8.3, None])
>>> sr.pctg.nans
0.4
>>> sr.pctg.zeros
0.2

Use df[col].ascii
-----------------

Plotting using :code:`.ascii.hist()`

>>> sr = pd.Series(["red", "blue", "red", "red", "orange", "blue"])
>>> sr.ascii.hist()
       red ##############################
      blue ####################
    orange ##########

Use df[col].map_numerical_binning
---------------------------------

>>> sr = pd.Series([23, 94, 44, 95, 29, 8, 17, 42, 29, 48,
...                 96, 95, 17, 97, 9, 85, 62, 71, 37, 10,
...                 41, 88, 18, 56, 85, 22, 97, 27, 69, 19,
...                 37, 10, 85, 11, 73, 96, 56, 0, 18, 3,
...                 54, 50, 91, 38, 46, 13, 78, 22, 6, 61])
>>> sr_cat = sr.map_numerical_binning([0, 18, 21, 25, 30, 100])
>>> sr_cat.ascii.hist()
   (0, 18] ############
  (18, 21] #
  (21, 25] ###
  (25, 30] ###
 (30, 100] ##############################

Use df[col].map_categorical_binning
-----------------------------------

>>> sr = pd.Series(["apple", "spinach", "cashew", "pear", "kailan",
...                 "macadamia", "orange"])
0        apple
1      spinach
2       cashew
3         pear
4       kailan
5    macadamia
6       orange
dtype: object

Specify a mapping with the new category as the key and the old categories as a list

>>> GROUPS = {
...     "fruits": ["apple", "pear", "orange"],
...     "vegetables": ["kailan", "spinach"],
...     "nuts": ["cashew", "macadamia"]}

Then call the :ref:`<series:Series.map_numerical_binning>` API.

>>> sr.map_categorical_binning(GROUPS)
0        fruits
1    vegetables
2          nuts
3        fruits
4    vegetables
5          nuts
6        fruits
dtype: category
Categories (3, object): [fruits, vegetables, nuts]

Read more :ref:`here <series:Series accessors>`.
