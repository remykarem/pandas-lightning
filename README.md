# pandas lightning

pandas-lightning is an API designed to abstract common
patterns and idioms in pandas. You can expect the following:

- Reduce repeated code
- Make code more readable
- Manage sequences of operations
- Prototype features more quickly
- Access methods intuitively using convenience functions

## Install

```bash
pip install git+https://github.com/remykarem/pandas-lightning#egg=pandas-lightning
```

## Features

```python
>>> import pandas as pd
>>> import pandas_lightning
```

Dataframe accessors like :code:`.lambdas` and series accessors like :code:`scaler`
will be available to your DataFrame and Series objects.

Some features include:

* [Change types](#change-types)
* [Create new features](#create-new-features)
* [Apply a sequence of functions to DataFrame](#apply-a-sequence-of-functions)
* [Drop columns with rules](#drop-columns-with-rules)
* [Categorical binning](#categorical-binning)

### Change types

Previously:

```python
>>> df = df.set_index("PassengerId")
>>> df["Name"] = df["Name"].astype(str)
>>> df["Sex"] = df["Sex"].astype("category")
>>> df["Embarked"] = df["Embarked"].astype("category")
>>> df["Pclass"] = df["Pclass"].astype(CategoricalDtype([3, 2, 1], ordered=True)
```

Now:

```python
>>> df = df.cast(
...     PassengerId="index",
...     Name=str,
...     Sex="category",
...     Embarked="category",
...     Pclass=[3, 2, 1])
```

### Create new features

Previously:

```python
>>> df["Cabin"] = df["Cabin"].str[0]
>>> df["HasCabinCode"] = ~df["Cabin"].isna()
>>> df["HasDep"] = df["SibSp"] + df["Parch"] > 0
>>> df["HasLetters"] = df["Ticket"].str.startswith(tuple(string.ascii_letters))
```

Now:

```python
>>> df = df.add_columns(
...   Cabin=lambda s: s.str[0],
...   HasCabinCode=("Cabin", lambda s: ~s.isna()),
...   HasDep=(["SibSp", "Parch"], lambda s, t: (s+t) > 0),
...   HasLetters=("Ticket", lambda s: s.str.startswith(tuple(string.ascii_letters)))
```

### Apply a sequence of functions

Define some functions

```python
>>> def drop_some_columns(data):
...     ...
...     return data
>>> def reindex(data):
...     ...
...     return data
>>> def rename_columns(data):
...     ...
...     return data
```

Previously:

```python
>>> df = drop_some_columns(df)
>>> df = rename_columns(df)
>>> df = reindex(df)
```

Now:

```python
>>> df = df.lambdas.dapply(
...     drop_some_columns,
...     rename_columns,
...     reindex
... )
```

### Drop columns with rules

```python
>>> df = pd.DataFrame({"X": [np.nan, np.nan, np.nan, np.nan, "hey"],
...                    "Y": [0, np.nan, 0, 0, 1],
...                    "Z": [1, 9, 5, 4, 2]})
```

```python
>>> df.lambdas.drop_columns_with_rules(
...     lambda s: s.pctg.nans > 0.75,
...     lambda s: s.pctg.zeros > 0.5)
   Z
0  1
1  9
2  5
3  4
4  2
```

### Categorical binning

```python
>>> sr = pd.Series(["apple", "spinach", "cashew", "pear", "kailan",
...                 "macadamia", "orange"])
>>> sr
0        apple
1      spinach
2       cashew
3         pear
4       kailan
5    macadamia
6       orange
dtype: object
```

```python
>>> GROUPS = {
...     "fruits": ["apple", "pear", "orange"],
...     "vegetables": ["kailan", "spinach"],
...     "nuts": ["cashew", "macadamia"]}
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
```

## Roadmap

- [ ] Hashing
- [ ] Pipelining

Read more here: https://pandas-lightning.readthedocs.io/
