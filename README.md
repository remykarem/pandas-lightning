# pandas-pipeline (WIP)

Organize your pipelines

Install

```bash
pip install git+https://github.com/remykarem/pandas-pipeline#egg=pandas-pipeline
```

Firstl, import

```python
import pandas_pipeline
```

such that the `.pipeline` will be available to your DataFrame objects.

Familiar API

* `astype`: convert the types of multiple columns
* `apply`: perform apply on multiple series objects
* `map`: perform map on multiple columns
* `fillna`

New

* `sapply`: perform vectorised operations on multiple series
* `dapply`
* `drop_duplicate_columns`
* `drop_columns_with_rules`
* `map_categorical_binning`
* `map_numerical_binning`

## `.pipeline.astype`

Instead of

```python
df["age"] = df["age"].astype(int)
df["shoe_size"] = df["shoe_size"].astype(str)
df["weight_new"] = df["weight"].astype("category")
```

do

```python
df = df.pipeline.astype({
    "age": int,
    "shoe_size": str,
    ("weight_new", "weight"): "category"
})
```

## `.pipeline.apply`

Instead of

```python
df["name"] = df["name"].apply(lambda x: x.lower())
```

do

```python
df = df.pipeline.apply({
    "name": lambda x: x.lower()
})
```

## `.pipeline.sapply`

Instead of

```python
df["Height"] = df["Height"] + 100
df["Cabin"] = df["Cabin"].str[0]
df["HasLetters"] = df["Ticket"].str.startswith("a")
df["HasCabinCode"] = ~df["CabinType"].isna()
```
do

```python
df = df.pipeline.sapply({
    "Height": lambda s: s+100,
    "Cabin": lambda s: s.str[0],
    ("HasLetters", "Ticket"): lambda s: s.str.startswith("a"),
    ("HasCabinCode", "CabinType"): lambda s: ~s.isna()
})
```

## `.pipeline.map_numerical_binning`

Instead of

```python
df["Age"] = pd.cut(df["Age"], [0, 12, 24, 60])
df["SibSp"] = pd.cut(df["SibSp"], 4)
```

do

```python
df = df.map_numerical_binning({
    "Age": [0, 12, 24, 60],
    "SibSp": 4
})

## `.pipeline.map_categorical_binning`

Instead of

```python
WORKING_CLASS = {
    'Never-worked': 'No income',
    'Without-pay': 'No income',
    'Self-emp-not-inc': 'No income',
    'Private': 'Private',
    'Local-gov': 'Govt',
    'State-gov': 'Govt',
    'Federal-gov': 'Govt'
}
df["workclass"] = df["workclass"].map(WORKING_CLASS).astype("category")
```

do

```python
WORKING_CLASS = {
    "No income": ['Never-worked', 'Without-pay', 'Self-emp-not-inc'],
    "Private": ["Private"],
    "Govt": ['Local-gov', 'State-gov', 'Federal-gov']
}
df = df.pipeline.map_categorical_binning(ordered=True, binnings={
    "workclass": WORKING_CLASS
})
```
