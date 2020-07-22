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

## `.pipeline.dapply`

Instead of

```python
df = df.drop(columns=["col_a", "col_b", "col_c"])
df = df.replace("?", np.nan)
```
do

```python
df = df.pipeline.dapply({
    lambda df: df.drop(columns=["col_a", "col_b", "col_c"]),
    lambda df: df.replace("?", np.nan),
})
```

do

```python
df = df.pipeline.map({

})
```

```python
df = df.pipeline.fillna({
})
```
