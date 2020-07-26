# pandas-addons (WIP)

pandas-addons is an API designed for data preprocessing based on common
patterns and idioms in pandas. You can expect the following:

- Reduce repeated code
- Make code more readable
- Manage sequences of operations
- Prototype features more quickly
- Access methods intuitively using convenience functions

## Install

```bash
pip install git+https://github.com/remykarem/pandas-addons#egg=pandas-addons
```

## Import

```python
>>> import pandas as pd
>>> import pandas_addons
```

Dataframe accessors like :code:`.lambdas` and series accessors like :code:`scaler`
will be available to your DataFrame and Series objects.

Familiar API

* `astype`: convert the types of multiple columns
* `apply`: perform apply on multiple series objects
* `map`: perform map on multiple columns

New

* `sapply`: perform vectorised operations on multiple series
* `dapply`
* `drop_duplicate_columns`
* `drop_columns_with_rules`
* `map_categorical_binning`
* `map_numerical_binning`
