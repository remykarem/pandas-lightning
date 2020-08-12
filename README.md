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

## Import

```python
>>> import pandas as pd
>>> import pandas_lightning
```

Dataframe accessors like :code:`.lambdas` and series accessors like :code:`scaler`
will be available to your DataFrame and Series objects.

Read more here: https://pandas-lightning.readthedocs.io/
