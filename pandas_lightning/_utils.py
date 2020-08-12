from random import sample, choices
import numpy as np
import pandas as pd


def makeDataFrame(nrows=20, ncols=5, proportion_missing=0):
    col_types = ["int", "float", "str", "datetime", "category"]
    cols = choices(col_types, k=ncols)

    df = []
    for col in cols:
        if col == "int":
            x = pd.Series(np.random.randint(0, high=20, size=nrows))
        elif col == "str":
            x = pd.Series(pd.util.testing.makeStringIndex(k=nrows))
        elif col == "category":
            x = pd.Series(pd.util.testing.makeCategoricalIndex(k=nrows, n=3))
        elif col == "datetime":
            x = pd.Series(pd.util.testing.makeTimeSeries(nper=nrows).index)
        else:
            x = pd.Series(np.random.randn(nrows))

        df.append(x)

    df = pd.concat(df, axis=1)

    if proportion_missing > 0:
        df = df.mask(np.random.random(df.shape) < proportion_missing)

    return df
