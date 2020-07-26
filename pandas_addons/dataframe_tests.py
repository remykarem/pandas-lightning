from itertools import combinations

import scipy
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


@pd.api.extensions.register_dataframe_accessor("tests")
class tests:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def categorical(self, alpha=0.05):
        df = self._obj.select_dtypes(include=["category"])
        pairs = combinations(df.columns, 2)
        for pair in pairs:
            col1, col2 = pair
            table = pd.crosstab(df[col1], df[col2])
            _, p, *_ = chi2_contingency(table)
            if p < alpha:
                print(f"{col1} and {col2} may be correlated")

    def numerical(self, top=5):
        corr = self._obj.corr()
        col_names = corr.columns
        corr = np.triu(corr.values)
        np.fill_diagonal(corr, 0)

        mat = scipy.sparse.coo_matrix(corr)
        sortind = np.argsort(-mat.data)
        it = zip(mat.row[sortind], mat.col[sortind], mat.data[sortind])

        df = pd.DataFrame(it)
        df.columns = ["col1", "col2", "corr"]
        df["col1"] = df["col1"].map(dict(enumerate(col_names)))
        df["col2"] = df["col2"].map(dict(enumerate(col_names)))

        df = df.iloc[:top]

        return df
