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

        corr_tidy = corr.abs().unstack().reset_index()
        corr_tidy = corr_tidy.rename(columns={
            "level_0": "feature_1",
            "level_1": "feature_2",
            0: "absolute_corr_coeff"})
        corr_tidy = corr_tidy.query("feature_1 != feature_2")
        corr_tidy = corr_tidy.sort_values(
            by="absolute_corr_coeff", ascending=False)

        return corr_tidy


    def get_cramersv(self):

        def cramers_corrected_stat(confusion_matrix):
            """ https://stackoverflow.com/a/39266194
                calculate Cramers V statistic for categorial-categorial association.
                uses correction from Bergsma and Wicher,
                Journal of the Korean Statistical Society 42 (2013): 323-328
            """
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum()
            phi2 = chi2/n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

        cols = self._obj.select_dtypes(include=["category", "bool"]).columns.tolist()
        if len(cols) != len(set(cols)):
            raise ValueError("Col names must be unique!")
        print(f"Testing {str(cols)}")

        pairs = combinations(cols, 2)

        cramers = []
        for pair in pairs:
            col1, col2 = pair
            confusion_matrix = pd.crosstab(
                self._obj[col1], self._obj[col2]).values
            coeff = cramers_corrected_stat(confusion_matrix)
            cramers.append(tuple([col1, col2, coeff]))

        cramers = pd.DataFrame(
            cramers, columns=["col1", "col2", "coefficient"])
        cramers.sort_values(by="coefficient", ascending=False,
                            inplace=True)

        return cramers
