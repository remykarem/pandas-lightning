from math import ceil, floor
import pandas as pd
import numpy as np
from pandas import CategoricalDtype


@pd.api.extensions.register_series_accessor("pctg")
class Percentage:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def zeros(self):
        return (self._obj == 0).sum() / len(self._obj)

    @property
    def nans(self):
        return self._obj.isna().sum() / len(self._obj)

    @property
    def uniques(self):
        return self._obj.nunique() / len(self._obj)


@pd.api.extensions.register_series_accessor("scaler")
class Scaler:
    """Uses sklearn vocabulary
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def standardize(self, ddof=1):
        return (self._obj - self._obj.mean()) / self._obj.std(ddof=ddof)

    def minmax(self, feature_range=(0, 1)):
        feature_min, feature_max = feature_range

        max_val = max(self._obj)
        min_val = min(self._obj)

        std = (self._obj - min_val) / (max_val - min_val)
        scaled = std * (feature_max - feature_min) + feature_min

        return scaled

    def normalize(self):
        """Scale features to have a unit norm
        """
        raise NotImplementedError

    def log1p(self):
        return np.log1p(self._obj)

    def expm1(self):
        return np.expm1(self._obj)


@pd.api.extensions.register_series_accessor("ascii")
class Ascii:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def hist(self, size=10, hashes=30, num_characters=10):

        sort = True

        if self._obj.dtype.name.startswith("float"):
            min_val = (floor(min(self._obj)/10))*10
            max_val = (ceil(max(self._obj)/10))*10
            sr = pd.cut(self._obj, range(min_val, max_val, size))
            sort = False
        elif self._obj.dtype.name.startswith("int") or \
                (self._obj.dtype.name == "category" and getattr(self._obj.dtype, "ordered")):
            sort = False
            sr = self._obj
        else:
            sr = self._obj

        freqs = sr.value_counts(sort=sort)
        max_val = max(freqs)
        sr = freqs/max_val * hashes

        for label, count in sr.to_dict().items():
            if len(label) > num_characters:
                label = str(label)[:(num_characters-3)] + "..."
            else:
                label = str(label)
                str_format = ">" + str(num_characters)
                label = format(label, str_format)
            print(label, int(count)*"#")


@pd.api.extensions.register_series_accessor("map_numerical_binning")
class MapNumericalBinning:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, binning, ordered=True, inplace=False):
        """
        Examples
        --------

        Ranged binning (list or range)
        >>> df.pipeline.map_numerical_binning({
                "age": [0,18,21,25,30,100]
            })

        Ranged binning (dictionary)
        >>> GROUPS = {
                "": 0,
                "kids": 12,
                "teens": 24,
                "adults": 60
            }
        >>> df.pipeline.map_numerical_binning({
                "age": GROUPS
            })

        Binning with equal size (int)
        >>> df.pipeline.map_numerical_binning({
                "age": 4
            })

        Binning by quantiles (tuple of str and int)
        >>> df.pipeline.map_numerical_binning({
                "age": ("quantiles", 4)
            })
        """
        if isinstance(binning, tuple):
            _, quantiles = binning
            s = pd.qcut(self._obj, quantiles)
        elif isinstance(binning, (list, range, int)):
            s = pd.cut(self._obj, binning)
        else:
            raise NotImplementedError

        return s


@pd.api.extensions.register_series_accessor("map_categorical_binning")
class MapCategoricalBinning:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def map_categorical_binning(self, binning, ordered=False, inplace=False):
        mapping = {v: k for k, values in binning.items()
                   for v in values}

        s = self._obj.map(mapping).astype(
            CategoricalDtype(mapping.keys(), ordered=ordered))

        return s
