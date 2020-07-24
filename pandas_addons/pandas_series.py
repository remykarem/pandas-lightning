from math import ceil, floor
from typing import OrderedDict, Union

import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype
from pandas.errors import EmptyDataError


@pd.api.extensions.register_series_accessor("pctg")
class pctg:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")

    @property
    def zeros(self):
        """Get the percentage of zeros in the Series

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series([1, 2, 0, 8.3, 0])
        >>> sr.pctg.zeros
        0.4

        Returns
        -------
        Series
            Return a copy of the Series
        """
        return (self._obj == 0).sum() / len(self._obj)

    @property
    def nans(self):
        """Get the percentage of missing values in the Series

        Example
        -------
        >>> import pandas as pd
        >>> import numpy as np
        >>> import pandas_addons
        >>> sr = pd.Series([1, np.nan, np.nan, 8.3, np.nan])
        >>> sr.pctg.nans
        0.6

        Returns
        -------
        Series
            Return a copy of the Series
        """
        return self._obj.isna().sum() / len(self._obj)

    @property
    def uniques(self):
        """Get the percentage of number of uniques divided by
        the length of the series.

        Notes
        -----
        This is useful to check the cardinality of a column with
        respect to its length. If percentage of uniques is close
        to 1, it probably means this column does not follow a
        categorical distribution.

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series(["hey", "I", "just", "met", "you"])
        >>> sr.pctg.uniques
        1.0

        Returns
        -------
        Series
            Return a copy of the Series
        """
        return self._obj.nunique() / len(self._obj)


@pd.api.extensions.register_series_accessor("scaler")
class scaler:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")
        elif not is_numeric_dtype(_obj):
            raise ValueError("Series must be numeric")

    def standardize(self, ddof=1):
        """Standardize features by removing the mean and
        scaling to unit variance. Similar to scikit-learn's
        StandardScaler.

        Parameters
        ----------
        ddof : int, optional
            Degrees of freedom, by default 1

        Examples
        --------

            >>> import pandas as pd
            >>> import pandas_addons
            >>> sr = pd.Series([1,2,3,4,5])
            >>> sr.scaler.standardize()
            0   -1.264911
            1   -0.632456
            2    0.000000
            3    0.632456
            4    1.264911
            dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return (self._obj - self._obj.mean()) / self._obj.std(ddof=ddof)

    def minmax(self, feature_range=(0, 1)):
        """Transform series by scaling to a given range

        Parameters
        ----------
        feature_range : tuple, optional
            Desired range of transformed data, by default (0, 1)

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.minmax()
        0    0.00
        1    0.25
        2    0.50
        3    0.75
        4    1.00
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series.
        """
        feature_min, feature_max = feature_range

        max_val = max(self._obj)
        min_val = min(self._obj)

        std = (self._obj - min_val) / (max_val - min_val)
        scaled = std * (feature_max - feature_min) + feature_min

        return scaled

    def log1p(self):
        """Transform to log(1+x)

        Notes
        -----
        This transformation is numerically stable for small
        numbers compared to the log(x) transformation.

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.log1p()
        0    0.693147
        1    1.098612
        2    1.386294
        3    1.609438
        4    1.791759
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return np.log1p(self._obj)

    def expm1(self):
        """Transform to exp(x)-1

        Notes
        -----
        This transformation is numerically stable for small
        numbers compared to the exp(x) transformation.

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series([1,2,3,4,5])
        >>> sr.scaler.expm1()
        0      1.718282
        1      6.389056
        2     19.085537
        3     53.598150
        4    147.413159
        dtype: float64

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return np.expm1(self._obj)


@pd.api.extensions.register_series_accessor("ascii")
class ascii:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")

    def hist(self,
             size: int = 10,
             hashes: int = 30,
             len_label: int = 10,
             max_categories: int = 50):
        """Plots a horizontal histogram using :code:`#`

        Parameters
        ----------
        size : int, optional
            Size of bins, by default 10
        hashes : int, optional
            Maximum number of hashes :code:`#` to display on the
            the label with the highest frequency, by default 30
        len_label : int, optional
            Maximum length of the text label, by default 10
        max_categories : int, optional
            Maximum number of categories to display, by default 50

        Notes
        -----
        This would be useful if you want to get a quick sense of
        the distribution of your data or if you do not have access
        to say a Jupyter notebook. The API is deliberately named after
        the standard library's :code:`.hist()` API.

        Examples
        --------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series(["red", "blue", "red", "red", "orange", "blue"])
        >>> sr.ascii.hist()
               red ##############################
              blue ####################
            orange ##########
        """

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
            label = str(label)
            if len(label) > len_label:
                label = label[:(len_label-3)] + "..."
            else:
                str_format = ">" + str(len_label)
                label = format(label, str_format)
            print(label, int(count)*"#")


@pd.api.extensions.register_series_accessor("map_numerical_binning")
class map_numerical_binning:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")

    def __call__(self,
                 binning: Union[list, range, dict, int],
                 by_quantiles: bool = False,
                 ordered: bool = True):
        """Bin a numerical feature into groups. This is useful to transform
        a continuous variable to categorical.

        Parameters
        ----------
        binning : Union[list, range, dict, int]
            Criteria to bin by.
        by_quantiles : bool, optional
            If the :code:`binning` is by quantiles, by default False. This is only
            applicable if :code:`binning` is an integer.
        ordered : bool, optional
            Whether to treat the bins as ordinal variable, by default True

        Notes
        -----
        The underlying APIs are :code:`pandas.cut` and :code:`pandas.qcut`.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> import pandas_addons
        >>> sr = pd.Series([23, 94, 44, 95, 29, 8, 17, 42, 29, 48,
        ...                 96, 95, 17, 97, 9, 85, 62, 71, 37, 10,
        ...                 41, 88, 18, 56, 85, 22, 97, 27, 69, 19,
        ...                 37, 10, 85, 11, 73, 96, 56, 0, 18, 3,
        ...                 54, 50, 91, 38, 46, 13, 78, 22, 6, 61])

        Ranged binning using :code:`list` or :code:`range`

        >>> sr_cat = sr.map_numerical_binning([0, 18, 21, 25, 30, 100])
        >>> sr_cat.ascii.hist()
           (0, 18] ############
          (18, 21] #
          (21, 25] ###
          (25, 30] ###
         (30, 100] ##############################

        Ranged binning using :code:`dictionary`. Any number
        that is not in these ranges are considered null.

            - Kids: 0 < age <= 12
            - Teens: 12 < age <= 24
            - Adults: 24 < age <= 60

        >>> GROUPS = {
                "": 0,  # You must define this
                "kids": 12,
                "teens": 24,
                "adults": 60
            }
        >>> sr_bin_group = sr.map_numerical_binning(GROUPS)
        >>> sr_bin_group.ascii.hist()
              kids ##############
             teens ##################
            adults ##############################

        Binning with equal size range using :code:`int`. Below the size of
        each range label is about 25.

        >>> sr_bin = sr.map_numerical_binning(4)
        >>> sr_bin.ascii.hist(len_label=15)
        (-0.097, 24.25] ##############################
          (24.25, 48.5] ###################
          (48.5, 72.75] ##############
          (72.75, 97.0] ########################

        Binning by quantiles (equal frequencies) using :code:`int` and
        :code:`by_quantiles` keyword argument. The resulting distribution
        is close to a uniform distribution. Below we see the frequencies of
        each label (the hashes) is about 13.

        >>> sr_bin_quant = sr.map_numerical_binning(4, by_quantiles=True)
        >>> sr_bin_quant.ascii.hist(len_label=15)
        (-0.001, 18.25] ##############################
          (18.25, 43.0] ###########################
          (43.0, 76.75] ###########################
          (76.75, 97.0] ##############################

        Returns
        -------
        pandas.Series
            A transformed copy of the original series
        """

        return self._map_numerical_binning(
            binning, by_quantiles=by_quantiles, ordered=ordered)

    def _map_numerical_binning(self, binning, by_quantiles, ordered):
        if isinstance(binning, tuple):
            _, quantiles = binning
            sr = pd.qcut(self._obj, quantiles)
        elif isinstance(binning, (list, range)):
            sr = pd.cut(self._obj, binning)
        elif isinstance(binning, int):
            if by_quantiles:
                sr = pd.qcut(self._obj, binning)
            else:
                sr = pd.cut(self._obj, binning)
        elif isinstance(binning, dict):
            labels, bins = binning.keys(), binning.values()
            labels, bins = list(labels)[1:], list(bins)
            sr = pd.cut(self._obj, bins=bins, labels=labels)
        else:
            raise NotImplementedError

        return sr


@pd.api.extensions.register_series_accessor("map_categorical_binning")
class map_categorical_binning:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")

    def __call__(self, binning: dict, ordered: bool = False):
        """Group categories into another set of categories.

        Parameters
        ----------
        binning : dict
            Mapping where the key is the name of the new category
            and the value is a list of the current categories.
        ordered : bool, optional
            Whether to use the order in the binning to represent
            the inherent order in the new categories, by default False

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
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

        Then create a mapping:

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

        Returns
        -------
        pandas.Series
            A transformed copy of the series
        """
        return self._map_categorical_binning(binning, ordered=ordered)

    def _map_categorical_binning(self, binning, ordered):
        binning = OrderedDict(binning)
        mapping = {old_cat: new_cat
                   for new_cat, old_cats in binning.items()
                   for old_cat in old_cats}

        sr = self._obj.map(mapping).astype(
            CategoricalDtype(binning.keys(), ordered=ordered))

        return sr
