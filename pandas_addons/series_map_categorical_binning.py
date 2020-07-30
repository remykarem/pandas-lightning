from collections import OrderedDict

import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.errors import EmptyDataError


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
