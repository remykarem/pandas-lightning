from typing import Union

import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError


@pd.api.extensions.register_series_accessor("map_numerical_binning")
class map_numerical_binning:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")

    def __call__(self,
                 binning: Union[list, range, dict, int, tuple],
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
        >>> import pandas_lightning
        >>> sr = pd.Series([23, 94, 44, 95, 29, 8, 17, 42, 29, 48,
        ...                 96, 95, 17, 97, 9, 85, 62, 71, 37, 10,
        ...                 41, 88, 18, 56, 85, 22, 97, 27, 69, 19,
        ...                 37, 10, 85, 11, 73, 96, 56, 0, 18, 3,
        ...                 54, 50, 91, 38, 46, 13, 78, 22, 6, 61])

        Ranged binning using :code:`range`. Below is grouping in 10's.

        >>> sr_cat = sr.map_numerical_binning(range(0,110,10))  # or (0,110,10)
        >>> sr_cat.ascii.hist()
           (0, 10] ######################
          (10, 20] ##########################
          (20, 30] ######################
          (30, 40] ###########
          (40, 50] ######################
          (50, 60] ###########
          (60, 70] ###########
          (70, 80] ###########
          (80, 90] ###############
         (90, 100] ##############################

        Ranged binning using :code:`list`. Below is grouping using the elements
        in the array as the bounds.

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
            binning = np.arange(*binning)
            sr = pd.cut(self._obj, binning)
        elif isinstance(binning, (list, range)):
            sr = pd.cut(self._obj, binning)
        elif isinstance(binning, int):
            if by_quantiles:
                sr = pd.qcut(self._obj, binning, duplicates="drop")
            else:
                sr = pd.cut(self._obj, binning)
        elif isinstance(binning, dict):
            labels, bins = binning.keys(), binning.values()
            labels, bins = list(labels)[1:], list(bins)
            sr = pd.cut(self._obj, bins=bins, labels=labels)
        else:
            raise NotImplementedError

        return sr
