import pandas as pd
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
        >>> import pandas_lightning
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
        >>> import pandas_lightning
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
        >>> import pandas_lightning
        >>> sr = pd.Series(["hey", "I", "just", "met", "you"])
        >>> sr.pctg.uniques
        1.0

        Returns
        -------
        Series
            Return a copy of the Series
        """
        return self._obj.nunique() / len(self._obj)
