import re
import warnings
from typing import Union

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype


@pd.api.extensions.register_dataframe_accessor("cast")
class cast:

    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj
        self.inplace = False

    def _validate_obj(self, pandas_obj):
        cols_with_space = [col for col in pandas_obj.columns if " " in col]
        if len(cols_with_space) > 0:
            warnings.warn("Dataframe consists of column names with spaces. "
                          "Consider cleaning these up.")

    def __call__(self, **dtypes: Union[type, str]) -> pd.DataFrame:
        """Convert dtypes of multiple columns using a dictionary

        Parameters
        ----------
        dtypes : dict
            Column name to data type mapping

        Notes
        -----
        You can also specify `"index"` and `"datetime"` on a column.
        Note that pandas does not have support for converting columns with NaNs
        to integer type. We will convert it to float automatically and indicate
        the user with a warning.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html

        Example
        -------
        Suppose we have a dataframe

        >>> import pandas as pd
        >>> from pandas.api.types import CategoricalDtype
        >>> import pandas_lightning
        >>> df = pd.DataFrame({
        ...     "X": list("ABACBB"),
        ...     "Y": list("121092"),
        ...     "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })
        >>> df
           X  Y     Z
        0  A  1   hot
        1  B  2  warm
        2  A  1   hot
        3  C  0  cold
        4  B  9  cold
        5  B  2   hot

        Change the types of the columns by writing

        >>> df = df.cast(
        ...     X="category",  # this will be nominal
        ...     Y=int,
        ...     Z=["cold", "warm", "hot"],  # this will be ordinal
        ... )

        which is equivalent to

        >>> df["X"] = df["X"].astype("category")
        >>> df["Y"] = df["Y"].astype(int)
        >>> df["Z"] = df["Z"].astype(CategoricalDtype(
        ...                 ["cold", "warm", "hot"], ordered=True))

        Returns
        -------
        pandas.DataFrame
            A dataframe whose columns have been converted accordingly
        """
        df = self._obj if self.inplace else self._obj.copy()

        for col, dtype in dtypes.items():

            # Check the value
            if isinstance(dtype, tuple):
                col_new = col
                col_old, dtype = dtype
            else:
                col_new, col_old = col, col

            # Check the dtype definition
            if isinstance(dtype, type):
                if dtype not in [int, float, bool, str,
                                 np.int8, np.int16, np.int32, np.int64,
                                 np.uint8, np.uint16, np.uint32, np.uint64]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, str) and dtype.startswith("timedelta"):
                pass
            elif isinstance(dtype, str):
                if dtype not in ["datetime", "index", "category", "int", "float", "bool", "str", "object", "string",
                                 "int8", "int16", "int32", "int64",
                                 "uint8", "uint16", "uint32", "uint64"]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, list):
                dtype = CategoricalDtype(dtype, ordered=True)
            elif isinstance(dtype, set):
                dtype = CategoricalDtype(dtype, ordered=False)
            elif callable(dtype):
                sort_fn = dtype
                uniques = df[col_old].unique().tolist()
                uniques.sort(key=sort_fn)
                dtype = CategoricalDtype(uniques, ordered=True)
            else:
                raise ValueError("Wrong type")

            # TODO strings are problematic across versions 😰

            # Handle pandas gotchas 😰
            if (dtype == int or dtype == "int") and df[col_old].hasnans:
                raise TypeError(
                    "Cannot convert non-finite values (NA or inf) to integer")
            elif (dtype == bool or dtype == "bool") and df[col_old].hasnans:
                raise TypeError("Casting to bool converts NaNs to True. "
                                f"There are NaNs in {col_old}. Cast to `float` instead.")

            # Cast type
            if dtype == "index":
                # df = df.set_index(col_old)
                df.set_index(col_old, inplace=True)
            elif dtype == "datetime":
                df[col_new] = pd.to_datetime(df[col_old])
            elif isinstance(dtype, str) and dtype.startswith("timedelta"):
                # * 'W'
                # * 'D' / 'days' / 'day'
                # * 'hours' / 'hour' / 'hr' / 'h'
                # * 'm' / 'minute' / 'min' / 'minutes' / 'T'
                # * 'S' / 'seconds' / 'sec' / 'second'
                # * 'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis' / 'L'
                # * 'us' / 'microseconds' / 'microsecond' / 'micro' / 'micros' / 'U'
                # * 'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'
                regex_grouped = re.search(r"\[(.*)\]", dtype)
                if regex_grouped is None:
                    raise ValueError(
                        "datetime should be in the format `timedelta[...]`")
                unit = regex_grouped.group(1)
                df[col_new] = pd.to_timedelta(df[col_old], unit=unit)
            else:
                df[col_new] = df[col_old].astype(dtype)

        return df
