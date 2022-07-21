import re
import pytz
import warnings
from typing import Union

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype, is_object_dtype, is_numeric_dtype

DATETIME_UNITS = ["Y", "M", "D", "h", "m", "s", "ms", "us", "ns"]
EPOCH_UNIT_BY_NUM_DIGITS = {
    10: "s",
    13: "ms",
    16: "us",
    19: "ns",
}


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
                if dtype not in ["index", "category", "int", "float", "bool", "str", "object", "string",
                                 "int8", "int16", "int32", "int64",
                                 "uint8", "uint16", "uint32", "uint64"] and \
                        not dtype.startswith("datetime64["):
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
            elif isinstance(dtype, str) and dtype.startswith("datetime64"):
                df[col_new] = cast_series_as_datetime(series=df[col_old], datetime_target=dtype)
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


def parse_datetime_target(datetime_target: str) -> tuple:
    match = re.match(r"datetime64\[(\w+)(, ([^,]+))?(, ([^,]+))?\]", datetime_target)

    assert match, "Wrong format. Datetime target must be in the format " \
                  "'datetime64[<unit>, <timezone>, <format>]'"

    unit, _, timezone, _, fmt = match.groups()

    assert unit in DATETIME_UNITS, f"Invalid unit {unit} found. " \
                                   f"Available datetime units are {DATETIME_UNITS}"

    if timezone == "?":
        timezone = None
    if timezone:
        assert timezone in pytz.all_timezones, \
            f"Timezone {timezone} is invalid. See pytz.all_timezones for list " \
            "of available timezones"

    return unit, timezone, fmt


def cast_series_as_datetime(series: pd.Series, datetime_target: str) -> pd.Series:
    unit, timezone, fmt = parse_datetime_target(datetime_target)

    if is_numeric_dtype(series):
        # This is epoch
        lengths = series.astype(int).astype(str).str.len()

        assert len(lengths.unique()) == 1, \
            "This series has more than 1 Unix epoch precision."
        assert lengths[0] in EPOCH_UNIT_BY_NUM_DIGITS, \
            "Format of Unix epoch timestamp is incorrect. The no. of digits is one of " \
            f"{list(EPOCH_UNIT_BY_NUM_DIGITS)}"

        s = series.astype(f"datetime64[{EPOCH_UNIT_BY_NUM_DIGITS[lengths[0]]}]")

    elif is_object_dtype(series):
        s = pd.to_datetime(series, utc=False, format=fmt)

    else:
        s = series

    if timezone:
        return s.astype(f"datetime64[{unit}]").dt.tz_localize(timezone)
    else:
        return s.astype(f"datetime64[{unit}]")
