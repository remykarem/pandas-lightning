import warnings
from typing import Union

import pandas as pd

TARGET_COL_NAME_SEPARATOR = "__"


@pd.api.extensions.register_dataframe_accessor("transform_columns")
class transform_columns:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj
        self.inplace = False

    def _validate_obj(self, pandas_obj):
        cols_with_space = [col for col in pandas_obj.columns if " " in col]
        if len(cols_with_space) > 0:
            warnings.warn("Dataframe consists of column names with spaces. "
                          "Consider cleaning these up.")

    def __call__(self, **lambdas) -> pd.DataFrame:
        """Perform multiple lambda operations on series

        Parameters
        ----------
        lambdas : dict
            A dictionary of column name to lambda mapping
        inplace : bool, optional
            Whether to modify the series inplace, by default False

        Examples
        --------
        >>> import pandas as pd
        >>> import pandas_lightning
        >>> df = pd.DataFrame({"X": list("ABACBB"),
        ...                    "Y": list("121092"),
        ...                    "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })
        >>> df = df.cast(
        ...     Y=int,
        ...     Z=["cold", "warm", "hot"])

        **Example 1**

            Rewriting to the same column

            >>> df = df.transform_columns(
            ...     X=lambda s: s + "rea",
            ...     Y=lambda s: s+100,
            ...     Z=lambda s: s.str.upper())

            which is the same as

            >>> df["X"] = df["X"] + "rea"
            >>> df["Y"] = df["Y"] + 100
            >>> df["Z"] = df["Z"].str.upper()

        **Example 2**

            Rewriting to the another column

            >>> df = df.transform_columns(
            ...     X_new=("X", lambda s: s + "rea"),
            ...     Y_new=("Y", lambda s: s+100),
            ...     Z_new=("Z", lambda s: s.str.upper())
            ... )

            which is the same as

            >>> df["X_new"] = df["X"] + "rea"
            >>> df["Y_new"] = df["Y"] + 100
            >>> df["Z_new"] = df["Z"].str.upper()

        **Example 3**

            Work with more than 1 column at a time

            >>> df.transform_columns(
            ...     XY=(["X", "Y"],
            ...         lambda x, y: x + (y+10).astype(str),
            ...     YZ=(["Y", "Z"],
            ...         lambda y, z: z.astype(str) + "-" + y.astype(str),
            ... )

            which is the same as

            >>> df["XY"] = df["X"] + df["Y"].astype(str)
            >>> df["YZ"] = df["Y"].astype(str) + "-" + df["Z"].astype(str)

        Returns
        -------
        pandas.DataFrame
            A transformed copy of the dataframe

        Raises
        ------
        ValueError
            If lambdas is not a dict
        ValueError
            [description]
        ValueError
            [description]
        """
        df = self._obj if self.inplace else self._obj.copy()

        def __sapply(data):

            for col, function in lambdas.items():

                # Unpack dictionary value
                if callable(function):
                    # 1-to-1
                    col_new, col_old = col, col
                elif isinstance(function, tuple) and len(function) == 1:
                    # 1-to-1
                    col_new, col_old = col, col
                    function = function[0]
                elif isinstance(function, tuple) and len(function) == 2:
                    # many-to-1
                    col_new = col
                    col_old, function = function
                else:
                    raise ValueError("Wrong type specified")

                if isinstance(col_old, (list, tuple)):
                    # many-to-1
                    series = [getattr(data, col) for col in col_old]
                    new_data = function(*series)
                else:
                    # 1-to-1
                    new_data = function(data[col_old])

                # ?-to-many
                new_col_names = parse_col_name_for_dunders(col_new)
                assert len(new_col_names) == get_num_of_columns(new_data), \
                    f"No. of new columns created ({get_num_of_columns(new_data)}) does " \
                    f"not match the {len(new_col_names)} target column names " \
                    f"{new_col_names}. Note that multiple target column names " \
                    f"are specified by '{TARGET_COL_NAME_SEPARATOR}'."

                if len(new_col_names) == 1:
                    data[new_col_names[0]] = new_data
                else:
                    data[new_col_names] = new_data

            return data

        df = __sapply(df)

        return df


def parse_col_name_for_dunders(col_name: str) -> list:
    return col_name.split(TARGET_COL_NAME_SEPARATOR)


def get_num_of_columns(obj: Union[pd.Series, pd.DataFrame, list]) -> int:
    if isinstance(obj, pd.Series):
        return 1
    elif isinstance(obj, list):
        return len(obj[0])
    else:
        return obj.shape[-1]
