from functools import reduce
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns


@pd.api.extensions.register_dataframe_accessor("lambdas")
class lambdas:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def drop_columns_with_rules(self, *functions):
        """Drop a column if any of the conditions defined in the
        functions or lambdas are met

        Parameters
        ----------
        functions : functions or lambdas
            Functions or lambdas that take in a series as a parameter
            and returns a :code:`bool`

        Examples
        --------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> import numpy as np
        >>> df = pd.DataFrame({"X": [np.nan, np.nan, np.nan, np.nan, "hey"],
        ...                    "Y": [0, np.nan, 0, 0, 1],
        ...                    "Z": [1, 9, 5, 4, 2]})

        One of the more common patterns is dropping a column that
        has more than a certain threshold.

        >>> df.lambdas.drop_columns_with_rules(
        ...     lambda s: s.pctg.nans > 0.75,
        ...     lambda s: s.pctg.zeros > 0.5)
           Z
        0  1
        1  9
        2  5
        3  4
        4  2

        Returns
        -------
        pandas.DataFrame
            Dataframe with dropped columns
        """
        df = self._obj.copy()

        cols_to_drop = []
        for col_name in df:
            for f in functions:
                if f(df[col_name]):
                    cols_to_drop.append(col_name)
                    break

        df = df.drop(columns=cols_to_drop)

        return df

    def pipeline(self, *functions):
        """Apply a sequence of functions on this dataframe.

        Parameters
        ----------
        functions : lambdas or functions
            Each function must return a dataframe object

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> df = pd.DataFrame({"X": list("ABACBB"),
        ...                    "Y": list("121092"),
        ...                    "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })

        Define some functions

        >>> def drop_some_columns(data):
        ...     ...
        ...     return data
        >>> def reindex(data):
        ...     ...
        ...     return data
        >>> def rename_columns(data):
        ...     ...
        ...     return data

        Then

        >>> df = df.lambdas.pipeline({
        ...     drop_some_columns,
        ...     rename_columns,
        ...     reindex
        ... })

        which is the same as

        >>> df = drop_some_columns(df)
        >>> df = rename_columns(df)
        >>> df = reindex(df)

        Returns
        -------
        pandas.DataFrame
            Mutated dataframe
        """
        df = self._obj.copy()
        run_steps = reduce(lambda f, g: lambda x: g(f(x)),
                           functions,
                           lambda x: x)
        return run_steps(df)

    def sapply(self, lambdas: dict, inplace: bool = False):
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
        >>> import pandas_addons
        >>> df = pd.DataFrame({"X": list("ABACBB"),
        ...                    "Y": list("121092"),
        ...                    "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })
        >>> df = df.lambdas.astype({
        ...     "Y": int,
        ...     "Z": ["cold", "warm", "hot"]})

        **Example 1**

            Rewriting to the same column

            >>> df = df.lambdas.sapply({
            ...     "X": lambda s: s + "rea",
            ...     "Y": lambda s: s+100,
            ...     "Z": lambda s: s.str.upper()
            ... })

            which is the same as

            >>> df["X"] = df["X"] + "rea"
            >>> df["Y"] = df["Y"] + 100
            >>> df["Z"] = df["Z"].str.upper()

        **Example 2**

            Rewriting to the another column

            >>> df = df.lambdas.sapply({
            ...     ("X_new", "X"): lambda s: s + "rea",
            ...     ("Y_new", "Y"): lambda s: s+100,
            ...     ("Z_new", "Z"): lambda s: s.str.upper()
            ... })

            which is the same as

            >>> df["X_new"] = df["X"] + "rea"
            >>> df["Y_new"] = df["Y"] + 100
            >>> df["Z_new"] = df["Z"].str.upper()

        **Example 3**

            Rewriting to the multiple columns

            >>> df = df.lambdas.sapply({
            ...     (("Z_upper", "Z_lower"), "Z"): lambda z: (z.str.upper(), z.str.lower())
            ... })

            which is the same as

            >>> df["Z_upper"] = df["Z"].str.upper()
            >>> df["Z_lower"] = df["Z"].str.lower()

        **Example 4**

            Work with 2 columns at a time

            >>> df.lambdas.sapply({
            ...     ("XY", ("X", "Y")):
            ...     lambda x, y: x + (y+10).astype(str),
            ...     ("YZ", ("Y", "Z")):
            ...     lambda y, z: z.astype(str) + "-" + y.astype(str),
            ... })

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
        if not isinstance(lambdas, dict):
            raise ValueError("Must be dict")
        if inplace:
            df = self._obj
        else:
            df = self._obj.copy()

        for cols, function in lambdas.items():
            if len(cols) == 1 or isinstance(cols, str):
                # `("col"): ...`
                # `"col": ...`
                cols_new, cols_old = cols, cols
            elif len(cols) == 2:
                # `("col_b", ("col_a1","col_a2")): ...`
                # `("col_b", ["col_a1","col_a2"]): ...`
                # `("col_b", "col_a"): ...`
                cols_new, cols_old = cols
            else:
                raise ValueError("Wrong key")

            # breakpoint()

            if isinstance(cols_old, str) and isinstance(cols_new, str):
                # 1-to-1
                df[cols_new] = function(df[cols_old])
            elif isinstance(cols_old, (tuple, list)) and isinstance(cols_new, str):
                # many-to-1
                series = [getattr(df, col_old) for col_old in cols_old]
                df[cols_new] = function(*series)
            elif isinstance(cols_old, str) and isinstance(cols_new, (tuple, list)):
                multiple_series = function(df[cols_old])
                for col_new, series in zip(cols_new, multiple_series):
                    df[col_new] = series
            else:
                raise ValueError("Wrong type specified")

        return df

    def map_conditional(self, mappings, inplace=False):
        """
        Example
        -------
        >>> df.lambdas.map_conditional({
        ...     ("colour", ("A", "B")): {
        ...         "green": lambda a, b: a + b > 1,
        ...         "orange": lambda a, b: a == "Z",
        ...     }
        ... }, default="black")
        """
        df = self._obj if inplace else self._obj.copy()

        for cols, mapping in mappings.items():
            if len(cols) == 1 or isinstance(cols, str):
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")

            df[col_new] = df[col_old].map(mapping)

        return df if inplace else None

    def apply(self, lambdas: dict, inplace: bool = False):
        """Specify what functions to apply to every element
        in specified columns

        Parameters
        ----------
        lambdas : dict
            Mapping of column name to function
        inplace : bool, optional
            Whether to modify the series inplace, by default False

        Notes
        -----
        The underlying API is the :code:`pandas.Series.apply`.

        Example
        -------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> df = pd.DataFrame({"X": list("ABACBB"),
        ...                    "Y": list("121092"),
        ...                    "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })
        >>> df

        Returns
        -------
        pandas.DataFrame
            A dataframe whose columns have been converted accordingly
        """
        df = self._obj if inplace else self._obj.copy()

        for cols, function in lambdas.items():
            if len(cols) == 1 or isinstance(cols, str):
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")
            df[col_new] = df[col_old].apply(function)

        return df

    def fillna(self, d, inplace=False):
        """
        Example
        -------
        df.lambdas.fillna({
            "Sex": lambda sex: sex.median(),
            ("Age", ("Sex", "Pclass")): lambda group: group.median()
        })
        """
        df = self._obj if inplace else self._obj.copy()

        for cols, fill_value in d.items():
            if isinstance(cols, str):
                df[cols] = df[cols].fillna(fill_value)
            elif isinstance(cols, tuple) and len(cols) == 2:
                col, group = cols
                df[col] = df.groupby(list(group))[col].apply(
                    lambda x: x.fillna(fill_value(x)))
            else:
                raise ValueError("Wrong key")

        return df

    def astype(self, dtypes: dict):
        """Convert dtypes of multiple columns using a dictionary

        Parameters
        ----------
        dtypes : dict
            Column name to data type mapping

        Notes
        -----
        You can also specify `"index"` and `"datetime"` on a column.

        Example
        -------
        Suppose we have a dataframe

        >>> import pandas as pd
        >>> from pandas.api.types import CategoricalDtype
        >>> import pandas_addons
        >>> df = pd.DataFrame({"X": list("ABACBB"),
        ...                    "Y": list("121092"),
        ...                    "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })
        >>> df
           X  Y     Z
        0  A  1   hot
        1  B  2  warm
        2  A  1   hot
        3  C  0  cold
        4  B  9  cold
        5  B  2   hot

        You can perform the following

        >>> df["X"] = df["X"].astype("category")
        >>> df["Y"] = df["Y"].astype(int)
        >>> df["Z"] = df["Z"].astype(CategoricalDtype(
        ...                 ["cold", "warm", "hot"], ordered=True))

        with

        >>> df = df.lambdas.astype({
        ...     "X": "category",
        ...     "Y": int,
        ...     "Z": ["cold", "warm", "hot"]  # this will be ordinal
        ... })

        which is the same as

        >>> df = df.lambdas.astype({
        ...     "X": "category",
        ...     "Y": int,
        ...     "Z": CategoricalDtype(["cold", "warm", "hot"], ordered=True)
        ... })

        Returns
        -------
        pandas.DataFrame
            A dataframe whose columns have been converted accordingly
        """

        df = self._obj

        for cols, dtype in dtypes.items():

            # Check the key
            if isinstance(cols, str) or len(cols) == 1:
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")

            # Check the value
            if isinstance(dtype, type):
                if dtype.__name__ not in ["int", "float", "bool", "str"]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, str):
                if dtype not in ["datetime", "index", "category", "int", "float", "bool", "str"]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, list):
                dtype = CategoricalDtype(dtype, ordered=True)
            elif callable(dtype):
                sort_fn = dtype
                uniques = df[col_old].unique().tolist()
                uniques.sort(key=sort_fn)
                dtype = CategoricalDtype(uniques, ordered=True)
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError("Wrong type")

            # Set
            if dtype == "index":
                df = df.set_index(col_old)
            elif dtype == "datetime":
                df[col_new] = pd.to_datetime(df[col_old])
            else:
                df[col_new] = df[col_old].astype(dtype)

        return df

    def info(self, mode="df"):
        """
        TODO
        put this method elsewhere
        """
        df = self._obj
        missing = [df[col].pctg.nans for col in df]
        zeros = [df[col].pctg.zeros for col in df]
        uniques = [df[col].pctg.uniques for col in df]

        info = pd.DataFrame({
            "missing": missing,
            "zeros": zeros,
            "uniques": uniques
        }, index=df.columns)

        if mode == "df":
            return info
        elif mode == "heatmap":
            sns.heatmap(info)
        else:
            raise ValueError("mode must be df or heatmap")
