import re
import warnings
from functools import reduce

import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype


@pd.api.extensions.register_dataframe_accessor("lambdas")
class lambdas:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj
        self._pipelines = None

    def _validate_obj(self, pandas_obj):
        cols_with_space = [col for col in pandas_obj.columns if " " in col]
        if len(cols_with_space) > 0:
            warnings.warn("Dataframe consists of column names with spaces. "
                          "Consider cleaning these up.")

    def __call__(self, pipelines: list = None):
        # Warning: `self._pipelines` is mutable by design
        if not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._pipelines = pipelines
        return self

    def astype(self, **dtypes):
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

        Change the types of the columns by writing

        >>> df = df.lambdas.astype({
        ...     X="category",  # this will be nominal
        ...     # X={"cold", "warm", "hot"}
        ...     Y=int,
        ...     Z=["cold", "warm", "hot"]  # this will be ordinal
        ... })

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
        df = self._obj.copy()

        for col, dtype in dtypes.items():

            # Check the value
            if isinstance(dtype, tuple):
                col_new = col
                col_old, dtype = dtype
            else:
                col_new, col_old = col, col

            # Check the dtype definition
            if isinstance(dtype, type):
                # TODO add numpy types
                if dtype.__name__ not in ["int", "float", "bool", "str"]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, str) and dtype.startswith("timedelta"):
                pass
            elif isinstance(dtype, str):
                if dtype not in ["datetime", "index", "category", "int", "float", "bool", "str", "object", "string"]:
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
                    raise ValueError("datetime should be in the format `timedelta[...]`")
                unit = regex_grouped.group(1)
                df[col_new] = pd.to_timedelta(df[col_old], unit=unit)
            else:
                df[col_new] = df[col_old].astype(dtype)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "astype"): dtypes})

        return df

    def sapply(self, **lambdas):
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
        ...     Y=int,
        ...     Z=["cold", "warm", "hot"]})

        **Example 1**

            Rewriting to the same column

            >>> df = df.lambdas.sapply(
            ...     X=lambda s: s + "rea",
            ...     Y=lambda s: s+100,
            ...     Z=lambda s: s.str.upper())

            which is the same as

            >>> df["X"] = df["X"] + "rea"
            >>> df["Y"] = df["Y"] + 100
            >>> df["Z"] = df["Z"].str.upper()

        **Example 2**

            Rewriting to the another column

            >>> df = df.lambdas.sapply(
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

            >>> df.lambdas.sapply(
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
        df = self._obj.copy()

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
                series = [getattr(df, col) for col in col_old]
                df[col_new] = function(*series)
            else:
                # 1-to-1
                df[col_new] = function(df[col_old])

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "sapply"): lambdas})

        return df

    def dapply(self, *functions):
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

        >>> df = df.lambdas.dapply(
        ...     drop_some_columns,
        ...     rename_columns,
        ...     reindex
        ... )

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
        df = run_steps(df)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "dapply"): functions})

        return df

    def apply(self, **lambdas):
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
        df = self._obj.copy()

        for col, function in lambdas.items():

            if callable(function):
                col_new, col_old = col, col
            elif isinstance(function, tuple) and len(function) == 1:
                col_new, col_old = col, col
                function = function[0]
            elif isinstance(function, tuple) and len(function) == 2:
                col_new = col
                col_old, function = function
            else:
                raise ValueError("Values must be a callable or a tuple")

            df[col_new] = df[col_old].apply(function)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "apply"): lambdas})

        return df

    def setna(self, **conditions):
        """You would do this on an existing column

        Returns
        -------
        pandas.DataFrame
            A copy of the dataframe

        Raises
        ------
        ValueError
            If wrong type is specified
        """
        df = self._obj.copy()

        for col, condition in conditions.items():

            # Unpack dictionary value
            if callable(condition):
                # 1-to-1
                col_to_set, context_col = col, col
            elif isinstance(condition, tuple) and len(condition) == 1:
                # 1-to-1
                col_to_set, context_col = col, col
                condition = condition[0]
            elif isinstance(condition, tuple) and len(condition) == 2:
                # many-to-1
                col_to_set = col
                context_col, condition = condition
            else:
                raise ValueError("Wrong type specified")

            if isinstance(context_col, (list, tuple)):
                # many-to-1
                series = [getattr(df, col) for col in context_col]
            else:
                # 1-to-1
                series = [df[context_col]]

            df.loc[condition(*series), col_to_set] = np.nan

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "setna"): conditions})

        return df

    def fillna(self, **d):
        """
        Example
        -------
        df.lambdas.fillna({
            "Sex": lambda sex: sex.median(),
            ("Age", ("Sex", "Pclass")): lambda group: group.median()
        })
        """
        df = self._obj.copy()

        for col, fill_value in d.items():

            # Unpack dictionary value
            if isinstance(fill_value, (int, float, str)):
                col_new, col_old = col, col
            elif callable(fill_value):
                col_new, col_old = col, col
            elif isinstance(fill_value, tuple) and len(fill_value) == 2:
                col_new = col
                col_old, fill_value = fill_value
            else:
                raise ValueError

            # Get fill value
            if callable(fill_value):
                # Get the series
                if isinstance(col_old, (list, tuple)):
                    series = [getattr(df, col) for col in col_old]
                else:
                    series = [df[col_old]]
                fill_value = fill_value(*series)

            # Fill na with fill value
            df[col_new] = df[col].fillna(fill_value)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "fillna"): d})

        return df

    def map_conditional(self, **mappings):
        """Map values from multiple columns based on conditional
        statements expressed as lambdas. Similar to `numpy.select`
        and `numpy.where`.

        You would use this for if-elif-else statements.
        If your statement is only an if-else and it is short,
        you would want to use `sapply` instead.

        Similar to pandas.Series.map.

        Parameters
        ----------
        lambdas : dict
            A nested dictionary where the key is the column names
            and the value is a dictionary of (newvalue: conditional statement)
            where the conditional statement is expressed as a lambda
            statement
        inplace : bool, optional
            Whether to modify the series inplace, by default False

        Example
        -------
        >>> df = pd.DataFrame({"X": [0, 5, 3, 3, 4, 1],
        ...                    "Y": [1, 2, 1, 0, 9, 2],
        ...                    "Z": ["hot","warm","hot","cold","cold","hot"]
        ... })

        Here is an example that maps the values across two series and creates
        a new series `W`.

        >>> df.lambdas.map_conditional(
        ...     W=(["X", "Y"], {
        ...         "green": lambda x, y: x + y > 1,
        ...         "orange": lambda x, y: x == 5,
        ...        }, "black")
        ... )
           X  Y     Z       W
        0  0  1   hot   black
        1  5  2  warm  orange
        2  3  1   hot   green
        3  3  0  cold   green
        4  4  9  cold  orange
        5  1  2   hot   black

        Here is an example that changes the value of Z. Note that this is a
        contrived example and you would want to use the pandas.Series.map API instead.

        >>> df.lambdas.map_conditional(
        ...     Z={"blue": lambda z: z == "cold",
        ...        "amber": lambda z: z == "warm",
        ...        "red": lambda z: z == "hot"})

        Returns
        -------
        pandas.DataFrame
            A transformed copy of the dataframe
        """
        df = self._obj.copy()

        default = None

        for col, definition in mappings.items():

            # Unpack dictionary value
            if isinstance(definition, dict):
                col_new, col_old = col, col
                mapping = definition
            elif isinstance(definition, tuple) and len(definition) == 2:
                col_new = col
                col_old, mapping = definition
            elif isinstance(definition, tuple) and len(definition) == 3:
                col_new = col
                col_old, mapping, default = definition
            else:
                raise ValueError

            # Get the series
            if isinstance(col_old, (list, tuple)):
                series = [getattr(df, col) for col in col_old]
            else:
                series = [df[col_old]]

            # Unpack mapping
            if not isinstance(mapping, dict):
                raise ValueError("Must be dictionary")

            # Use np.select
            choicelist, conditions = list(
                mapping.keys()), list(mapping.values())
            condlist = [cond(*series) for cond in conditions]
            df[col_new] = np.select(condlist=condlist, choicelist=choicelist,
                                    default=default)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "map_conditional"): mappings})

        return df

    def drop_if_exist(self, columns: list):
        df = self._obj

        columns_ = columns.copy()
        for i, col in enumerate(columns):
            if col not in df:
                columns_.pop(i)

        df = df.drop(columns=columns_)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("lambdas", "drop_if_exist"): columns})

        return df

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

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add(
                    {("lambdas", "drop_columns_with_rules"): functions})

        return df

    # def merge(self, ):
    #     df = self._obj

    #     df.merge(df_users, on="user_id", how="left").merge(df_email, on="grass_date", how="left")

    #     df.lambdas.merge({
    #         ("left", "user_id"): df_users,
    #         ("left", "grass_date"): df_email
    #     })
