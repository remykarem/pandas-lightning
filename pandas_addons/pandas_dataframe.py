import math
from functools import reduce
from itertools import combinations

import pandas as pd
from pandas import CategoricalDtype
from pandas.api.types import is_datetime64_any_dtype, is_bool_dtype
from pandas.api.types import is_categorical_dtype


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

        >>> df = df.lambdas.dapply({
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

    def map_numerical_binning(self, bins, ordered=True, inplace=False):
        """
        Examples
        --------

        Ranged binning (list or range)

        >>> df.pipeline.map_numerical_binning({
        ...     "age": [0,18,21,25,30,100]
        ... })

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
        df = self._obj if inplace else self._obj.copy()

        for cols, binning in bins.items():
            if len(cols) == 1 or isinstance(cols, str):
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")

            if isinstance(binning, tuple):
                _, quantiles = binning
                df[col_new] = pd.qcut(df[col_old], quantiles)
            elif isinstance(binning, (list, range, int)):
                df[col_new] = pd.cut(df[col_old], binning)
            else:
                raise NotImplementedError

        return df

    def map_categorical_binning(self, bins, ordered=False, inplace=False):
        """Categorical binning

        Args:
            bins (dict): mappings
            ordered (bool, optional): [description]. Defaults to False.
            inplace (bool, optional): [description]. Defaults to False.

        Returns:
            pandas.Series: dfvdsv
        """
        df = self._obj if inplace else self._obj.copy()

        for cols, bin_ in bins.items():
            if len(cols) == 1 or isinstance(cols, str):
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")

            mapping = {v: k for k, values in bin_.items()
                       for v in values}

            df[col_new] = df[col_old].map(mapping).astype(
                CategoricalDtype(mapping.keys(), ordered=ordered))

        return df

    def map(self, mappings, inplace=False):
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
        df = self._obj if inplace else self._obj.copy()

        for cols, fill_value in d.items():
            if len(cols) == 1 or isinstance(cols, str):
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")

            df[col_new] = df[col_old].fillna(fill_value)

        return df

    def astype(self, dtypes: dict):
        """Convert dtypes of multiple columns using a dictionary

        Parameters
        ----------
        dtypes : dict
            Column name to data type mapping

        Notes
        -----
        The underlying API is :code:`pandas.Series.astype`.

        Example
        -------
        Suppose we have a dataframe

        >>> import pandas as pd
        >>> from pandas import CategoricalDtype
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

            # Check the value
            if isinstance(dtype, type):
                if dtype.__name__ not in ["int", "float", "bool", "str"]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, str):
                if dtype not in ["datetime", "index", "category", "int", "float", "bool", "str"]:
                    raise ValueError("Wrong type")
            elif isinstance(dtype, list):
                dtype = CategoricalDtype(dtype, ordered=True)
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError("Wrong type")

            # Check the key
            if isinstance(cols, str) or len(cols) == 1:
                col_new, col_old = cols, cols
            elif len(cols) == 2:
                col_new, col_old = cols
            else:
                raise ValueError("Wrong key")

            # Set
            if dtype == "index":
                df = df.set_index(col_old)
            elif dtype == "datetime":
                df[col_new] = pd.to_datetime(df[col_old])
            else:
                df[col_new] = df[col_old].astype(dtype)

        return df

    def to_numerics(self,
                    target: str = None,
                    one_hot: bool = True,
                    to_numpy: bool = True,
                    inplace: bool = False,
                    remove_missing: bool = True):
        df = self._obj if inplace else self._obj.copy()

        # 1. Remove non-numerics
        # User is responsible to change to category
        df = df.select_dtypes(exclude=["object", "datetime"])

        # 2. Remove Nans
        if remove_missing:
            df = df.dropna(axis=0)

        # 3. Deal with ordinal categories and boolean categories
        # Keep track of nominals
        ordinal_mappings = {}
        nominal_categories = []
        for col in df:
            if is_categorical_dtype(df[col]) and df[col].dtype.ordered:
                ordinal_mappings[col] = dict(enumerate(df[col].cat.categories))
                df[col] = df[col].cat.codes
            elif is_categorical_dtype(df[col]) and not df[col].dtype.ordered:
                nominal_categories.append(col)
            elif is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)

        # 4. Deal with nominal categories
        if one_hot:
            df = pd.get_dummies(df, columns=nominal_categories)
        else:
            df = df.drop(columns=nominal_categories)

        df_X = df.loc[:, set(df.columns) - set([target])]

        # 5. Whether to convert to numpy
        if to_numpy:
            if target:
                return df_X.values, df[target].values, \
                    df_X.columns, [target], ordinal_mappings
            else:
                return df.values, df.columns, ordinal_mappings
        else:
            if target:
                return df_X, df[target], ordinal_mappings
            else:
                return df, ordinal_mappings


@pd.api.extensions.register_dataframe_accessor("optimize")
class optimize:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.history = None
        self._params = {}

    def drop_duplicate_columns(self, inplace: bool = False):
        """Drop duplicate columns that have exactly the same
        values and datatype

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform inplace operation, by default False

        Returns
        -------
        pandas.DataFrame
            Dataframe with no duplicate columns
        """
        to_drop = []
        pairs = combinations(self._obj.columns, 2)

        num_combinations = math.comb(len(self._obj.columns), 2)
        print(f"Checking {num_combinations} combinations")

        for pair in pairs:
            col_a, col_b = pair
            if col_a in to_drop or col_b in to_drop:
                continue
            if self._obj[col_a].equals(self._obj[col_b]):
                to_drop.append(col_b)

        print(f"Duplicate columns: {to_drop}")

        return self._obj.drop(columns=to_drop, inplace=inplace)

    def convert_categories(self, max_cardinality: int = 20,
                           inplace: bool = False):
        """Convert columns to category whenever possible

        Parameters
        ----------
        max_cardinality : int, optional
            The maximum no. of uniques before a column can be converted
            to a category type, by default 20
        inplace : bool, optional
            [description], by default False

        Returns
        -------
        pandas.DataFrame
            A transformed dataframe
        """
        if inplace:
            df = self._obj
        else:
            df = self._obj.copy()

        for col in df:
            if not df[col].dtype == "object":
                continue

            num_uniques = df[col].nunique()

            if num_uniques == 1:
                # Drop this column!
                print(f"{col} has only 1 value.")
            elif num_uniques <= max_cardinality:
                df[col] = df[col].astype("category")

        if inplace:
            return None
        else:
            return df

    def profile(self, dry_run=True, max_cardinality=20):

        befores = 0
        afters = 0

        for col in self._obj:

            num_uniques = self._obj[col].nunique()
            before = self._obj[col].memory_usage(index=False)
            befores += before

            if num_uniques == 1:
                print(f"{col} has only 1 value. Drop to save {before}.")
            elif num_uniques == 2:
                if self._obj[col].dtype.name in ["category", "bool"]:
                    print(f"{col} looks good")
                    continue
                after = self._obj[col].astype(
                    "category").memory_usage(index=False)
                afters += after
                savings = (before-after)/before * 100

                # if containts yes-no or true-false or t-f or 1-0

                print(f"{col} can be optimised. "
                      "Consider bool or cat. "
                      f"Consider category. Save {savings:.0f}%")
            elif num_uniques == 3:
                if self._obj[col].dtype.name in ["category", "bool"]:
                    print(f"{col} looks good")
                    continue
                after = self._obj[col].astype(
                    "category").memory_usage(index=False)
                afters += after
                savings = (before-after)/before * 100

            elif num_uniques <= max_cardinality:
                # print(self._obj[col].memory_usage(index=False))
                if self._obj[col].dtype == "object":
                    print(f"{col} can be optimised. "
                          f"{num_uniques} uniques found. "
                          f"Consider category.")
                elif self._obj[col].dtype.name.startswith("int"):
                    after = self._obj[col].astype(
                        "category").memory_usage(index=False)
                    afters += after
                    savings = (before-after)/before * 100
                    print(f"{col} can be optimised. "
                          f"{num_uniques} uniques found. "
                          f"Consider category. Save {savings:.0f}%")
                elif self._obj[col].dtype.name in ["category", "bool"]:
                    print(f"{col} looks good")
                    continue
                else:
                    print(f"{col} not evaluated")
            else:
                print(f"{col} looks good")

        # print(f"Before: {befores/1e6:.1f}MB")
        # print(f"Total savings: {(befores-afters)/1e6:.1f}MB")
        print(f"Total savings: {afters/1e6:.1f}MB")
