import pandas as pd
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_categorical_dtype


@pd.api.extensions.register_dataframe_accessor("dataset")
class dataset:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.history = None
        self._params = {}

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
