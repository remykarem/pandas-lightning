import pandas as pd
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_categorical_dtype, CategoricalDtype
from pandas import IntervalIndex


@pd.api.extensions.register_dataframe_accessor("dataset")
class dataset:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def to_numerics(self,
                    target: str = None,
                    nominal: str = "one-hot",
                    to_numpy: bool = True,
                    inplace: bool = False,
                    remove_missing: bool = True):
        df = self._obj if inplace else self._obj.copy()

        if target and target not in df:
            raise KeyError(f"{target} is not found in the DataFrame")

        # 1. Remove non-numerics
        # User is responsible to change to category
        df = df.select_dtypes(exclude=["object", "datetime"])

        # 2. Remove Nans
        # Note: Categories that are label-encoded or one-hot encoded are -1 if
        # they were NaNs but not dropped
        if remove_missing:
            df = df.dropna(axis=0)

        # 3. Deal with ordinal categories and boolean categories
        # Keep track of nominals
        ordinal_mappings = {}
        nominal_categories = []
        ordinal_categories = []
        bool_categories = []
        numeric_categories = []
        for col in df:
            if is_categorical_dtype(df[col]) and df[col].dtype.ordered:
                ordinal_mappings[col] = dictionarize(df[col].cat.categories)
                df[col] = df[col].cat.codes
                ordinal_categories.append(col)
            elif is_categorical_dtype(df[col]) and not df[col].dtype.ordered:
                nominal_categories.append(col)
            elif is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
                bool_categories.append(col)
            else:
                numeric_categories.append(col)

        # 4. Deal with nominal categories
        nominal_mappings = {}
        if nominal == "one-hot":
            df = pd.get_dummies(df, columns=nominal_categories)
        elif nominal == "drop":
            df = df.drop(columns=nominal_categories)
        elif nominal == "label":
            for col in nominal_categories:
                df[col] = df[col].astype(CategoricalDtype(df[col].unique()))
                nominal_mappings[col] = dictionarize(df[col].cat.categories)
                df[col] = df[col].cat.codes
        else:
            raise ValueError("`nominal` must be one of 'one-hot', 'drop' or "
                             "'label-encode'")

        metadata = {}
        metadata["nominal_category"] = nominal_categories
        metadata["ordinal_category"] = ordinal_categories
        metadata["bool_category"] = bool_categories
        metadata["numeric_category"] = numeric_categories
        metadata["nominal_category_mappings"] = nominal_mappings
        metadata["ordinal_category_mappings"] = ordinal_mappings
        if nominal == "label":
            df = df[numeric_categories + ordinal_categories +
                    bool_categories + nominal_categories]
            metadata["nominal_cat_in_last_x_columns"] = len(nominal_categories) if target is None else \
                len(nominal_categories) - (target in nominal_categories)

        # 5. Whether to convert to numpy
        if to_numpy:
            if target:
                y = df.pop(target)
                return df.values, y.values, metadata
            else:
                return df.values, metadata
        else:
            if target:
                y = df.pop(target)
                return df, y, metadata
            else:
                return df, metadata


def dictionarize(categories):
    if isinstance(categories, IntervalIndex):
        return {i: (cat.left, cat.right)
                for i, cat in enumerate(categories)}
    else:
        return dict(enumerate(categories))
