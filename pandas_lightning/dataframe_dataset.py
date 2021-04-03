import warnings
import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype
from pandas.api.types import is_categorical_dtype, CategoricalDtype
from pandas import IntervalIndex


@pd.api.extensions.register_dataframe_accessor("dataset")
class dataset:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._pipelines = None

    def __call__(self, pipelines: list = None):
        # Warning: `self._pipelines` is mutable by design
        if pipelines and not isinstance(pipelines, list):
            pipelines = [pipelines]

        self._pipelines = pipelines
        return self

    def undersample(self, col, replace=False, min_count=None, random_state=None):
        df = self._obj.copy()

        if min_count is None:
            min_count = df[col].value_counts().min()

        dataframes = []
        for _, group in df.groupby(col):
            dataframes.append(group.sample(
                min_count, replace=replace, random_state=random_state))
        df_new = pd.concat(dataframes)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("dataset", "undersample"): {
                    "col": col,
                    "replace": replace,
                    "min_count": min_count,
                    "random_state": random_state
                }})

        return df_new

    def oversample(self, col, max_count=None, random_state=None):
        """
        https://stackoverflow.com/questions/48373088/duplicating-training-examples-to-handle-class-imbalance-in-a-pandas-data-frame
        """
        df = self._obj.copy()

        if max_count is None:
            max_count = df[col].value_counts().max()

        dataframes = [df]
        for _, group in df.groupby(col):
            dataframes.append(group.sample(
                abs(max_count-len(group)), replace=True, random_state=random_state))
        df_new = pd.concat(dataframes)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("dataset", "oversample"): {
                    "col": col,
                    "max_count": max_count,
                    "random_state": random_state
                }})

        return df_new

    def to_X_y(self,
               target: str = None,
               nominal: str = "one-hot",
               remove_missing: bool = False):
        """
        Change everything to a numeric type
        """
        df = self._obj.copy()

        if target and target not in df:
            raise KeyError(f"{target} is not found in the DataFrame")

        # TODO check if column names are unique

        # 1. Remove `object` and `datetime` types as we will not handle
        # them. User is responsible to change to category.
        num_cols_before = df.shape[-1]
        df = df.select_dtypes(exclude=["object", "datetime"])
        num_cols_after = df.shape[-1]
        if num_cols_after != num_cols_before:
            warnings.warn("Found `object` and/or `datetime` categories. "
                          "These categories are removed.")

        # 2. Remove Nans
        # Note: Categories that are label-encoded or one-hot encoded are -1 if
        # they are NaNs but not dropped
        if remove_missing:
            df = df.dropna(axis=0, inplace=True)

        # 3. Handle ordinal categories and boolean categories
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

        # 4. Handle nominal categories
        nominal_mappings = {}
        if nominal == "one-hot":
            df = pd.get_dummies(df, columns=nominal_categories)
        elif nominal == "label":
            for col in nominal_categories:
                uniques = df[col].unique().tolist()
                if np.nan in uniques:
                    uniques.remove(np.nan)
                df[col] = df[col].astype(CategoricalDtype(uniques))
                nominal_mappings[col] = dictionarize(df[col].cat.categories)
                df[col] = df[col].cat.codes
        elif nominal == "drop":
            df = df.drop(columns=nominal_categories, inplace=True)
        else:
            raise ValueError("`nominal` must be one of 'one-hot', 'label', or 'drop'")

        # For one-hot encoding, this is the `nominal_category` is the prefix.
        # User is responsible to name columns to handle this
        metadata = {}
        metadata["nominal"] = nominal_categories
        metadata["nominal_mappings"] = nominal_mappings
        metadata["nominal_in_last_n_cols"] = None
        metadata["ordinal"] = ordinal_categories
        metadata["ordinal_mappings"] = ordinal_mappings
        metadata["bool"] = bool_categories
        metadata["numeric"] = numeric_categories

        # Rearrange columns such that nominal categories are
        # at the back for easy slicing access
        if nominal in ["label", "keep"]:
            df = df[numeric_categories + ordinal_categories +
                    bool_categories + nominal_categories]
            metadata["nominal_in_last_n_cols"] = len(nominal_categories) if target is None else \
                len(nominal_categories) - (target in nominal_categories)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("dataset", "to_X_y"): {
                    "target": target,
                    "nominal": nominal,
                    "remove_missing": remove_missing
                }})

        # 5. Separate target
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
