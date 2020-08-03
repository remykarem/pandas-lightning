import pandas as pd
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
        if not isinstance(pipelines, list):
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
                pipeline.add({("dataset","undersample"): {
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
                max_count-len(group), replace=True, random_state=random_state))
        df_new = pd.concat(dataframes)

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("dataset","oversample"): {
                    "col": col,
                    "max_count": max_count,
                    "random_state": random_state
                }})

        return df_new

    def to_numerics(self,
                    target: str = None,
                    nominal: str = "one-hot",
                    to_numpy: bool = False,
                    remove_missing: bool = True,
                    inplace: bool = False):
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
        elif nominal == "keep" and to_numpy is False:
            pass
        else:
            raise ValueError("`nominal` must be one of 'one-hot', 'drop' or "
                             "'label'")

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

        if self._pipelines is not None:
            for pipeline in self._pipelines:
                pipeline.add({("dataset","to_numerics"): {
                    "target": target,
                    "nominal": nominal,
                    "to_numpy": to_numpy,
                    "inplace": inplace,
                    "remove_missing": remove_missing
                }})

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
