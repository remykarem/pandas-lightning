import warnings
import pandas as pd
import numpy as np
from pandas.api.types import is_bool_dtype, is_integer_dtype
from pandas.api.types import is_categorical_dtype, CategoricalDtype
from pandas import IntervalIndex


@pd.api.extensions.register_dataframe_accessor("to_X_y")
class to_X_y:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def to_X_y(self,
               *,
               target: str,
               nominal: str,
               nominal_max_cardinality: int,
               nans: str) -> pd.DataFrame:
        """Change everything to a numeric type

        Args:
            target (str): Column name of target variable (for regression or classification)
            nominal (str): Strategy to deal with nominal columns. One of 'one-hot',
                'label', 'keep' or 'drop'
            nans (str): Strategy to deal with missing values. One of 'remove' or 'keep'.

        Raises:
            KeyError: [description]
            ValueError: [description]

        Returns:
            pd.DataFrame: A dataframe fit for modelling
        """
        df = self._obj.copy()

        if target and target not in df:
            raise KeyError(f"{target} is not found in the DataFrame")

        # TODO check if column names are unique

        removed_categories = []

        # 1. Remove `object` and `datetime` types as we will not handle
        # them. User is responsible to change to category.
        num_cols_before = df.shape[-1]
        removed_categories.extend(df.select_dtypes(include=["object", "datetime"]).columns.tolist())
        df = df.select_dtypes(exclude=["object", "datetime"])
        num_cols_after = df.shape[-1]
        if num_cols_after != num_cols_before:
            warnings.warn("Found `object` and/or `datetime` categories. "
                          "These categories are removed.")

        # 2. Remove Nans
        # Note: Categories that are label-encoded or one-hot encoded are -1 if
        # they are NaNs but not dropped
        if nans == "remove":
            df.dropna(axis=0, inplace=True)

        # 3. Handle ordinal categories and boolean categories
        ordinal_mappings = {}
        nominal_categories = []
        nominal_categories_high_cardinality = []
        ordinal_categories = []
        bool_categories = []
        numeric_categories = []
        for col in df:
            if is_ordinal(df[col]):
                ordinal_mappings[col] = dictionarize(df[col].cat.categories)
                df[col] = df[col].cat.codes
                ordinal_categories.append(col)
            elif is_nominal(df[col]):
                if len(df[col].cat.categories) <= nominal_max_cardinality:
                    nominal_categories.append(col)
                else:
                    nominal_categories_high_cardinality.append(col)
            elif is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
                bool_categories.append(col)
            else:
                numeric_categories.append(col)

        # 4. Handle nominal categories
        nominal_mappings = {}
        df.drop(columns=nominal_categories_high_cardinality, inplace=True)
        removed_categories.extend(nominal_categories_high_cardinality)
        
        if nominal == "one-hot":
            df = pd.get_dummies(df, columns=nominal_categories)
        elif nominal == "label":
            for col in nominal_categories:
                # nominal mappings are for string categories
                if is_integer_dtype(df[col].cat.categories.dtype):
                    continue
                uniques = df[col].unique().tolist()
                if np.nan in uniques:
                    uniques.remove(np.nan)
                df[col] = df[col].astype(CategoricalDtype(uniques))
                nominal_mappings[col] = dictionarize(df[col].cat.categories)
                df[col] = df[col].cat.codes
        elif nominal == "drop":
            df.drop(columns=nominal_categories, inplace=True)
            removed_categories.extend(nominal_categories)
        elif nominal == "keep":
            pass
        else:
            raise ValueError(
                "`nominal` must be one of 'one-hot', 'label', 'keep' or 'drop'")

        # For one-hot encoding, this is the `nominal_category` is the prefix.
        # User is responsible to name columns to handle this
        metadata = {}
        if nominal != "drop":
            metadata["nominal"] = {
                "col_names": nominal_categories,
                "col_indices": None,
                "one-hot": nominal == "one-hot",
                "label_mappings": nominal_mappings
            }
        metadata["ordinal"] = {
            "col_names": ordinal_categories,
            "label_mappings": ordinal_mappings
        }
        metadata["bool"] = {
            "col_names": bool_categories
        }
        metadata["numeric"] = {
            "col_names": numeric_categories
        }
        metadata["removed"] = {
            "col_names": removed_categories
        }

        # Rearrange columns such that nominal categories are
        # at the back for easy slicing access
        if nominal in ["label", "keep"]:
            df = df[numeric_categories + ordinal_categories +
                    bool_categories + nominal_categories]
            last_n = len(nominal_categories) if target is None else \
                len(nominal_categories) - (target in nominal_categories)
            metadata["nominal"]["col_indices"] = list(
                range(len(df.columns)-int(bool(target))))[-last_n:]

        final_col_names = df.columns.tolist()

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


def is_ordinal(series: pd.Series) -> bool:
    return is_categorical_dtype(series) and series.dtype.ordered


def is_nominal(series: pd.Series) -> bool:
    return is_categorical_dtype(series) and not series.dtype.ordered
