import re
import warnings
from typing import Union, Dict, Tuple

import pandas as pd

TARGET_COL_NAME_SEPARATOR = "__"


@pd.api.extensions.register_dataframe_accessor("add_groupby_columns")
class add_groupby_columns:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, pandas_obj):
        cols_with_space = [col for col in pandas_obj.columns if " " in col]
        if len(cols_with_space) > 0:
            warnings.warn("Dataframe consists of column names with spaces. "
                          "Consider cleaning these up.")

    def __call__(self, **lambdas: Dict[str, Union[str, Tuple[str, str, str]]]) -> pd.DataFrame:
        """

        Before:

        >>> df["PassengerGroupSize"] = df.groupby('PassengerGroup')['PassengerNumber'].transform('count')

        After:

        >>> df.group_by(
        ...     PassengerGroupSize="count(PassengerNumber) BY PassengerGroup"
        ... )

        """
        df = self._obj.copy()

        # for target_col, groupby_tuple_expr in lambdas.items():
        #     transform_expr, groupby_col = groupby_tuple_expr
        #     matches = re.match(r"(\w+)\((\w+)\)", transform_expr)
        #     if not matches:
        #         raise ValueError("Invalid transform expression")
        #     transform, col_to_transform = matches.groups()
        #
        #     df[target_col] = df.groupby(groupby_col)[col_to_transform].transform(transform)

        for target_col, groupby_expr in lambdas.items():

            if isinstance(groupby_expr, str):
                matches = re.match(r"(\w+)\((\w+)\) BY (\w+)", groupby_expr)
                if not matches:
                    raise ValueError("Invalid transform expression")
                transform_function, col_to_transform, groupby_col = matches.groups()

            elif isinstance(groupby_expr, tuple):
                transform_function, col_to_transform, groupby_col = groupby_expr

            else:
                raise ValueError("Invalid transform expression")

            df[target_col] = df.groupby(groupby_col)[col_to_transform].transform(transform_function)

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
