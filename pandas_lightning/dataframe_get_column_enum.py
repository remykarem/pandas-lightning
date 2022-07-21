import pandas as pd


@pd.api.extensions.register_dataframe_accessor("get_column_enum")
class get_column_enum:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def __call__(self: pd.DataFrame) -> type:
        return type("Col", (), {col: col for col in self._obj})
