import pandas as pd


@pd.api.extensions.register_dataframe_accessor("dataset")
class dataset:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def undersample(self, col, replace=False, min_count=None, random_state=None):
        df = self._obj.copy()

        if min_count is None:
            min_count = df[col].value_counts().min()

        dataframes = []
        for _, group in df.groupby(col):
            dataframes.append(group.sample(
                min_count, replace=replace, random_state=random_state))
        df_new = pd.concat(dataframes)

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

        return df_new
