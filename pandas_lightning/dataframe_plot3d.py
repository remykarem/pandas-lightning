from __future__ import annotations
import warnings
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

TARGET_COL_NAME_SEPARATOR = "__"


@pd.api.extensions.register_dataframe_accessor("plot3d")
class plot3d:

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj
        self.grid: Optional[sns.PairGrid] = None
        self.hue_dim: Optional[str] = None

    def _validate_obj(self, pandas_obj: pd.DataFrame):
        cols_with_space = [col for col in pandas_obj.columns if " " in col]
        if len(cols_with_space) > 0:
            warnings.warn("Dataframe consists of column names with spaces. "
                          "Consider cleaning these up.")

    def __call__(self, x: str, y: str, z: str, hue: str):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.set_size_inches(18.5, 10.5)

        data = self._obj.dropna()

        ax.scatter3D(
            data=data,
            xs=data[x],
            ys=data[y],
            zs=data[z],
            s=80,
            c=data[hue].cat.codes,
            cmap=sns.color_palette("Set2", n_colors=len(data[hue].cat.categories), as_cmap=True)
        )
