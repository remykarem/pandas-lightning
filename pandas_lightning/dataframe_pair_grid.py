from __future__ import annotations
import warnings
from typing import Optional, List, Callable

import pandas as pd
import seaborn as sns
from pandas.core.dtypes.common import is_categorical_dtype, is_numeric_dtype

TARGET_COL_NAME_SEPARATOR = "__"


@pd.api.extensions.register_dataframe_accessor("pair_grid")
class pair_grid:

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

    def __call__(self,
                 numeric_columns: Optional[List[str]] = None,
                 hue_dim: Optional[str] = None,
                 height: float = 3,
                 aspect: float = 1.3,
                 ) -> pair_grid:
        """
        https://seaborn.pydata.org/tutorial/axis_grids.html

        Args:
            row_dim: The name of the row category
            col_dim: The name of the column category
            height: The height of every plot
            aspect: Higher values increase the width of every plot
            **kwargs: Other values to pass to sns.FacetGrid

        Returns: FacetGrid object

        """
        if numeric_columns and not all(map(lambda col: is_numeric_dtype(self._obj[col]), numeric_columns)):
            raise ValueError(f"Some columns in {numeric_columns} are not numeric.")

        if hue_dim and not is_categorical_dtype(self._obj[hue_dim]):
            raise ValueError(f"{hue_dim} is not a category. Cast it as .astype('category') first.")

        if numeric_columns:
            self.grid = sns.PairGrid(self._obj[numeric_columns+[hue_dim]], hue=hue_dim, height=height, aspect=aspect)
        else:
            self.grid = sns.PairGrid(self._obj, hue=hue_dim, height=height, aspect=aspect)

        self.hue_dim = hue_dim
        return self

    def plot(self,
             upper: Optional[Callable] = None,
             lower: Optional[Callable] = None,
             diag: Optional[Callable] = None,
        ):
        if upper:
            self.grid.map_upper(upper)
        if lower:
            self.grid.map_lower(lower)
        if diag:
            self.grid.map_diag(diag)
        if self.hue_dim:
            self.grid.add_legend()

    def histplot(self, *args, **kwargs):
        self.grid.map(sns.histplot, *args, **kwargs)
        self.grid.add_legend()

    def barplot(self, *args, **kwargs):
        self.grid.map(sns.barplot, *args, **kwargs)
        self.grid.add_legend()

    def heatmap(self, *args, **kwargs):
        self.grid.map(sns.heatmap, *args, **kwargs)
        self.grid.add_legend()

    def kdeplot(self, *args, **kwargs):
        self.grid.map(sns.kdeplot, *args, **kwargs)
        self.grid.add_legend()

    def displot(self, *args, **kwargs):
        self.grid.map(sns.displot, *args, **kwargs)
        self.grid.add_legend()

    def countplot(self, *args, **kwargs):
        self.grid.map(sns.countplot, *args, **kwargs)
        self.grid.add_legend()

    def scatterplot(self, *args, **kwargs):
        self.grid.map(sns.scatterplot, *args, **kwargs)
        self.grid.add_legend()

    def lineplot(self, *args, **kwargs):
        self.grid.map(sns.lineplot, *args, **kwargs)
        self.grid.add_legend()

    def boxplot(self, *args, **kwargs):
        self.grid.map(sns.boxplot, *args, **kwargs)
        self.grid.add_legend()

    def violinplot(self, *args, **kwargs):
        self.grid.map(sns.violinplot, *args, **kwargs)
        self.grid.add_legend()

    def stripplot(self, *args, **kwargs):
        self.grid.map(sns.stripplot, *args, **kwargs)
        self.grid.add_legend()

    def catplot(self, *args, **kwargs):
        self.grid.map(sns.catplot, *args, **kwargs)
        self.grid.add_legend()

    def pairplot(self, *args, **kwargs):
        self.grid.map(sns.pairplot, *args, **kwargs)
        self.grid.add_legend()
