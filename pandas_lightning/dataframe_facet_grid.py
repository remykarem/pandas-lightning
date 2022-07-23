from __future__ import annotations
import warnings
from typing import Optional

import pandas as pd
import seaborn as sns
from pandas.core.dtypes.common import is_categorical_dtype

TARGET_COL_NAME_SEPARATOR = "__"


@pd.api.extensions.register_dataframe_accessor("facet_grid")
class facet_grid:

    def __init__(self, pandas_obj: pd.DataFrame):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj
        self.grid: Optional[sns.FacetGrid] = None
        self.hue_dim: Optional[str] = None

    def _validate_obj(self, pandas_obj: pd.DataFrame):
        cols_with_space = [col for col in pandas_obj.columns if " " in col]
        if len(cols_with_space) > 0:
            warnings.warn("Dataframe consists of column names with spaces. "
                          "Consider cleaning these up.")

    def __call__(self,
                 row_dim: Optional[str] = None,
                 col_dim: Optional[str] = None,
                 hue_dim: Optional[str] = None,
                 height: float = 3,
                 aspect: float = 1.3,
                 **kwargs
                 ) -> facet_grid:
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
        if row_dim and not is_categorical_dtype(self._obj[row_dim]):
            raise ValueError(f"{row_dim} is not a category. Cast it as .astype('category') first.")
        if col_dim and not is_categorical_dtype(self._obj[col_dim]):
            raise ValueError(f"{col_dim} is not a category. Cast it as .astype('category') first.")
        if hue_dim and not is_categorical_dtype(self._obj[hue_dim]):
            raise ValueError(f"{hue_dim} is not a category. Cast it as .astype('category') first.")

        self.grid = sns.FacetGrid(
            self._obj,
            row=row_dim,
            col=col_dim,
            hue=hue_dim,
            height=height,
            aspect=aspect,
            margin_titles=True,
            **kwargs
        )
        if hue_dim:
            self.hue_dim = hue_dim
        return self

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
