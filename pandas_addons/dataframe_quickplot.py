from typing import Union
from collections import defaultdict
from functools import partial
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_bool_dtype
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("quickplot")
class quickplot:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.num1 = None
        self.num2 = None
        self.cat12 = None  # Up to 12 categories
        self.cat2 = None  # Up to  2 categories
        self.cat3 = None  # Up to  3 categories
        self.config = None

    def __repr__(self):
        return "hi"

    def __call__(self, / ,
                 numerical: Union[str, list] = None,
                 categorical: Union[str, list] = None):
        """
        1: numerical
        1: categorical
        2: numerical + numerical
        2: numerical + categorical

        3: categorical + categorical
        3: numerical + categorical + categorical
        3: numerical + numerical + categorical
        3: categorical + categorical + categorical

        ---
        4: numerical + categorical + categorical + categorical
        """

        if numerical is None:
            numerical = []
        if categorical is None:
            categorical = []
        if numerical is not None:
            if isinstance(numerical, str):
                numerical = [numerical]
            for col in numerical:
                if not is_numeric_dtype(self._obj[col]):
                    raise ValueError

        if categorical is not None:
            if isinstance(categorical, str):
                categorical = [categorical]
            for col in categorical:
                if not (is_categorical_dtype(self._obj[col]) or is_bool_dtype(self._obj[col])):
                    raise ValueError

        if len(categorical) == 0:
            config = (len(numerical), 0, 0)
        elif len(categorical) == 1:
            config = (len(numerical), 1, 0)
        elif len(categorical) == 2:
            config = (len(numerical), 1, 1)
        elif len(categorical) == 3:
            config = (len(numerical), 1, 2)
        else:
            raise ValueError("Max 3 categories only")

        if config == (1, 0, 0):
            print("histogram, boxplot, violinplot, stripplot, qqplot")
        elif config == (0, 1, 0):
            print("countplot")
        elif config == (0, 1, 1):
            print("countplot")
        elif config == (1, 1, 0):
            print("barplot, boxplot, violinplot, stripplot, ridgeplot")
        elif config == (2, 0, 0):
            print("lineplot, scatterplot, hexbinplot, kdeplot")
        elif config == (2, 1, 0):
            print("lineplot")
        elif config == (1, 1, 1):
            print("boxplot, violinplot, stripplot")
        elif config == (0, 1, 2):
            print("catplot")
        else:
            raise ValueError

        self.config = config
        self.numerical_ = numerical.copy()
        self.categorical_ = categorical.copy()

        return self

    def barplot(self):
        if self.config == (1, 1, 0):
            sns.barplot(x=self.categorical_[0], y=self.numerical_[0], data=df2)

    def histogram(self):
        if self.config == (1, 0, 0):
            sns.distplot(self._obj[self.numerical_[0]])

    def countplot(self):
        if self.config == (0, 1, 0):
            sns.countplot(y=self._obj[self.categorical_[0]], order=self._obj[self.categorical_[0]].cat.categories.tolist())
        if self.config == (0, 1, 1):
            sns.countplot(x=self.categorical_[0], hue=self.categorical_[1], data=self._obj)

    def scatterplot(self):
        if self.config == (2, 0, 0):
            sns.jointplot(x=self.numerical_[0], y=self.numerical_[
                          1], data=self._obj)

    def kdeplot(self):
        if self.config == (2, 0, 0):
            sns.jointplot(x=self.numerical_[1], y=self.numerical_[0], data=self._obj,
                          kind="kde")

    def lineplot(self):
        if self.config == (2, 0, 0):
            sns.relplot(x=self.numerical_[1], y=self.numerical_[
                        0], ci=None, kind="line", data=self._obj)
        elif self.config == (2, 1, 0):
            sns.relplot(x=self.numerical_[1], y=self.numerical_[0], hue=self.categorical_[0],
                        hue_order=self._obj[self.categorical_[
                            0]].cat.categories.tolist(),
                        ci=None, kind="line", data=self._obj)

    def hexbinplot(self):
        if self.config == (2, 0, 0):
            sns.jointplot(x=self.numerical_[0], y=self.numerical_[1], data=self._obj,
                          kind="hexbin")

    def boxplot(self, **kwargs):
        if self.config == (1, 0, 0):
            sns.boxplot(self.numerical_[0], data=self._obj, **kwargs)
        elif self.config == (1, 1, 0):
            sns.boxplot(x=self.categorical_[0], y=self.numerical_[
                        0], data=self._obj, **kwargs)
        elif self.config == (1, 1, 1):
            sns.boxplot(x=self.categorical_[0], y=self.numerical_[0], hue=self.categorical_[1],
                        data=self._obj, **kwargs)

    def violinplot(self):
        if self.config == (1, 0, 0):
            sns.violinplot(self.numerical_[0], data=self._obj)
        elif self.config == (1, 1, 0):
            sns.violinplot(x=self.categorical_[
                           0], y=self.numerical_[0], data=self._obj)
        elif self.config == (1, 1, 1):
            sns.violinplot(x=self.categorical_[0], y=self.numerical_[0], hue=self.categorical_[1],
                           data=self._obj, split=True)

    def stripplot(self):
        if self.config == (1, 0, 0):
            sns.stripplot(self.numerical_[0], data=self._obj)
        elif self.config == (1, 1, 0):
            sns.stripplot(x=self.categorical_[
                          0], y=self.numerical_[0], data=self._obj)
        elif self.config == (1, 1, 1):
            sns.stripplot(x=self.categorical_[0], y=self.numerical_[0], hue=self.categorical_[1],
                          data=self._obj, jitter=0.1)

    def qqplot(self):
        if self.config == (1, 0, 0):
            stats.probplot(self._obj[self.numerical_[0]], plot=sns.mpl.pyplot)
        else:
            raise ValueError

    def catplot(self):
        kwargs = dict(zip(["x", "hue", "col"], self.categorical_))
        if self.config == (0, 1, 2):
            sns.catplot(**kwargs, kind="count", data=self._obj)
        else:
            raise ValueError

    def ridgeplot(self):
        if self.config == (1, 1, 0):
            _ridgeplot(x=self.numerical_[0],
                       y=self.categorical_[0], data=self._obj)
        else:
            raise ValueError


def _ridgeplot(x, y, data):

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(data, row=y, hue=y, aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, x, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, x, clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, x)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
