from . import dataframe_dataset, dataframe_lambdas, dataframe_optimize, dataframe_cast, dataframe_transform_columns
from . import dataframe_to_X_y, dataframe_get_column_enum
from . import dataframe_quickplot, dataframe_tests
from . import series_asciiplot, series_datetime
from . import series_map_categorical_binning, series_map_numerical_binning
from . import series_pctg, series_scaler, series_tests
from .series_scaler import standardize
from ._utils import makeDataFrame
import seaborn as sns

__version__ = "0.0.1"

sns.set_theme()
