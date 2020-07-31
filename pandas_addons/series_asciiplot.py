from math import ceil, floor
import pandas as pd
from pandas.errors import EmptyDataError


@pd.api.extensions.register_series_accessor("asciiplot")
class asciiplot:
    def __init__(self, pandas_obj):
        self._validate_obj(pandas_obj)
        self._obj = pandas_obj

    def _validate_obj(self, _obj):
        if len(_obj) == 0:
            raise EmptyDataError("Series is empty")

    def hist(self,
             size: int = 10,
             hashes: int = 30,
             len_label: int = 10,
             max_categories: int = 20):
        """Plots a horizontal histogram using :code:`#`

        Parameters
        ----------
        size : int, optional
            Size of bins, by default 10
        hashes : int, optional
            Maximum number of hashes :code:`#` to display on the
            the label with the highest frequency, by default 30
        len_label : int, optional
            Maximum length of the text label, by default 10
        max_categories : int, optional
            Maximum number of categories to display, by default 50

        Notes
        -----
        This would be useful if you want to get a quick sense of
        the distribution of your data or if you do not have access
        to say a Jupyter notebook. The API is deliberately named after
        the standard library's :code:`.hist()` API.

        Examples
        --------
        >>> import pandas as pd
        >>> import pandas_addons
        >>> sr = pd.Series(["red", "blue", "red", "red", "orange", "blue"])
        >>> sr.ascii.hist()
               red ##############################
              blue ####################
            orange ##########
        """

        sort = True

        if self._obj.dtype.name.startswith("float"):
            min_val = (floor(min(self._obj)/10))*10
            max_val = (ceil(max(self._obj)/10))*10
            sr = pd.cut(self._obj, range(min_val, max_val, size))
            sort = False
        elif self._obj.dtype.name.startswith("int") or \
                (self._obj.dtype.name == "category" and getattr(self._obj.dtype, "ordered")):
            sort = False
            sr = self._obj
        else:
            sr = self._obj

        freqs = sr.value_counts(sort=sort).iloc[:max_categories]
        max_val = max(freqs)
        sr = freqs/max_val * hashes

        for label, count in sr.to_dict().items():
            label = str(label)
            if len(label) > len_label:
                label = label[:(len_label-3)] + "..."
            else:
                str_format = ">" + str(len_label)
                label = format(label, str_format)
            print(label, int(count)*"#")

