from itertools import combinations
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("optimize")
class optimize:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.history = None
        self._params = {}

    def drop_duplicate_columns(self, inplace: bool = False):
        """Drop duplicate columns that have exactly the same
        values and datatype

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform inplace operation, by default False

        Returns
        -------
        pandas.DataFrame
            Dataframe with no duplicate columns
        """
        to_drop = []
        pairs = list(combinations(self._obj.columns, 2))

        num_combinations = len(pairs)
        print(f"Checking {num_combinations} combinations")

        for pair in pairs:
            col_a, col_b = pair
            if col_a in to_drop or col_b in to_drop:
                continue
            if self._obj[col_a].equals(self._obj[col_b]):
                to_drop.append(col_b)

        print(f"Duplicate columns: {to_drop}")

        return self._obj.drop(columns=to_drop, inplace=inplace)

    def convert_categories(self, max_cardinality: int = 20,
                           inplace: bool = False):
        """Convert columns to category whenever possible

        Parameters
        ----------
        max_cardinality : int, optional
            The maximum no. of uniques before a column can be converted
            to a category type, by default 20
        inplace : bool, optional
            [description], by default False

        Returns
        -------
        pandas.DataFrame
            A transformed dataframe
        """
        if inplace:
            df = self._obj
        else:
            df = self._obj.copy()

        for col in df:
            if not df[col].dtype == "object":
                continue

            num_uniques = df[col].nunique()

            if num_uniques == 1:
                # Drop this column!
                print(f"{col} has only 1 value.")
            elif num_uniques <= max_cardinality:
                df[col] = df[col].astype("category")

        if inplace:
            return None
        else:
            return df

    def profile(self, dry_run=True, max_cardinality=20):

        befores = 0
        afters = 0

        for col in self._obj:

            num_uniques = self._obj[col].nunique()
            before = self._obj[col].memory_usage(index=False)
            befores += before

            if num_uniques == 1:
                print(f"{col} has only 1 value. Drop to save {before}.")
            elif num_uniques == 2:
                if self._obj[col].dtype.name in ["category", "bool"]:
                    print(f"{col} looks good")
                    continue
                after = self._obj[col].astype(
                    "category").memory_usage(index=False)
                afters += after
                savings = (before-after)/before * 100

                # if containts yes-no or true-false or t-f or 1-0

                print(f"{col} can be optimised. "
                      "Consider bool or cat. "
                      f"Consider category. Save {savings:.0f}%")
            elif num_uniques == 3:
                if self._obj[col].dtype.name in ["category", "bool"]:
                    print(f"{col} looks good")
                    continue
                after = self._obj[col].astype(
                    "category").memory_usage(index=False)
                afters += after
                savings = (before-after)/before * 100

            elif num_uniques <= max_cardinality:
                # print(self._obj[col].memory_usage(index=False))
                if self._obj[col].dtype == "object":
                    print(f"{col} can be optimised. "
                          f"{num_uniques} uniques found. "
                          f"Consider category.")
                elif self._obj[col].dtype.name.startswith("int"):
                    after = self._obj[col].astype(
                        "category").memory_usage(index=False)
                    afters += after
                    savings = (before-after)/before * 100
                    print(f"{col} can be optimised. "
                          f"{num_uniques} uniques found. "
                          f"Consider category. Save {savings:.0f}%")
                elif self._obj[col].dtype.name in ["category", "bool"]:
                    print(f"{col} looks good")
                    continue
                else:
                    print(f"{col} not evaluated")
            else:
                print(f"{col} looks good")

        # print(f"Before: {befores/1e6:.1f}MB")
        # print(f"Total savings: {(befores-afters)/1e6:.1f}MB")
        print(f"Total savings: {afters/1e6:.1f}MB")
