import string

import pandas as pd
from pandas import CategoricalDtype
import pandas_pipeline

df = pd.read_csv(
    "/Users/raimibinkarim/Library/Mobile Documents/com~apple~CloudDocs/Datasets Tabular/titanic.csv")
df = df.pipeline.astype({
    "PassengerId": "index",
    "Pclass": CategoricalDtype([3, 2, 1], ordered=True),
    "Sex": "category",
    "Embarked": "category"
})
df = df.pipeline.map_numerical_binning({
    "Age": range(0, 100, 10),
    "Fare": ("quartile", 4)
}, ordered=True)
df = df.pipeline.sapply({
    ("HasDep", ("SibSp", "Parch")): lambda s, t: (s+t) > 0,
    ("HasLetters", "Ticket"):
    lambda s: s.str.startswith(tuple(string.ascii_letters)),
    ("CabinType", "Cabin"): lambda s: s.str[0],
    ("HasCabinCode", "CabinType"): lambda s: ~s.isna()
})
