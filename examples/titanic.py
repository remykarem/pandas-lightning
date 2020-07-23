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
df = df.pipeline.sapply({
    ("CabinType", "Cabin"): lambda s: s.str[0],
    ("HasLetters", "Ticket"):
    lambda s: s.str.startswith(tuple(string.ascii_letters)),
    ("HasCabinCode", "CabinType"): lambda s: ~s.isna()
})
