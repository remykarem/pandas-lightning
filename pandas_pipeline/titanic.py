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
    ("Cabin", "CabinType"): lambda s: s.str[0],
    ("Ticket", "HasLetters"):
    lambda s: s.str.startswith(tuple(string.ascii_letters)),
    ("CabinType", "HasCabinCode"): lambda s: ~s.isna()
})
