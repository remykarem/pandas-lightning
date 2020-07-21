import pandas as pd
from pandas import CategoricalDtype



df = pd.read_csv("/Users/raimibinkarim/Library/Mobile Documents/com~apple~CloudDocs/Datasets Tabular/titanic.csv",
                 index_col="PassengerId")
df = df.pipeline.convert_dtypes({
    "Pclass": CategoricalDtype([3, 2, 1], ordered=True),
    "Sex": "category",
    "Embarked": "category"
})
df = df.pipeline.sapply({
    ("Cabin", "CabinType"): lambda s: s.str[0]
})
df = df.pipeline.compose(get_ticket_first_letters)


