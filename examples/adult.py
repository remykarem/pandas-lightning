import numpy as np
import pandas as pd
import pandas_lightning

WORKING_CLASS = {
    "No income": ['Never-worked', 'Without-pay', 'Self-emp-not-inc'],
    "Private": ["Private"],
    "Govt": ['Local-gov', 'State-gov', 'Federal-gov']
}

EDUCATION = [
    'Preschool',
    '1st-4th',
    '5th-6th',
    '7th-8th',
    '9th',
    '10th',
    '11th',
    '12th',
    'HS-grad',
    'Assoc-acdm',
    'Assoc-voc',
    'Prof-school',
    'Some-college'
    'Bachelors',
    'Masters',
    'Doctorate',
]

df = pd.read_csv("/Users/raimibinkarim/Downloads/adult.csv")
df = df.optimize.convert_categories()
df = df.rename(columns={col: col.replace(".", "_") for col in df})
df = df.replace("?", np.nan)
df = df.cast(
    education_num="category"
)
df = df.add_columns(
    income=lambda s: s == ">50K",
    fnlwght=lambda s: s.scaler.standardize(),
)
df = df.lambdas.map_categorical_binning(ordered=True, binnings={
    "workclass": WORKING_CLASS
})
df = df.drop(columns=["education"])
