import pytest
import pandas as pd
from pandas import DataFrame  # typing
import pandas_lightning
import numpy as np


@pytest.fixture
def df() -> DataFrame:
    return pd.read_csv("./tests/titanic.csv")


@pytest.fixture
def df2() -> DataFrame:
    return pd.read_csv("./tests/titanic.csv")


def test_int_type(df: DataFrame, df2: DataFrame) -> None:
    df["Survived"] = df["Survived"].astype(int)
    df2 = df2.lambdas.astype(
        Survived=int
    )
    assert df.equals(df2)


def test_int_string(df: DataFrame, df2: DataFrame) -> None:

    dtypes = ["int", "int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64"]

    for dtype in dtypes:
        df["Survived"] = df["Survived"].astype(dtype)
        df2 = df2.lambdas.astype(
            Survived=dtype
        )
        assert df.equals(df2)


def test_int_nptype(df: DataFrame, df2: DataFrame) -> None:

    dtypes = [np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64]

    for dtype in dtypes:
        df["Survived"] = df["Survived"].astype(dtype)
        df2 = df2.lambdas.astype(
            Survived=dtype
        )
        assert df.equals(df2)
