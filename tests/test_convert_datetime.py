import pandas as pd

from pandas_lightning.dataframe_cast import parse_datetime_target, cast_series_as_datetime


def test_parse_datetime_target():
    assert parse_datetime_target("datetime64[s]") == ('s', None, None)
    assert parse_datetime_target("datetime64[s]") == ('s', None, None)
    assert parse_datetime_target("datetime64[s, Asia/Singapore]") == ('s', 'Asia/Singapore', None)
    assert parse_datetime_target("datetime64[ms, Asia/Singapore]") == ('ms', 'Asia/Singapore', None)
    assert parse_datetime_target("datetime64[ms, ?, %Y%m|%d]") == ('ms', None, '%Y%m|%d')


def test_cast_series_as_datetime():
    cast_series_as_datetime(pd.Series([1658394599000, 1657969777000]), "datetime64[ns, Asia/Singapore]")
    cast_series_as_datetime(pd.Series([1658394599000, 1657969777000]), "datetime64[ns]")
    cast_series_as_datetime(pd.Series(["2022|10|01", "2022|12|03"]), "datetime64[s, ?, %Y|%m|%d]")
