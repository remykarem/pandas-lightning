def fillna_bool(series):
    """
    {0,NaN,1} --(x2)--> {0,NaN,2}
    {0,NaN,2} --(-1)--> {-1,NaN,1}
    {-1,NaN,1} --(fill 0)--> {-1,0,1}
    """
    assert series.unique().tolist() == [0, 1]
    assert series.hasnans
    return (series.astype(float)*2-1).fillna(0).astype(int)
