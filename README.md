# pandas-pipeline (WIP)

Aggregate your pipeline

Firstl, import

```python
import pandas_pipeline
```

such that the `.pipeline` will be available to your DataFrame objects. 

* convert_types: convert the types of multiple columns
* applys: perform apply on multiple series objects
* sapplys: perform vectorised operations on multiple series
* maps: perform map on multiple columns

* ordinal encoding
* membership binning

```python
df = df.pipeline.convert_types({
})
```

```python
df = df.pipeline.applys({
})
```

```python
df = df.pipeline.sapplys({
})
```

```python
df = df.pipeline.maps({
})
```

```python
df = df.pipeline.fillnas({
})
```
