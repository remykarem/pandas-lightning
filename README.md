# pandas-pipeline (WIP)

Organize your pipelines

Install

```bash
pip install git+https://github.com/remykarem/pandas-pipeline#egg=pandas-pipeline
```

Firstl, import

```python
import pandas_pipeline
```

such that the `.pipeline` will be available to your DataFrame objects. 

* convert_dtypes: convert the types of multiple columns
* apply: perform apply on multiple series objects
* sapply: perform vectorised operations on multiple series
* map: perform map on multiple columns

* ordinal encoding
* membership binning
* drop duplicate columns

```python
df = df.pipeline.convert_dtypes({
})
```

```python
df = df.pipeline.apply({
})
```

```python
df = df.pipeline.sapply({
})
```

```python
df = df.pipeline.map({
})
```

```python
df = df.pipeline.fillna({
})
```
