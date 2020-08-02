from .dataframe_lambdas import lambdas


class Pipeline:
    def __init__(self):
        self._pipeline = []

    def __call__(self, pandas_obj):
        obj = self.run(pandas_obj)
        return obj

    def __repr__(self):
        return str([list(m.keys())[0] for m in self._pipeline])

    def add(self, obj):
        self._pipeline.append(obj)

    def run(self, pandas_obj):
        for pipe_item in self._pipeline:
            l = lambdas(pandas_obj)
            callable_str = list(pipe_item.keys())[0]
            fn = getattr(l, callable_str)
            mapping = list(pipe_item.values())[0]

            pandas_obj = fn(mapping)
        return pandas_obj

    def reset(self):
        self._pipeline.clear()
