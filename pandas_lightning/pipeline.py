import copy
from .dataframe_lambdas import lambdas
from .dataframe_dataset import dataset


class Pipeline:
    API = 1

    def __init__(self, name=None):
        self.name = name
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

            accessor_str, fn_str = list(pipe_item.keys())[0]
            accessor_obj = eval(accessor_str)
            accessor = accessor_obj(pandas_obj)

            fn = getattr(accessor, fn_str)
            mapping = list(pipe_item.values())[0]
            if accessor_str == "lambdas":
                if fn_str in ["drop_columns_with_rules", "dapply"]:
                    pandas_obj = fn(*mapping)
                else:
                    pandas_obj = fn(mapping)
            elif accessor_str == "dataset":
                pandas_obj = fn(**mapping)
            else:
                raise RuntimeError

        return pandas_obj

    def reset(self):
        self._pipeline.clear()

    def copy(self):
        """Returns a copy of the current Pipeline object
        """
        pipeline = Pipeline()
        pipeline._pipeline = copy.deepcopy(self._pipeline)
        return pipeline
