# based on https://stackoverflow.com/a/27737385/16438094

from functools import wraps
from time import perf_counter
import numpy as np

all_timings = {}


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        l = all_timings.setdefault(f.__name__, [])
        l.append(te - ts)
        return result
    return wrap


def print_timing():
    for k, v in all_timings.items():
        print(
            "{} avg. exec time: {} (over {} calls)".format(
                k,
                np.mean(v),
                len(v)
            )
        )
