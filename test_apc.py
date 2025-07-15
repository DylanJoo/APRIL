import time
from functools import wraps

def trace_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__qualname__} took {end - start:.6f} seconds")
        return result
    return wrapper

class MyClass:
    @trace_time
    def slow_function(self):
        time.sleep(2)  # Simulating work

obj = MyClass()
obj.slow_function()
obj.slow_function()
obj.slow_function()

