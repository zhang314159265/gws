import time
import functools

def f():
    print("This is f")

def deco(f=None, profile=False):
    if f:
        @functools.wraps(f)
        def f_wrapper(*args, **kwargs):
            if profile:
                start_ts = time.time()
    
            print("Before calling f")
            out = f()
            print("After calling f")
    
            if profile:
                elapse = time.time() - start_ts
                print(f"{elapse=:.3f}s")
            return out
        return f_wrapper
    else:
        return functools.partial(deco, profile=profile)

deco(f)()
deco(profile=True)(f)()
print(f"Func name {deco(f).__name__}")

def alt_deco(f=None, profile=False):
    """
    An alternative implementation similar to triton.jit
    """
    def inner_deco(f):
        @functools.wraps(f)
        def f_wrapper(*args, **kwargs):
            if profile:
                start_ts = time.time()
    
            print("(alt) Before calling f")
            out = f()
            print("(alt) After calling f")
    
            if profile:
                elapse = time.time() - start_ts
                print(f"(alt) {elapse=:.3f}s")
            return out
        return f_wrapper

    if f:
        return inner_deco(f)
    else:
        return inner_deco

alt_deco(f)()
alt_deco(profile=True)(f)()
