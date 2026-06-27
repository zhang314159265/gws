import time

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class Work:
    def __init__(self, id):
        self.id = id

    def run(self):
        time.sleep(10)
        return self.id

workset = [Work(i) for i in range(10)]

EXPECT = 45

def sequential_run():
    act = 0
    start = time.time()
    for w in workset:
        act += w.run()
    end = time.time()

    assert act == EXPECT
    print(f"sequential_run takes {end - start:.3f} seconds")

def pool_run(use_threadpool, use_map=True):
    start = time.time()
    pooltype = ThreadPoolExecutor if use_threadpool else ProcessPoolExecutor
    pool = pooltype(20)

    print(f"{use_map=}")
    if use_map:
        out = sum(pool.map(lambda w: w.run(), workset))
    else:
        future_list = [pool.submit(w.run) for w in workset]
        out = 0
        for future in future_list:
            out += future.result()
    end = time.time()

    assert out == EXPECT
    print(f"{pooltype.__name__} takes {end - start:.3f} seconds")


def threadpool_run():
    pool_run(True)

def processpool_run():
    pool_run(False)

# sequential_run()
threadpool_run()
# processpool_run()
