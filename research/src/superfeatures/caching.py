"""
Ported from `research/GA_test.ipynb` cell 14/16 — see `docs/RESTRUCTURING_TODO.md` and the
port plan in `docs/RESEARCH_STRUCTURE.md` / `docs/GA_structure.md`. Extracted mechanically
from the notebook cell source (not retyped) to avoid transcription drift; the notebook
remains the frozen parity reference. Only the import block was touched: names the cell relied
on getting from the shared notebook kernel namespace (cell 0's imports) are now imported
explicitly here, since a standalone module has no such shared kernel state.

Two LRU caches: `DataCache` (keyed by base feature name - persisted train/test Spark frames)
and `PredictionCache` (keyed by expression identity - scored RMSE). The sibling
`ResultsTracker` class cells 12/14/16 originally shared this file with now lives in
`reporting/history.py` (in-memory run-history tracking is a separate concern from caching).
"""

import threading

from cachetools import LRUCache
from pyspark import StorageLevel


class DataCache:
    """
    Lock guards every access to self.cache - evaluate_population (ga/engine.py) scores
    individuals via a ThreadPoolExecutor, so get()/store() are called concurrently from multiple
    worker threads. cachetools.LRUCache is not thread-safe on its own: two threads racing inside
    __setitem__'s eviction (or this class's own manual maxsize check below) can have one thread's
    popitem() name a victim key the other thread has already removed, raising KeyError - seen in
    practice under PredictionCache's identical pattern (see caching.py's git history/CLAUDE.md).
    """
    def __init__(self, maxsize=24):
        self.cache = LRUCache(maxsize=maxsize) #, callback=self._evict
        self._lock = threading.Lock()

    def _evict(self, key, value):
        try:
            for df in value:
                df.unpersist()
        except Exception as e:
            print(f"[UNPERSIST ERROR] {e}")

    def get(self, key):
        """
        Retrieve cached value if available.
        Returns: (hit: bool, value: tuple or None)
        """
        with self._lock:
            if key in self.cache:
                return True, self.cache[key]
            else:
                return False, None

    def store(self, key, data):
        """
        Cache the three DataFrames with proper .cache() and .count() calls.
        """
        #data = data.repartition(8)
        data.persist(StorageLevel.MEMORY_AND_DISK)
        data.take(1)

        with self._lock:
            # Manual eviction if full
            if len(self.cache) >= self.cache.maxsize:
                old_key, old_value = self.cache.popitem()
                self._evict(old_key, old_value)

            self.cache[key] = data

class LocalDataCache:
    """
    Pandas-DataFrame sibling of DataCache, for GeneticAlgorithm1's fit_backend="local" path (see
    ga/engine.py). Same get/store call shape as DataCache - (hit: bool, value) from get(), plain
    store(key, data) - so evaluate_fitness_static's leaf_fn needs no branching beyond which cache
    class was constructed. No .persist()/.take(1)/.unpersist() lifecycle at all: a pandas
    DataFrame is a plain in-process object, released by Python's own GC once the LRUCache drops
    the last reference - unlike DataCache's Spark frames, which need an explicit unpersist() to
    release executor-side cached blocks.

    Lock guards every access for the same reason as DataCache/PredictionCache - concurrent
    ThreadPoolExecutor workers, cachetools.LRUCache isn't thread-safe on its own.
    """
    def __init__(self, maxsize=24):
        self.cache = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                return True, self.cache[key]
            else:
                return False, None

    def store(self, key, data):
        with self._lock:
            self.cache[key] = data


class PredictionCache:
    """
    Lock guards every access to self.cache - see DataCache's docstring above for why. This is the
    cache where the race was actually observed: two ThreadPoolExecutor workers both inside
    store()'s self.cache[key] = rmse (cachetools' internal __setitem__ eviction) at once, one
    thread's popitem() naming a victim key the other had already evicted -> KeyError inside
    cachetools' own pop() call.
    """
    def __init__(self, maxsize=24):
        self.cache = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()

    def get(self, key):
        """
        Retrieve cached RMSE if available.
        Returns: (hit: bool, rmse: float or None)
        """
        with self._lock:
            if key in self.cache:
                return True, self.cache[key]
            else:
                return False, None

    def store(self, key, rmse):
        """
        Store the RMSE value in cache
        """
        with self._lock:
            self.cache[key] = rmse
