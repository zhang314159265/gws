"""A first-fit memory allocator over a fixed range of units.

The memory is `n` consecutive units indexed 0 .. n-1. Allocation uses the
first-fit strategy: scanning the free runs in increasing start order, pick the
first run large enough to satisfy the request, regardless of its size.
"""


class MemoryAllocator:
    def __init__(self, n):
        # Free space as a list of [start, length] runs, kept sorted by start
        # and always coalesced (no two runs are adjacent or overlapping).
        self._free = [[0, n]] if n > 0 else []
        # Maps the start index of each live allocation to its length.
        self._allocated = {}

    def allocate(self, size):
        if size <= 0:
            return -1

        # First fit: the first run (lowest start) that is large enough.
        for i, (_, length) in enumerate(self._free):
            if length >= size:
                start = self._free[i][0]
                self._free[i][0] += size
                self._free[i][1] -= size
                if self._free[i][1] == 0:
                    del self._free[i]
                self._allocated[start] = size
                return start
        return -1

    def free(self, addr):
        size = self._allocated.pop(addr, None)
        if size is None:
            return 0
        self._insert_free(addr, size)
        return size

    def _insert_free(self, start, length):
        """Insert a free run, keeping the list sorted and coalesced."""
        i = 0
        while i < len(self._free) and self._free[i][0] < start:
            i += 1
        self._free.insert(i, [start, length])

        # Coalesce with the following run, then the preceding one.
        if i + 1 < len(self._free) and self._free[i][0] + self._free[i][1] == self._free[i + 1][0]:
            self._free[i][1] += self._free[i + 1][1]
            del self._free[i + 1]
        if i > 0 and self._free[i - 1][0] + self._free[i - 1][1] == self._free[i][0]:
            self._free[i - 1][1] += self._free[i][1]
            del self._free[i]
