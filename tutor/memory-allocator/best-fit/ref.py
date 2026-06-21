"""A best-fit memory allocator over a fixed range of units.

The memory is `n` consecutive units indexed 0 .. n-1. Allocation uses the
best-fit strategy: among every maximal run of free units large enough to
satisfy the request, pick the one with the smallest length, breaking ties by
the lowest start index.
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

        # Best fit: smallest sufficient run, ties broken by lowest start.
        best = -1
        for i, (_, length) in enumerate(self._free):
            if length >= size and (best == -1 or length < self._free[best][1]):
                best = i
        if best == -1:
            return -1

        start = self._free[best][0]
        self._free[best][0] += size
        self._free[best][1] -= size
        if self._free[best][1] == 0:
            del self._free[best]

        self._allocated[start] = size
        return start

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
