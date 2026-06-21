from sortedcontainers import SortedList
from dataclasses import dataclass

@dataclass
class Node:
    size: int
    start: int

    @property
    def end(self):
        return self.start + self.size

    def __lt__(self, other):
        return self.size < other.size or (self.size == other.size and self.start < other.start)

class MemoryAllocator:
    def __init__(self, capacity):
        self.free_list = SortedList()
        self.start_to_node = {}
        self.end_to_node = {} # exclusive end
        self.ptr_to_size = {}
        self.add_block(Node(capacity, 0))

    def add_block(self, node):
        self.free_list.add(node)
        self.start_to_node[node.start] = node
        self.end_to_node[node.end] = node

    def remove_block(self, node, idx=None):
        if idx is None:
            idx = self.free_list.index(node)
        del self.free_list[idx]
        del self.start_to_node[node.start]
        del self.end_to_node[node.end]

    def _merge_node(self, left, right):
        assert left.end == right.start
        return Node(start=left.start, size=left.size + right.size)

    def free(self, ptr):
        if ptr not in self.ptr_to_size:
            return 0
        size = self.ptr_to_size[ptr]
        del self.ptr_to_size[ptr]

        new_node = Node(start=ptr, size=size)

        # merge with left
        left_node = self.end_to_node.get(new_node.start, None)
        if left_node:
            new_node = self._merge_node(left_node, new_node)
            self.remove_block(left_node)

        # merge with right
        right_node = self.start_to_node.get(new_node.end, None)
        if right_node:
            new_node = self._merge_node(new_node, right_node)
            self.remove_block(right_node)

        # add new_node
        self.add_block(new_node)

        return size

    def allocate(self, size):
        if size <= 0:
            return -1
        search_node = Node(size, 0)
        found_idx = self.free_list.bisect_left(search_node)
        if found_idx == len(self.free_list):
            return -1

        fit_node = self.free_list[found_idx]
        self.remove_block(fit_node, found_idx)
        self.ptr_to_size[fit_node.start] = size

        if fit_node.size > size:
            self.add_block(Node(fit_node.size - size, fit_node.start + size))

        return fit_node.start
