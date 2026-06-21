from dataclasses import dataclass

@dataclass
class FreeNode:
    start: int
    size: int
    prev = None
    next = None

    @property
    def end(self):
        return self.start + self.size

class MemoryAllocator:
    def __init__(self, size):
        self.free_list = FreeNode(0, size)
        self.allocated = {}  # ptr to size

    def erase_node(self, node):
        if node.prev:
            node.prev.next = node.next
        else:
            self.free_list = node.next

        if node.next:
            node.next.prev = node.prev

    def allocate(self, size):
        if size <= 0:
            return -1
        cur = self.free_list
        while cur and cur.size < size:
            cur = cur.next
        if not cur:
            return -1

        ptr = cur.start
        self.allocated[ptr] = size

        if cur.size > size:
            cur.start += size
            cur.size -= size
        else:
            self.erase_node(cur)
        return ptr

    def insert_node_at(self, node, left, right):
        node.left = left
        node.right = right

        if left:
            left.next = node
        else:
            self.free_list = node

        if right:
            right.prev = node

    @staticmethod
    def merge(left, right):
        assert left.end == right.start
        return FreeNode(start=left.start, size=left.size + right.size)

    def free(self, ptr):
        if ptr not in self.allocated:
            return 0

        size = self.allocated.pop(ptr)

        newnode = FreeNode(start=ptr, size=size)
        if not self.free_list:
            self.free_list = newnode
            return size

        left, right = None, self.free_list
        while right and right.start <= ptr:
            left, right = right, right.next

        if left and left.end == newnode.start:
            newnode = self.merge(left, newnode)
            oldleft = left
            left = left.prev
            self.erase_node(oldleft)

        if right and newnode.end == right.start:
            newnode = self.merge(newnode, right)
            oldright = right
            right = right.next
            self.erase_node(oldright)

        self.insert_node_at(newnode, left, right)
        return size
