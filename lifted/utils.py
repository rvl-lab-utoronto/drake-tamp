import heapq
import itertools


class PriorityQueue:
    def __init__(self, init=[]):
        self.heap = []
        self.counter = itertools.count()
        for item in init:
            self.push(item, 0)

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, next(self.counter), item))

    def pop(self):
        return heapq.heappop(self.heap)[-1]

    def peep(self):
        return self.heap[0][-1]

    def __len__(self):
        return len(self.heap)


class Identifiers:
    idx = 0

    @classmethod
    def next(cls):
        cls.idx += 1
        return f"?x{cls.idx}"


class Unsatisfiable(Exception):
    pass
