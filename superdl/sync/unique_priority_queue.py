import heapq
import threading

class UniquePriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_dict = {}
        self.lock = threading.Lock()

    def push(self, item, priority):
        with self.lock:
            if item in self.entry_dict:
                # Update priority if the new priority is higher
                if priority < self.entry_dict[item]:
                    entry = [priority, item]
                    self.entry_dict[item] = priority
                    heapq.heappushpop(self.heap, entry)
            else:
                entry = [priority, item]
                heapq.heappush(self.heap, entry)
                self.entry_dict[item] = priority

    def pop(self):
        with self.lock:
            while self.heap:
                priority, item = heapq.heappop(self.heap)
                if item is not None:
                    del self.entry_dict[item]
                    return item
            raise IndexError("pop from an empty UniquePriorityQueue")

    def is_empty(self):
        with self.lock:
            return not bool(self.heap)



# Usage example
if __name__ == "__main__":  
    # Example usage:
    unique_pq = UniquePriorityQueue()

    unique_pq.push("Task1", 3)
    unique_pq.push("Task2", 1)
    unique_pq.push("Task3", 2)
    unique_pq.push("Task1", 4)  # Updates priority of "Task1" to 4

    while not unique_pq.is_empty():
        print(unique_pq.pop())
